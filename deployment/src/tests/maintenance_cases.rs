use super::*;
use crate::{offline, retention};
#[cfg(target_os = "linux")]
#[test]
fn offline_extraction_verifies_every_file_and_platform_before_installation() {
    let t = tempfile::tempdir().unwrap();
    let source = t.path().join("source");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("Cargo.lock"), "locked").unwrap();
    fs::write(
        source.join("rust-toolchain.toml"),
        "[toolchain]\nchannel = \"1.96.0\"\n",
    )
    .unwrap();
    for attack in [
        "valid",
        "wrong-host",
        "wrong-lock",
        "wrong-profile",
        "duplicate",
        "missing",
        "tampered",
        "symlink",
    ] {
        let mut m = offline_manifest();
        m["host"] = offline::host().unwrap();
        m["cargo_lock_sha256"] = json!(hash(source.join("Cargo.lock")).unwrap());
        for (name, record) in m["files"].as_object_mut().unwrap() {
            let data = if name == "Cargo.lock" {
                b"locked".as_slice()
            } else {
                b"input".as_slice()
            };
            record["size"] = json!(data.len());
            record["sha256"] = json!(hash_bytes(data));
        }
        if attack == "wrong-host" {
            m["host"]["architecture"] = json!(if std::env::consts::ARCH == "aarch64" {
                "x86_64"
            } else {
                "aarch64"
            });
        }
        if attack == "wrong-lock" {
            m["cargo_lock_sha256"] = json!("0".repeat(64));
        }
        if attack == "wrong-profile" {
            m["profile"] = json!("gpu");
        }
        let archive = t.path().join(format!("{attack}.tar.gz"));
        let mut tar = tar::Builder::new(flate2::write::GzEncoder::new(
            fs::File::create(&archive).unwrap(),
            flate2::Compression::default(),
        ));
        let bytes = serde_json::to_vec(&m).unwrap();
        let mut h = tar::Header::new_ustar();
        h.set_mode(0o644);
        h.set_size(bytes.len() as u64);
        h.set_cksum();
        tar.append_data(&mut h, "offline-manifest.json", bytes.as_slice())
            .unwrap();
        for (name, record) in m["files"].as_object().unwrap() {
            if attack == "missing" && name == "rustup" {
                continue;
            }
            let data = if name == "Cargo.lock" {
                b"locked".as_slice()
            } else if attack == "tampered" && name == "rustup" {
                b"wrong".as_slice()
            } else {
                b"input".as_slice()
            };
            let mut h = tar::Header::new_ustar();
            h.set_mode(0o644);
            h.set_size(record["size"].as_u64().unwrap());
            if attack == "symlink" && name == "rustup" {
                h.set_entry_type(tar::EntryType::Symlink);
                h.set_link_name("/etc/passwd").unwrap();
            }
            h.set_cksum();
            tar.append_data(&mut h, name, data).unwrap();
            if attack == "duplicate" && name == "rustup" {
                tar.append_data(&mut h, name, data).unwrap();
            }
        }
        tar.into_inner().unwrap().finish().unwrap();
        let result = offline::extract(
            &archive,
            &hash(&archive).unwrap(),
            &t.path().join(format!("out-{attack}")),
            &source,
            "cpu",
        );
        assert_eq!(result.is_ok(), attack == "valid", "{attack}: {result:?}");
    }
}
fn old(p: &Path) {
    filetime::set_file_mtime(p, filetime::FileTime::from_unix_time(1, 0)).unwrap();
}
fn release(root: &Path, id: &str) {
    let p = root.join("releases").join(id);
    fs::create_dir_all(&p).unwrap();
    fs::write(p.join("payload"), "data").unwrap();
    old(&p);
}
fn job(root: &Path, id: &str, state: &str) {
    let p = root.join("jobs").join(format!("{id}.json"));
    atomic(&p, &json!({"state":state,"release_id":id})).unwrap();
    old(&p);
}
#[test]
fn retention_preserves_selected_rollback_newest_and_nonterminal_work() {
    let t = tempfile::tempdir().unwrap();
    for n in ["active", "previous", "pending", "garbage", "newest"] {
        release(t.path(), n);
    }
    atomic(
        t.path().join("selection.json"),
        &json!({"kind":"release","release_id":"active"}),
    )
    .unwrap();
    atomic(
        t.path().join("previous.json"),
        &json!({"kind":"release","release_id":"previous"}),
    )
    .unwrap();
    job(t.path(), "pending", "building");
    job(t.path(), "garbage", "failed");
    filetime::set_file_mtime(t.path().join("releases/newest"), filetime::FileTime::now()).unwrap();
    let lock = t.path().join("lock");
    let dry = retention::cleanup_with_lock(t.path(), 1, 14, false, &lock).unwrap();
    assert!(dry["paths"]
        .as_array()
        .unwrap()
        .contains(&json!("releases/garbage")));
    assert!(t.path().join("releases/garbage").exists());
    retention::cleanup_with_lock(t.path(), 1, 14, true, &lock).unwrap();
    assert!(!t.path().join("releases/garbage").exists());
    for name in ["active", "previous", "pending", "newest"] {
        assert!(t.path().join("releases").join(name).exists());
    }
}
#[test]
fn retention_refuses_pending_recovery_or_invalid_policy() {
    let t = tempfile::tempdir().unwrap();
    let lock = t.path().join("lock");
    assert!(retention::cleanup_with_lock(t.path(), 0, 1, true, &lock).is_err());
    assert!(retention::cleanup_with_lock(t.path(), 1, 0, true, &lock).is_err());
    atomic(
        t.path().join("activation.json"),
        &json!({"phase":"verifying"}),
    )
    .unwrap();
    assert!(retention::cleanup_with_lock(t.path(), 1, 14, true, &lock).is_err());
}
#[test]
fn retention_does_not_remove_unknown_uploads_or_builds() {
    let t = tempfile::tempdir().unwrap();
    for group in ["incoming", "staging", "initialization"] {
        let p = t.path().join(group).join("unknown");
        fs::create_dir_all(&p).unwrap();
        old(&p);
    }
    let v = retention::cleanup_with_lock(t.path(), 1, 1, true, &t.path().join("lock")).unwrap();
    assert!(v["paths"].as_array().unwrap().is_empty());
}
#[test]
fn retention_lock_prevents_racing_deployment() {
    let t = tempfile::tempdir().unwrap();
    let lock = t.path().join("lock");
    let _guard = Lock::acquire(&lock).unwrap();
    assert!(retention::cleanup_with_lock(t.path(), 1, 14, true, &lock).is_err());
}
#[cfg(unix)]
#[test]
fn retention_rejects_symlinks_before_any_deletion() {
    let t = tempfile::tempdir().unwrap();
    release(t.path(), "old");
    release(t.path(), "newest");
    std::os::unix::fs::symlink("/tmp", t.path().join("releases/old/escape")).unwrap();
    filetime::set_file_mtime(t.path().join("releases/newest"), filetime::FileTime::now()).unwrap();
    old(&t.path().join("releases/old"));
    assert!(retention::cleanup_with_lock(t.path(), 1, 14, true, &t.path().join("lock")).is_err());
    assert!(t.path().join("releases/old").exists());
}
fn offline_manifest() -> Value {
    let mut files = serde_json::Map::new();
    for name in [
        "rustup",
        "Packages",
        "Cargo.lock",
        "toolchain/bin/rustc",
        "toolchain/bin/cargo",
        "packages/build.deb",
    ] {
        files.insert(
            name.into(),
            json!({"size":1,"sha256":"a".repeat(64),"mode":0o644}),
        );
    }
    json!({"schema_version":1,"kind":"vortex-offline","host":{"os":"ubuntu","ubuntu":"22.04","architecture":"aarch64"},"profile":"cpu","toolchain":"1.96.0","cargo_lock_sha256":"b".repeat(64),"files":files})
}
#[test]
fn offline_manifest_requires_complete_verified_inputs() {
    let m = offline_manifest();
    offline::manifest_valid(&m).unwrap();
    for name in [
        "rustup",
        "Cargo.lock",
        "toolchain/bin/rustc",
        "toolchain/bin/cargo",
        "packages/build.deb",
    ] {
        let mut bad = m.clone();
        bad["files"].as_object_mut().unwrap().remove(name);
        assert!(offline::manifest_valid(&bad).is_err(), "{name}");
    }
}
#[test]
fn offline_manifest_rejects_unsafe_paths_credentials_and_metadata() {
    let m = offline_manifest();
    for name in [
        "../outside",
        "/etc/passwd",
        "cargo/credentials.toml",
        "cargo/config.toml",
        "packages/../escape.deb",
        "toolchain//file",
        "toolchain/../../escape",
        "packages/a.sh",
    ] {
        let mut v = m.clone();
        v["files"][name] = json!({"size":1,"sha256":"a".repeat(64),"mode":0o644});
        assert!(offline::manifest_valid(&v).is_err(), "{name}");
    }
    for key in [
        "schema_version",
        "kind",
        "profile",
        "toolchain",
        "cargo_lock_sha256",
    ] {
        let mut v = m.clone();
        v[key] = json!("invalid");
        assert!(offline::manifest_valid(&v).is_err(), "{key}");
    }
}
#[test]
fn offline_manifest_rejects_overflows_modes_and_bad_host() {
    let m = offline_manifest();
    for (key, value) in [
        ("size", json!(u64::MAX)),
        ("size", json!(-1)),
        ("mode", json!(0o777)),
        ("sha256", json!("z".repeat(64))),
    ] {
        let mut v = m.clone();
        v["files"]["rustup"][key] = value;
        assert!(offline::manifest_valid(&v).is_err());
    }
    for key in ["os", "ubuntu", "architecture"] {
        let mut v = m.clone();
        v["host"][key] = json!("unknown");
        assert!(offline::manifest_valid(&v).is_err());
    }
}
