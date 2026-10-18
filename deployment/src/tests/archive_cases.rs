use super::*;
use std::io::Read;
#[test]
fn packaging_cannot_write_into_its_source_root() {
    let t = tempfile::tempdir().unwrap();
    let error = package::pack(
        t.path(),
        t.path(),
        Path::new("unused"),
        None,
        None,
        Path::new("unused"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("Separate source and output"));
    assert_eq!(fs::read_dir(t.path()).unwrap().count(), 0);
}
#[test]
fn legacy_python_manifests_remain_verified_rollback_targets() {
    let (t, mut manifest, sel) = staged();
    let directory = t
        .path()
        .join("releases")
        .join(string(&sel, "release_id").unwrap());
    manifest["files"]
        .as_object_mut()
        .unwrap()
        .remove("tools/vortexctl");
    fs::remove_file(directory.join("tools/vortexctl")).unwrap();
    for name in [
        "scripts/run_vortex.py",
        "scripts/deploy_release.py",
        "scripts/activate_release.py",
        "scripts/initialize_jetson.py",
    ] {
        let bytes = b"# historical controller, never executed by vortexctl\n";
        fs::create_dir_all(directory.join("scripts")).unwrap();
        fs::write(directory.join(name), bytes).unwrap();
        manifest["files"][name] =
            json!({"size":bytes.len(),"sha256":hash_bytes(bytes),"mode":0o644});
    }
    manifest["snapshot_sha256"] = json!(hash_bytes(
        package::canonical(&manifest["files"]).as_bytes()
    ));
    package::validate_manifest(&manifest, string(&sel, "release_id").unwrap()).unwrap();
    atomic(directory.join("manifest.json"), &manifest).unwrap();
    activation::ready(t.path(), string(&sel, "release_id").unwrap()).unwrap();
    package::verify(&directory, &manifest).unwrap();
}
#[test]
fn hostile_archives_cannot_extract_or_escape() {
    let (t, p) = fixture();
    let original = t.path().join("output/release.tar.gz");
    let mut source = tar::Archive::new(flate2::read::GzDecoder::new(
        fs::File::open(original).unwrap(),
    ));
    let entries: Vec<_> = source
        .entries()
        .unwrap()
        .map(|entry| {
            let mut entry = entry.unwrap();
            let name = entry.path().unwrap().to_string_lossy().to_string();
            let header = entry.header().clone();
            let mut data = Vec::new();
            entry.read_to_end(&mut data).unwrap();
            (name, header, data)
        })
        .collect();
    for attack in [
        "duplicate",
        "missing",
        "tampered",
        "mode",
        "symlink",
        "sparse",
        "traversal",
        "unexpected",
        "manifest_missing",
        "oversized_manifest",
    ] {
        let file = t.path().join(format!("{attack}.tar.gz"));
        let mut tar = tar::Builder::new(flate2::write::GzEncoder::new(
            fs::File::create(&file).unwrap(),
            flate2::Compression::default(),
        ));
        for (i, (name, header, data)) in entries.iter().enumerate() {
            if (attack == "missing" && i == 1) || (attack == "manifest_missing" && i == 0) {
                continue;
            }
            let mut h = header.clone();
            let mut data = data.clone();
            if i == 0 && attack == "oversized_manifest" {
                h.set_size(3 * 1024 * 1024);
                data = vec![b' '; 3 * 1024 * 1024];
            }
            if i == 1 {
                match attack {
                    "tampered" => {
                        data[0] ^= 1;
                    }
                    "mode" => h.set_mode(0o777),
                    "symlink" => {
                        h.set_entry_type(tar::EntryType::Symlink);
                        h.set_link_name("/tmp/outside").unwrap();
                    }
                    "sparse" => h.set_entry_type(tar::EntryType::GNUSparse),
                    "traversal" | "unexpected" => {
                        let n = if attack == "traversal" {
                            "../outside"
                        } else {
                            "config/password.json"
                        };
                        h.as_mut_bytes()[..100].fill(0);
                        h.as_mut_bytes()[..n.len()].copy_from_slice(n.as_bytes());
                    }
                    _ => {}
                }
            }
            h.set_cksum();
            tar.append(&h, data.as_slice()).unwrap();
            if attack == "duplicate" && i == 1 {
                tar.append(&h, data.as_slice()).unwrap();
            }
            let _ = name;
        }
        tar.into_inner().unwrap().finish().unwrap();
        let dest = t.path().join(format!("extract-{attack}"));
        assert!(
            package::extract(&file, &dest, string(&p, "release_id").unwrap()).is_err(),
            "Accepted {attack}"
        );
        assert!(!t.path().join("outside").exists());
    }
}
#[test]
fn manifest_sizes_and_required_files_are_checked_before_extraction() {
    let (t, p) = fixture();
    let m = package::extract(
        &t.path().join("output/release.tar.gz"),
        &t.path().join("base"),
        string(&p, "release_id").unwrap(),
    )
    .unwrap();
    for attack in ["negative", "overflow", "mode", "checksum", "required"] {
        let mut v = m.clone();
        let files = v["files"].as_object_mut().unwrap();
        match attack {
            "negative" => files.get_mut("Cargo.toml").unwrap()["size"] = json!(-1),
            "overflow" => files.get_mut("Cargo.toml").unwrap()["size"] = json!(u64::MAX),
            "mode" => files.get_mut("Cargo.toml").unwrap()["mode"] = json!(0o777),
            "checksum" => files.get_mut("Cargo.toml").unwrap()["sha256"] = json!("z".repeat(64)),
            "required" => {
                files.remove("Cargo.lock");
            }
            _ => unreachable!(),
        }
        v["snapshot_sha256"] = json!(hash_bytes(package::canonical(&v["files"]).as_bytes()));
        assert!(
            package::validate_manifest(&v, string(&p, "release_id").unwrap()).is_err(),
            "{attack}"
        );
    }
}
#[test]
fn canonical_manifest_preserves_python_unicode_escaping() {
    assert_eq!(
        package::canonical(&json!({"src/é.rs":"🤖"})),
        r#"{"src/\u00e9.rs": "\ud83e\udd16"}"#
    );
}
