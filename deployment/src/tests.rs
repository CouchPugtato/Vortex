use crate::{
    activation::{self, Manager},
    common::*,
    initialize, launcher, package, service,
};
use anyhow::Result;
use serde_json::{json, Value};
use std::{
    cell::Cell,
    fs,
    path::{Path, PathBuf},
};
mod archive_cases;
mod diagnostics_cases;
mod maintenance_cases;
fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .into()
}
fn device() -> Value {
    read(repository().join("config/device.json")).unwrap()
}
fn elf(path: &Path) {
    let mut bytes = vec![0u8; 64];
    bytes[..6].copy_from_slice(b"\x7fELF\x02\x01");
    bytes[18] = 183;
    fs::write(path, bytes).unwrap();
}
fn fixture() -> (tempfile::TempDir, Value) {
    let t = tempfile::tempdir().unwrap();
    let out = t.path().join("output");
    fs::create_dir(&out).unwrap();
    let controller = t.path().join("controller");
    elf(&controller);
    let dev = t.path().join("device.json");
    let mut c = device();
    c["build_profile"] = json!("cpu");
    atomic(&dev, &c).unwrap();
    let m = package::pack(&repository(), &out, &dev, None, None, &controller).unwrap();
    (t, m)
}
fn staged() -> (tempfile::TempDir, Value, Value) {
    let (t, p) = fixture();
    let identity = string(&p, "release_id").unwrap();
    fs::create_dir(t.path().join("releases")).unwrap();
    let dir = t.path().join("releases").join(identity);
    let manifest =
        package::extract(&t.path().join("output/release.tar.gz"), &dir, identity).unwrap();
    atomic(
        dir.join("release.json"),
        &json!({"release_id":identity,"profile":"cpu"}),
    )
    .unwrap();
    let sel = json!({"kind":"release","release_id":identity,"launch_token":"token"});
    (t, manifest, sel)
}
struct Fake {
    active: Cell<bool>,
    starts: Cell<u32>,
    stops: Cell<u32>,
    fail_stop: Cell<bool>,
}
impl Fake {
    fn new() -> Self {
        Self {
            active: Cell::new(false),
            starts: Cell::new(0),
            stops: Cell::new(0),
            fail_stop: Cell::new(false),
        }
    }
}
impl Manager for Fake {
    fn info(&self) -> Result<Value> {
        Ok(json!({"active":self.active.get(),"pid":if self.active.get(){123}else{0}}))
    }
    fn stop(&self) -> Result<()> {
        self.stops.set(self.stops.get() + 1);
        anyhow::ensure!(!self.fail_stop.get(), "Stop failed");
        self.active.set(false);
        Ok(())
    }
    fn start(&self) -> Result<()> {
        self.starts.set(self.starts.get() + 1);
        self.active.set(true);
        Ok(())
    }
}

#[test]
fn device_validation_accepts_all_profiles() {
    for p in ["cpu", "gpu", "tensorrt", "gpu-tensorrt"] {
        let mut c = device();
        c["build_profile"] = json!(p);
        launcher::config(&c).unwrap();
    }
}
#[test]
fn device_validation_rejects_bad_cameras() {
    for cams in [
        json!([]),
        json!([0, 0]),
        json!([-1]),
        json!([64]),
        json!([1.0]),
        json!([true]),
        json!(["0"]),
        Value::Null,
    ] {
        let mut c = device();
        c["cameras"] = cams;
        assert!(launcher::config(&c).is_err());
    }
}
#[test]
fn device_validation_rejects_bad_values() {
    for (key, bad) in [
        ("schema_version", json!(true)),
        ("schema_version", json!(2)),
        ("runtime", json!("bridge")),
        ("build_profile", json!("automatic")),
        ("runtime_config", json!("\n")),
        ("tag_map", Value::Null),
    ] {
        let mut c = device();
        c[key] = bad;
        assert!(launcher::config(&c).is_err(), "{key}");
    }
}
#[test]
fn networking_validation_rejects_bad_values() {
    for (section, key, bad) in [
        ("networktables", "enabled", json!(1)),
        ("networktables", "team", json!(0)),
        ("networktables", "team", json!(100000)),
        ("networktables", "mode", json!("remote")),
        ("networktables", "table", json!("Vortex")),
        ("udp", "port", json!(0)),
        ("udp", "port", json!(65536)),
        ("udp", "enabled", json!("true")),
    ] {
        let mut c = device();
        c[section][key] = bad;
        assert!(launcher::config(&c).is_err(), "{section}.{key}");
    }
}
#[test]
fn all_resource_profiles_validate() {
    let c = device();
    launcher::resources(&repository(), &c).unwrap();
}
#[test]
fn identifiers_reject_path_and_shell_syntax() {
    for s in ["", "../x", "/x", "x y", "x;id", "-x", "a.b", "a\nx", "a\\b"] {
        assert!(id(s).is_err());
    }
    assert!(id(&"x".repeat(97)).is_err());
    for s in ["a", "x_1-2"] {
        id(s).unwrap();
    }
}
#[test]
fn checksums_are_exact_lowercase_hex() {
    checksum(&"a".repeat(64)).unwrap();
    for s in [
        "a".repeat(63),
        "a".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
    ] {
        assert!(checksum(&s).is_err());
    }
}
#[test]
fn archive_paths_are_allowlisted() {
    for s in [
        "src/main.rs",
        "src/nested/a.cu",
        "Cargo.lock",
        "tools/vortexctl",
    ] {
        package::payload(s).unwrap();
    }
    for s in [
        "../src/a.rs",
        "/src/a.rs",
        "src//a.rs",
        "src/./a.rs",
        "src/../a.rs",
        "src/.git/a.rs",
        "src/target/a.rs",
        "src/a.py",
        "src\\a.rs",
        "src/a.rs:stream",
        "src/a\n.rs",
        "config/password.json",
        "README.md",
    ] {
        assert!(package::payload(s).is_err(), "{s}");
    }
}
#[test]
fn canonical_format_matches_legacy_python() {
    assert_eq!(
        package::canonical(&json!({"z":[1,true],"a":{"b":3}})),
        r#"{"a": {"b": 3}, "z": [1, true]}"#
    );
}
#[test]
fn atomic_json_replaces_without_partial_content() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("nested/data.json");
    for i in 0..20 {
        atomic(&p, &json!({"value":i})).unwrap();
        assert_eq!(read(&p).unwrap()["value"], i);
    }
    assert_eq!(fs::read_dir(p.parent().unwrap()).unwrap().count(), 1);
}
#[test]
fn exclusive_lock_cannot_be_reentered() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("lock");
    let l = Lock::acquire(&p).unwrap();
    assert!(Lock::acquire(&p).is_err());
    drop(l);
    Lock::acquire(&p).unwrap();
}
#[test]
fn job_claim_prevents_duplicate_workers() {
    let t = tempfile::tempdir().unwrap();
    claim(t.path(), "job", "claim").unwrap();
    assert!(claim(t.path(), "job", "claim").is_err());
}
#[test]
fn elf_rejects_wrong_architecture_and_format() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("binary");
    elf(&p);
    assert_eq!(package::elf(&p, Some("aarch64")).unwrap(), "aarch64");
    assert!(package::elf(&p, Some("x86_64")).is_err());
    fs::write(&p, b"not an executable").unwrap();
    assert!(package::elf(&p, None).is_err());
}
#[test]
fn package_roundtrip_has_no_python_deployment_payload() {
    let (t, p) = fixture();
    let dest = t.path().join("extracted");
    let m = package::extract(
        &t.path().join("output/release.tar.gz"),
        &dest,
        string(&p, "release_id").unwrap(),
    )
    .unwrap();
    package::verify(&dest, &m).unwrap();
    assert!(!m["files"]
        .as_object()
        .unwrap()
        .keys()
        .any(|s| s.ends_with(".py")));
    assert!(dest.join("tools/vortexctl").exists());
    assert!(!dest.join("deployment").exists());
    assert_eq!(p["controller_architecture"], "aarch64");
}
#[test]
fn package_detects_payload_tampering() {
    let (t, m, s) = staged();
    let dir = t
        .path()
        .join("releases")
        .join(string(&s, "release_id").unwrap());
    fs::write(dir.join("src/main.rs"), "tampered").unwrap();
    assert!(package::verify(&dir, &m).is_err());
}
#[test]
fn manifest_rejects_identity_profile_and_checksum_changes() {
    let (_, p) = fixture();
    let t = fixture();
    let dir = t.0.path().join("m");
    let m = package::extract(
        &t.0.path().join("output/release.tar.gz"),
        &dir,
        string(&t.1, "release_id").unwrap(),
    )
    .unwrap();
    for key in [
        "schema_version",
        "release_id",
        "build_profile",
        "features",
        "snapshot_sha256",
    ] {
        let mut changed = m.clone();
        changed[key] = json!("bad");
        assert!(
            package::validate_manifest(&changed, string(&t.1, "release_id").unwrap()).is_err(),
            "{key}"
        );
    }
    assert!(package::validate_manifest(&m, string(&p, "release_id").unwrap()).is_err());
}
#[test]
fn extraction_rejects_wrong_release_and_existing_destination() {
    let (t, p) = fixture();
    let tar = t.path().join("output/release.tar.gz");
    assert!(package::extract(&tar, &t.path().join("wrong"), "other").is_err());
    assert!(!t.path().join("wrong").exists());
    let dest = t.path().join("existing");
    fs::create_dir(&dest).unwrap();
    assert!(package::extract(&tar, &dest, string(&p, "release_id").unwrap()).is_err());
}
#[test]
fn source_fingerprint_detects_changes_and_ignores_resources() {
    let t = tempfile::tempdir().unwrap();
    fs::create_dir(t.path().join("src")).unwrap();
    for n in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "rust-toolchain.toml",
        "src/a.rs",
    ] {
        fs::write(t.path().join(n), n).unwrap();
    }
    let before = launcher::source_digest(t.path()).unwrap();
    fs::write(t.path().join("src/a.png"), "image").unwrap();
    assert_eq!(before, launcher::source_digest(t.path()).unwrap());
    fs::write(t.path().join("src/a.rs"), "changed").unwrap();
    assert_ne!(before, launcher::source_digest(t.path()).unwrap());
}
#[test]
fn build_stamp_rejects_stale_source_or_binary() {
    let (t, _, s) = staged();
    let dir = t
        .path()
        .join("releases")
        .join(string(&s, "release_id").unwrap());
    let c = read(dir.join("config/device.json")).unwrap();
    let binary = launcher::binary(&dir, launcher::MAIN);
    fs::create_dir_all(binary.parent().unwrap()).unwrap();
    fs::write(&binary, "binary").unwrap();
    mode(&binary, 0o755).unwrap();
    let stamp = json!({"schema_version":1,"runtime":"main","profile":"cpu","source_sha256":launcher::source_digest(&dir).unwrap(),"binaries":{launcher::MAIN:hash(&binary).unwrap()}});
    atomic(dir.join("target/release/vortex-build.json"), &stamp).unwrap();
    launcher::checked_binary(&dir, &c, launcher::MAIN).unwrap();
    fs::write(&binary, "modified").unwrap();
    assert!(launcher::checked_binary(&dir, &c, launcher::MAIN).is_err());
}
#[test]
fn health_rejects_stale_identity_and_pipeline_failures() {
    let (t, _, s) = staged();
    let dir = t
        .path()
        .join("releases")
        .join(string(&s, "release_id").unwrap());
    let mut c = read(dir.join("config/device.json")).unwrap();
    c["cameras"] = json!([0]);
    atomic(dir.join("config/device.json"), &c).unwrap();
    let info = json!({"active":true,"pid":123});
    let good = json!({"schema_version":1,"pid":123,"release_id":s["release_id"],"launch_token":"token","uptime_ms":1000,"cameras":[{"index":0,"frames":20,"last_frame_age_ms":5,"pipeline_ok":true,"backends":["apriltag-cpu"]}]});
    activation::health(t.path(), &s, &info, &good).unwrap();
    for (key, bad) in [
        ("pid", json!(124)),
        ("schema_version", json!(2)),
        ("launch_token", json!("old")),
        ("release_id", json!("old")),
        ("uptime_ms", json!(-1)),
        ("cameras", json!([])),
    ] {
        let mut r = good.clone();
        r[key] = bad;
        assert!(
            activation::health(t.path(), &s, &info, &r).is_err(),
            "{key}"
        );
    }
    for (key, bad) in [
        ("frames", json!(0)),
        ("index", json!(1)),
        ("pipeline_ok", json!(false)),
        ("last_frame_age_ms", json!(1501)),
        ("backends", json!([])),
    ] {
        let mut r = good.clone();
        r["cameras"][0][key] = bad;
        assert!(
            activation::health(t.path(), &s, &info, &r).is_err(),
            "{key}"
        );
    }
}
#[test]
fn gpu_and_yolo_backend_requirements_follow_profile() {
    let mut c = device();
    let mut r = json!({"object_detection":{"use_nn":true}});
    for (p, expected) in [
        ("cpu", vec!["apriltag-cpu"]),
        ("gpu", vec!["apriltag-gpu"]),
        ("tensorrt", vec!["apriltag-cpu", "yolo-tensorrt"]),
        ("gpu-tensorrt", vec!["apriltag-gpu", "yolo-tensorrt"]),
    ] {
        c["build_profile"] = json!(p);
        assert_eq!(launcher::required(&c, &r).unwrap(), expected);
    }
    r["object_detection"]["use_nn"] = json!(false);
    assert_eq!(launcher::required(&c, &r).unwrap(), vec!["apriltag-gpu"]);
}
#[test]
fn compatibility_checks_ubuntu_and_jetpack() {
    let t = tempfile::tempdir().unwrap();
    fs::write(
        t.path().join("os-release"),
        "ID=ubuntu\nVERSION_ID=\"22.04\"\n",
    )
    .unwrap();
    initialize::compatibility("cpu", "x86_64", t.path(), &|_| true).unwrap();
    assert!(initialize::compatibility("gpu", "x86_64", t.path(), &|_| true).is_err());
    assert!(initialize::compatibility("gpu", "aarch64", t.path(), &|p| p
        == Path::new("/run/systemd/system"))
    .is_err());
    initialize::compatibility("gpu-tensorrt", "aarch64", t.path(), &|_| true).unwrap();
    fs::write(t.path().join("os-release"), "ID=debian\nVERSION_ID=12\n").unwrap();
    assert!(initialize::compatibility("cpu", "x86_64", t.path(), &|_| true).is_err());
}
#[test]
fn initialization_rejects_unsafe_accounts_and_paths() {
    let sha = "a".repeat(64);
    for user in ["root", "Bad", "user;cmd", ""] {
        let o = initialize::Options {
            root: Path::new("/home/v/Vortex.deployment"),
            legacy: Path::new("/home/v/Vortex"),
            user,
            service: "vortex.service",
            job: "job",
            release: "release",
            sha: &sha,
            offline: None,
        };
        assert!(o.validate().is_err());
    }
}
#[test]
fn diagnostics_detach_only_removes_own_lease() {
    let t = tempfile::tempdir().unwrap();
    fs::create_dir(t.path().join("diagnostics")).unwrap();
    let req = t.path().join("diagnostics/request.json");
    atomic(&req, &json!({"client_id":"one"})).unwrap();
    let m = Fake::new();
    service::diagnostics(t.path(), 0, "two", true, &m).unwrap();
    assert!(req.exists());
    service::diagnostics(t.path(), 0, "one", true, &m).unwrap();
    assert!(!req.exists());
    assert_eq!(m.starts.get() + m.stops.get(), 0);
}
#[test]
fn diagnostics_stopped_service_never_controls_service() {
    let t = tempfile::tempdir().unwrap();
    let m = Fake::new();
    let r = service::diagnostics(t.path(), 0, "client", false, &m).unwrap();
    assert_eq!(r["ready"], false);
    assert_eq!(m.starts.get() + m.stops.get(), 0);
    assert!(!t.path().join("diagnostics").exists());
}
#[test]
fn service_names_are_restricted() {
    for s in ["vortex.service", "vortex-vision.service"] {
        crate::common::service(s).unwrap();
    }
    for s in [
        "../a.service",
        "bad;cmd.service",
        "x.service\n",
        "-a.service",
        "a.timer",
    ] {
        assert!(crate::common::service(s).is_err());
    }
}
#[cfg(unix)]
mod unix {
    use super::*;
    fn second_release(root: &Path, old: &Value) -> Value {
        let source = root
            .join("releases")
            .join(string(old, "release_id").unwrap());
        let dest = root.join("releases/second");
        fs::create_dir(&dest).unwrap();
        for entry in walkdir::WalkDir::new(&source).min_depth(1) {
            let entry = entry.unwrap();
            let target = dest.join(entry.path().strip_prefix(&source).unwrap());
            if entry.file_type().is_dir() {
                fs::create_dir_all(&target).unwrap();
            } else {
                fs::copy(entry.path(), &target).unwrap();
            }
        }
        let mut manifest = read(dest.join("manifest.json")).unwrap();
        manifest["release_id"] = json!("second");
        atomic(dest.join("manifest.json"), &manifest).unwrap();
        atomic(
            dest.join("release.json"),
            &json!({"release_id":"second","profile":"cpu"}),
        )
        .unwrap();
        json!({"kind":"release","release_id":"second","launch_token":"second-token"})
    }
    fn committed(root: &Path, selection: &Value) {
        activation::switch(root, selection).unwrap();
        atomic(root.join("active.json"), selection).unwrap();
    }
    #[test]
    fn failed_candidate_restores_previous_release_with_fresh_identity() {
        let (t, _, old) = staged();
        let next = second_release(t.path(), &old);
        committed(t.path(), &old);
        let m = Fake::new();
        m.active.set(true);
        let old_c = fs::read(
            t.path()
                .join("releases")
                .join(string(&old, "release_id").unwrap())
                .join("config/device.json"),
        )
        .unwrap();
        let result = activation::activate_with(
            t.path(),
            next["release_id"].as_str(),
            false,
            "op",
            &m,
            &|_, _| Ok(json!({})),
            &|_, s, _| {
                anyhow::ensure!(s["release_id"] != "second", "Candidate camera failed");
                Ok(json!({"verification":"all_cameras_advancing"}))
            },
            &t.path().join("lock"),
        )
        .unwrap();
        assert_eq!(result["state"], "rolled_back");
        assert_eq!(result["selection"]["release_id"], old["release_id"]);
        assert_ne!(result["selection"]["launch_token"], old["launch_token"]);
        assert_eq!(m.starts.get(), 2);
        assert!(!t.path().join("activation.json").exists());
        assert_eq!(
            fs::read(
                t.path()
                    .join("releases")
                    .join(string(&old, "release_id").unwrap())
                    .join("config/device.json")
            )
            .unwrap(),
            old_c
        );
    }
    #[test]
    fn rollback_failure_preserves_recovery_journal() {
        let (t, _, old) = staged();
        let next = second_release(t.path(), &old);
        committed(t.path(), &old);
        let m = Fake::new();
        m.active.set(true);
        let result = activation::activate_with(
            t.path(),
            next["release_id"].as_str(),
            false,
            "op",
            &m,
            &|_, _| Ok(json!({})),
            &|_, _, _| anyhow::bail!("Cameras unavailable"),
            &t.path().join("lock"),
        )
        .unwrap();
        assert_eq!(result["state"], "rollback_failed");
        assert!(result["verification"].is_null());
        assert!(t.path().join("activation.json").exists());
        assert!(result["rollback_error"]
            .as_str()
            .unwrap()
            .contains("Cameras unavailable"));
    }
    #[test]
    fn failed_stop_keeps_old_process_and_launch_token() {
        let (t, _, old) = staged();
        let next = second_release(t.path(), &old);
        committed(t.path(), &old);
        let m = Fake::new();
        m.active.set(true);
        m.fail_stop.set(true);
        let result = activation::activate_with(
            t.path(),
            next["release_id"].as_str(),
            false,
            "op",
            &m,
            &|_, _| Ok(json!({})),
            &|_, s, _| {
                assert_eq!(*s, old);
                Ok(json!({"verification":"all_cameras_advancing"}))
            },
            &t.path().join("lock"),
        )
        .unwrap();
        assert_eq!(result["state"], "rolled_back");
        assert_eq!(m.starts.get(), 0);
        assert_eq!(activation::selected(t.path()).unwrap(), old);
    }
    #[test]
    fn manual_rollback_rotates_selection_and_previous() {
        let (t, _, old) = staged();
        let next = second_release(t.path(), &old);
        committed(t.path(), &next);
        atomic(t.path().join("previous.json"), &old).unwrap();
        let m = Fake::new();
        m.active.set(true);
        let result = activation::activate_with(
            t.path(),
            None,
            true,
            "op",
            &m,
            &|_, _| Ok(json!({})),
            &|_, _, _| Ok(json!({"verification":"all_cameras_advancing"})),
            &t.path().join("lock"),
        )
        .unwrap();
        assert_eq!(result["state"], "active");
        assert_eq!(result["action"], "rollback");
        assert_eq!(result["selection"]["release_id"], old["release_id"]);
        assert_eq!(read(t.path().join("previous.json")).unwrap(), next);
    }
    #[test]
    fn invalid_recovery_journal_never_stops_service() {
        let t = tempfile::tempdir().unwrap();
        let m = Fake::new();
        for phase in ["unknown", "prepared"] {
            let j = json!({"schema_version":1,"phase":phase,"old":{"kind":"invalid"},"new":{"kind":"none"},"previous_before":{"kind":"none"}});
            atomic(t.path().join("activation.json"), &j).unwrap();
            assert!(activation::recover(t.path(), Some(&m)).is_err());
            assert_eq!(m.stops.get(), 0);
            assert_eq!(read(t.path().join("activation.json")).unwrap(), j);
        }
    }
    #[test]
    fn symlink_payloads_and_managed_paths_are_rejected() {
        let t = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink("/tmp", t.path().join("escape")).unwrap();
        assert!(child(t.path(), "escape/file").is_err());
        assert!(regular(&t.path().join("escape")).is_err());
    }
    #[test]
    fn activation_commits_only_after_health() {
        let (t, _, sel) = staged();
        let m = Fake::new();
        let result = activation::activate_with(
            t.path(),
            sel["release_id"].as_str(),
            false,
            "operation",
            &m,
            &|_, _| Ok(json!({})),
            &|root, new, m| {
                assert!(m.info()?["active"] == true);
                assert!(!root.join("active.json").exists());
                assert_eq!(read(root.join("selection.json"))?, *new);
                Ok(json!({"verification":"all_cameras_advancing"}))
            },
            &t.path().join("lock"),
        )
        .unwrap();
        assert_eq!(result["state"], "active");
        assert_eq!(
            read(t.path().join("active.json")).unwrap(),
            result["selection"]
        );
        assert!(!t.path().join("activation.json").exists());
        assert_eq!(m.starts.get(), 1);
    }
    #[test]
    fn first_install_health_failure_stops_candidate_and_restores_none() {
        let (t, _, sel) = staged();
        let m = Fake::new();
        let result = activation::activate_with(
            t.path(),
            sel["release_id"].as_str(),
            false,
            "operation",
            &m,
            &|_, _| Ok(json!({})),
            &|_, new, m| {
                if new["kind"] == "release" {
                    anyhow::bail!("camera failure")
                }
                assert_eq!(m.info()?["active"], false);
                Ok(json!({"verification":"stopped_no_previous_release"}))
            },
            &t.path().join("lock"),
        )
        .unwrap();
        assert_eq!(result["state"], "rolled_back");
        assert_eq!(activation::selected(t.path()).unwrap()["kind"], "none");
        assert!(!t.path().join("current").exists());
    }
    #[test]
    fn preflight_failure_never_stops_existing_service() {
        let (t, _, sel) = staged();
        let m = Fake::new();
        assert!(activation::activate_with(
            t.path(),
            sel["release_id"].as_str(),
            false,
            "operation",
            &m,
            &|_, _| anyhow::bail!("bad executable"),
            &|_, _, _| unreachable!(),
            &t.path().join("lock")
        )
        .is_err());
        assert_eq!(m.stops.get(), 0);
        assert!(!t.path().join("activation.json").exists());
    }
    #[test]
    fn unconfirmed_journal_recovers_previous_without_claiming_health() {
        let (t, _, s) = staged();
        let journal = json!({"schema_version":1,"operation_id":"op","phase":"verifying","old":{"kind":"none"},"new":s,"previous_before":{"kind":"none"},"boot_id":"old","pid":0,"process_start":"0"});
        atomic(t.path().join("activation.json"), &journal).unwrap();
        activation::switch(t.path(), &s).unwrap();
        let r = activation::recover(t.path(), None).unwrap();
        assert_eq!(r["state"], "restored_previous_selection");
        assert_eq!(r["runtime_health_verified"], false);
        assert_eq!(activation::selected(t.path()).unwrap()["kind"], "none");
    }
    #[test]
    fn confirmed_journal_keeps_healthy_selection() {
        let (t, _, s) = staged();
        let journal = json!({"schema_version":1,"operation_id":"op","phase":"confirmed","old":{"kind":"none"},"new":s,"previous_before":{"kind":"none"},"boot_id":"old","pid":0,"process_start":"0"});
        atomic(t.path().join("activation.json"), &journal).unwrap();
        let r = activation::recover(t.path(), None).unwrap();
        assert_eq!(r["state"], "completed_confirmed_activation");
        assert_eq!(activation::selected(t.path()).unwrap(), s);
    }
    #[test]
    fn live_owner_recovery_is_refused() {
        let t = tempfile::tempdir().unwrap();
        let journal = json!({"schema_version":1,"operation_id":"op","phase":"prepared","old":{"kind":"none"},"new":{"kind":"none"},"previous_before":{"kind":"none"},"boot_id":fs::read_to_string("/proc/sys/kernel/random/boot_id").unwrap().trim(),"pid":std::process::id(),"process_start":activation::process_identity(std::process::id() as u64)});
        atomic(t.path().join("activation.json"), &journal).unwrap();
        assert!(activation::recover(t.path(), None).is_err());
        assert!(t.path().join("activation.json").exists());
    }
}
