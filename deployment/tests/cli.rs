//! Exercise actual CLI processes and persistent files. Linux command shims are
//! confined to a temporary PATH; tests never contact a service manager or camera.
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};
fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_vortexctl")
}
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .into()
}
fn parse(o: Output) -> Value {
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&o.stdout).unwrap()
}
#[test]
fn help_and_version_run_without_python_or_project() {
    for flag in ["--help", "--version"] {
        let o = Command::new(binary())
            .arg(flag)
            .env("PATH", "")
            .output()
            .unwrap();
        assert!(o.status.success());
    }
}
#[test]
fn missing_required_options_fail_actionably() {
    for action in [
        "pack",
        "stage",
        "start-job",
        "activate",
        "initialize-submit",
    ] {
        let o = Command::new(binary()).arg(action).output().unwrap();
        assert!(!o.status.success());
        assert!(String::from_utf8_lossy(&o.stderr).contains("required"));
    }
}
#[test]
fn relative_managed_roots_are_rejected() {
    let o = Command::new(binary())
        .args(["status", "--root", "relative"])
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("absolute"));
}
#[test]
fn incompatible_launch_flags_are_rejected() {
    for flags in [
        vec!["--build", "--check"],
        vec!["--online"],
        vec!["--with-debug-bridge"],
    ] {
        let o = Command::new(binary())
            .args(["launch", "--source"])
            .arg(repo())
            .args(flags)
            .output()
            .unwrap();
        assert!(!o.status.success());
    }
}
#[test]
fn config_validation_runs_without_compiler() {
    let o = Command::new(binary())
        .args(["launch", "--source"])
        .arg(repo())
        .arg("--validate-config")
        .env("PATH", "")
        .output()
        .unwrap();
    assert_eq!(parse(o)["valid"], true);
}
#[test]
fn build_check_never_implicitly_compiles() {
    let t = tempfile::tempdir().unwrap();
    let mut c: Value =
        serde_json::from_slice(&fs::read(repo().join("config/device.json")).unwrap()).unwrap();
    c["runtime_config"] = json!(repo().join("config/config.json"));
    c["tag_map"] = json!(repo().join(c["tag_map"].as_str().unwrap()));
    fs::write(t.path().join("device.json"), c.to_string()).unwrap();
    let o = Command::new(binary())
        .args(["launch", "--source"])
        .arg(t.path())
        .args(["--device-config", "device.json", "--check"])
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(!t.path().join("target").exists());
}

#[cfg(target_os = "linux")]
mod linux {
    fn packaged(h: &Harness) -> Value {
        let output = h.temp.path().join("package");
        let mut c: Value =
            serde_json::from_slice(&fs::read(repo().join("config/device.json")).unwrap()).unwrap();
        c["build_profile"] = json!("cpu");
        let dev = h.temp.path().join("device.json");
        fs::write(&dev, c.to_string()).unwrap();
        parse(
            Command::new(binary())
                .args(["pack", "--source"])
                .arg(repo())
                .arg("--output")
                .arg(&output)
                .arg("--device-config")
                .arg(&dev)
                .arg("--controller")
                .arg(binary())
                .output()
                .unwrap(),
        )
    }
    fn fake_compiler(h: &Harness, extra: &str) {
        let path = h.home.join(".cargo/bin");
        fs::create_dir_all(&path).unwrap();
        let script = format!(
            r#"#!/bin/sh
set -eu
printf '%s\n' "$@" > "$HOME/build-args"
test -z "${{CARGO_TARGET_DIR:-}}"
test -z "${{CARGO_BUILD_TARGET:-}}"
test -z "${{RUSTUP_TOOLCHAIN:-}}"
test "$RUSTUP_AUTO_INSTALL" = 0
cache=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = --target-dir ]; then shift; cache="$1"; fi
  shift
done
/bin/mkdir -p "$cache/release"
printf '#!/bin/sh\nexit 0\n' > "$cache/release/dumapril-taglocalization"
/bin/chmod 755 "$cache/release/dumapril-taglocalization"
{extra}
"#
        );
        let p = path.join("cargo");
        fs::write(&p, script).unwrap();
        fs::set_permissions(p, fs::Permissions::from_mode(0o755)).unwrap();
    }
    fn stage(h: &Harness, p: &Value) -> Output {
        h.command("stage")
            .arg("--archive")
            .arg(h.temp.path().join("package/release.tar.gz"))
            .args([
                "--sha256",
                p["archive_sha256"].as_str().unwrap(),
                "--release-id",
                p["release_id"].as_str().unwrap(),
            ])
            .env("CARGO_BUILD_TARGET", "inherited-invalid-target")
            .env("CARGO_TARGET_DIR", "inherited-invalid-cache")
            .env("RUSTUP_TOOLCHAIN", "inherited-invalid-toolchain")
            .output()
            .unwrap()
    }
    #[test]
    fn complete_stage_builds_offline_and_owns_immutable_binary() {
        let h = Harness::new();
        let p = packaged(&h);
        fake_compiler(&h, "");
        let ready = parse(stage(&h, &p));
        assert_eq!(ready["state"], "ready");
        assert_eq!(ready["activated"], false);
        let args = fs::read_to_string(h.home.join("build-args")).unwrap();
        assert!(args.starts_with("+1.96.0\nbuild\n"));
        for flag in ["--offline", "--locked", "--no-default-features"] {
            assert!(args.contains(flag));
        }
        assert!(!h.root.join("selection.json").exists());
        let own = Path::new(ready["release_path"].as_str().unwrap())
            .join("target/release/dumapril-taglocalization");
        let original = fs::read(&own).unwrap();
        fs::write(
            h.root
                .join("build-cache/cpu/release/dumapril-taglocalization"),
            "changed-cache",
        )
        .unwrap();
        assert_eq!(fs::read(&own).unwrap(), original);
        assert!(!stage(&h, &p).status.success());
    }
    #[test]
    fn failed_build_never_leaves_ready_release_or_valid_stamp() {
        let h = Harness::new();
        let p = packaged(&h);
        fake_compiler(&h, "exit 42");
        assert!(!stage(&h, &p).status.success());
        let identity = p["release_id"].as_str().unwrap();
        let job: Value = serde_json::from_slice(
            &fs::read(h.root.join(format!("jobs/{identity}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(job["state"], "failed");
        assert!(!h.root.join(format!("releases/{identity}")).exists());
        assert!(!h
            .root
            .join(format!(
                "staging/{identity}/target/release/vortex-build.json"
            ))
            .exists());
    }
    #[test]
    fn source_mutation_during_build_cannot_be_published() {
        let h = Harness::new();
        let p = packaged(&h);
        fake_compiler(&h, "printf 'changed source' > src/main.rs");
        let o = stage(&h, &p);
        assert!(!o.status.success());
        assert!(String::from_utf8_lossy(&o.stderr).contains("Source changed"));
        assert!(!h
            .root
            .join("releases")
            .join(p["release_id"].as_str().unwrap())
            .exists());
    }
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    struct Harness {
        temp: tempfile::TempDir,
        root: PathBuf,
        bin: PathBuf,
        home: PathBuf,
    }
    impl Harness {
        fn new() -> Self {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().join("managed");
            let bin = temp.path().join("bin");
            let home = temp.path().join("home");
            for p in [&root, &bin, &home] {
                fs::create_dir(p).unwrap();
            }
            Self {
                temp,
                root,
                bin,
                home,
            }
        }
        fn script(&self, name: &str, body: &str) {
            let p = self.bin.join(name);
            fs::write(&p, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
            fs::set_permissions(p, fs::Permissions::from_mode(0o755)).unwrap();
        }
        fn command(&self, action: &str) -> Command {
            let mut c = Command::new(binary());
            c.arg(action)
                .arg("--root")
                .arg(&self.root)
                .env("HOME", &self.home)
                .env("PATH", &self.bin);
            c
        }
        fn supervisor(&self, state: &str) {
            self.script("loginctl", "echo yes");
            self.script("systemctl",&format!("printf 'ActiveState={state}\\nLoadState=loaded\\nResult=success\\nExecMainStatus=0\\n'"));
            self.script(
                "systemd-run",
                "printf '%s\\n' \"$@\" > \"$HOME/submitted-args\"",
            );
        }
        fn descriptor(&self, identity: &str, state: &str, dispatch: &str, submitted: f64) {
            let jobs = self.root.join("jobs");
            fs::create_dir_all(&jobs).unwrap();
            use sha2::{Digest, Sha256};
            let unit = format!(
                "vortex-job-{}.service",
                &format!(
                    "{:x}",
                    Sha256::digest(format!("{}/{identity}", self.root.display()).as_bytes())
                )[..32]
            );
            fs::write(jobs.join(format!("{identity}.unit.json")),json!({"unit":unit,"job_id":identity,"manager":"user","kind":"stage","dispatch":dispatch,"submitted_at":submitted}).to_string()).unwrap();
            fs::write(
                jobs.join(format!("{identity}.json")),
                json!({"release_id":identity,"state":state,"activated":false}).to_string(),
            )
            .unwrap();
        }
        fn status(&self, identity: &str) -> Output {
            self.command("status")
                .args(["--release-id", identity])
                .output()
                .unwrap()
        }
    }
    #[test]
    fn persistent_submission_uses_exactly_one_worker_and_no_restart() {
        let h = Harness::new();
        h.supervisor("active");
        let o = h
            .command("start-job")
            .args([
                "--kind",
                "activate",
                "--job-id",
                "operation",
                "--release-id",
                "release",
            ])
            .output()
            .unwrap();
        let v = parse(o);
        assert_eq!(v["dispatch"], "submitted");
        let args = fs::read_to_string(h.home.join("submitted-args")).unwrap();
        assert!(
            args.contains("--user\n")
                && args.contains("--property=Restart=no")
                && args.contains("--property=KillMode=control-group")
        );
        assert!(args.contains("/vortexctl\nactivate\n"));
        assert!(!h
            .command("start-job")
            .args([
                "--kind",
                "activate",
                "--job-id",
                "operation",
                "--release-id",
                "release"
            ])
            .output()
            .unwrap()
            .status
            .success());
    }
    #[test]
    fn missing_lingering_prevents_dispatch() {
        let h = Harness::new();
        h.script("loginctl", "echo no");
        let o = h
            .command("start-job")
            .args(["--kind", "rollback", "--job-id", "operation"])
            .output()
            .unwrap();
        assert!(!o.status.success());
        assert!(!h.root.join("jobs/operation.dispatch").exists());
    }
    #[test]
    fn lost_submission_reply_preserves_descriptor() {
        let h = Harness::new();
        h.supervisor("active");
        h.script("systemd-run", "exit 1");
        let o = h
            .command("start-job")
            .args(["--kind", "rollback", "--job-id", "operation"])
            .output()
            .unwrap();
        assert!(!o.status.success());
        let d: Value =
            serde_json::from_slice(&fs::read(h.root.join("jobs/operation.unit.json")).unwrap())
                .unwrap();
        assert_eq!(d["dispatch"], "submitting");
        assert!(h.root.join("jobs/operation.dispatch").exists());
    }
    #[test]
    fn interrupted_worker_never_stays_building() {
        let h = Harness::new();
        h.supervisor("inactive");
        h.descriptor("release", "building", "submitted", 0.);
        let s = parse(h.status("release"));
        assert_eq!(s["state"], "interrupted");
        assert!(s["error"].as_str().unwrap().contains("worker stopped"));
    }
    #[test]
    fn active_worker_is_still_followed() {
        let h = Harness::new();
        h.supervisor("active");
        h.descriptor("release", "building", "submitted", 0.);
        assert_eq!(parse(h.status("release"))["state"], "building");
    }
    #[test]
    fn completed_result_survives_collected_transient_unit() {
        let h = Harness::new();
        h.supervisor("inactive");
        for terminal in [
            "ready",
            "active",
            "rolled_back",
            "rollback_failed",
            "failed",
            "recovered",
            "interrupted",
        ] {
            h.descriptor("release", terminal, "submitted", 0.);
            assert_eq!(parse(h.status("release"))["state"], terminal);
        }
    }
    #[test]
    fn unreachable_manager_is_not_reported_as_interrupted() {
        let h = Harness::new();
        h.script("systemctl", "echo unavailable >&2; exit 1");
        h.descriptor("release", "building", "submitted", 0.);
        let o = h.status("release");
        assert!(!o.status.success());
        assert!(String::from_utf8_lossy(&o.stderr).contains("Cannot inspect"));
    }
    #[test]
    fn forged_manager_descriptor_is_rejected() {
        let h = Harness::new();
        h.supervisor("active");
        h.descriptor("release", "building", "submitted", 0.);
        let p = h.root.join("jobs/release.unit.json");
        let mut v: Value = serde_json::from_slice(&fs::read(&p).unwrap()).unwrap();
        v["unit"] = json!("other.service");
        fs::write(&p, v.to_string()).unwrap();
        assert!(!h.status("release").status.success());
    }
    #[test]
    fn bounded_log_tail_restores_build_output() {
        let h = Harness::new();
        fs::create_dir(h.root.join("jobs")).unwrap();
        let mut data = "x".repeat(20000);
        data.push_str("compiler failure\n");
        fs::write(h.root.join("jobs/release.log"), data).unwrap();
        let v = parse(h.status("release"));
        assert_eq!(v["log_tail"].as_str().unwrap().len(), 16384);
        assert!(v["log_tail"]
            .as_str()
            .unwrap()
            .ends_with("compiler failure\n"));
    }
    #[test]
    fn stage_controller_must_be_uploaded_verified_location() {
        let h = Harness::new();
        h.supervisor("active");
        let o = h
            .command("start-job")
            .args([
                "--kind",
                "stage",
                "--job-id",
                "release",
                "--release-id",
                "release",
                "--sha256",
                &"a".repeat(64),
            ])
            .output()
            .unwrap();
        assert!(!o.status.success());
        assert!(!h.root.join("jobs/release.dispatch").exists());
    }
    #[test]
    fn service_entry_refuses_unselected_release() {
        let h = Harness::new();
        let o = h.command("service-run").output().unwrap();
        assert!(!o.status.success());
        assert!(String::from_utf8_lossy(&o.stderr).contains("No selected release"));
    }
    #[test]
    fn stage_verification_failure_is_durable_and_does_not_touch_live_selection() {
        let h = Harness::new();
        let archive = h.temp.path().join("archive");
        fs::write(&archive, "corrupted").unwrap();
        let sentinel = json!({"kind":"none","marker":"unchanged"});
        fs::write(h.root.join("selection.json"), sentinel.to_string()).unwrap();
        let o = h
            .command("stage")
            .arg("--archive")
            .arg(&archive)
            .args(["--sha256", &"a".repeat(64), "--release-id", "release"])
            .output()
            .unwrap();
        assert!(!o.status.success());
        let job: Value =
            serde_json::from_slice(&fs::read(h.root.join("jobs/release.json")).unwrap()).unwrap();
        assert_eq!(job["state"], "failed");
        assert_eq!(job["activated"], false);
        let sel: Value =
            serde_json::from_slice(&fs::read(h.root.join("selection.json")).unwrap()).unwrap();
        assert_eq!(sel, sentinel);
        assert!(!h.root.join("current").exists());
    }
}
