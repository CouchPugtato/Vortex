//! Opt-in privileged integration test. Uses a fresh account/unit and never vision hardware.
#![cfg(target_os = "linux")]
use anyhow::{ensure, Result};
use common::*;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};
use vortexctl::{common, launcher, offline, package};
fn controller() -> PathBuf {
    std::env::var_os("VORTEX_TEST_CONTROLLER")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_vortexctl")))
}
fn repository() -> PathBuf {
    std::env::var_os("VORTEX_TEST_REPOSITORY")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .into()
        })
}
struct Sandbox {
    user: String,
    home: PathBuf,
    root: PathBuf,
    unit: String,
}
impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = Command::new("systemctl")
            .args(["disable", "--now", &self.unit])
            .output();
        let _ = fs::remove_file(Path::new("/etc/systemd/system").join(&self.unit));
        let _ = fs::remove_file(
            Path::new("/etc/sudoers.d")
                .join(format!("vortex-{}", self.unit.trim_end_matches(".service"))),
        );
        let _ = Command::new("systemctl").arg("daemon-reload").output();
        let _ = Command::new("loginctl")
            .args(["disable-linger", &self.user])
            .output();
        let _ = Command::new("loginctl")
            .args(["terminate-user", &self.user])
            .output();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if Command::new("userdel")
                .arg(&self.user)
                .status()
                .is_ok_and(|s| s.success())
            {
                break;
            }
            if Instant::now() >= deadline {
                eprintln!("Test account cleanup needs inspection: {}", self.user);
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        if self.home.parent() == Some(Path::new("/home")) && self.user.starts_with("vortex-test-") {
            let _ = fs::remove_dir_all(&self.home);
        }
    }
}
impl Sandbox {
    fn command(&self, action: &str) -> Command {
        let mut c = Command::new("runuser");
        c.args(["-u", &self.user, "--", "env"])
            .arg(format!("HOME={}", self.home.display()))
            .arg(self.root.join("vortexctl"))
            .args([action, "--root"])
            .arg(&self.root);
        c
    }
    fn call(&self, action: &str, args: &[&str]) -> Result<Value> {
        let o = checked(self.command(action).args(args), 120)?;
        Ok(serde_json::from_slice(&o.stdout)?)
    }
    fn wait_job(&self, op: &str) -> Result<Value> {
        let end = Instant::now() + Duration::from_secs(100);
        loop {
            let v = self.call("status", &["--operation-id", op])?;
            if [
                "active",
                "failed",
                "rolled_back",
                "rollback_failed",
                "interrupted",
            ]
            .contains(&v["state"].as_str().unwrap_or(""))
            {
                return Ok(v);
            }
            ensure!(
                Instant::now() < end,
                "Service test operation timed out: {v}"
            );
            std::thread::sleep(Duration::from_millis(250));
        }
    }
    fn submit(&self, op: &str, release: Option<&str>) -> Result<Value> {
        let kind = if release.is_some() {
            "activate"
        } else {
            "rollback"
        };
        let mut args = vec!["--kind", kind, "--job-id", op];
        if let Some(r) = release {
            args.extend(["--release-id", r]);
        }
        self.call("start-job", &args)?;
        self.wait_job(op)
    }
    fn prepare(&self, mode: &str) -> Result<String> {
        let repository = repository();
        let repo = repository.as_path();
        let out = tempfile::tempdir_in(&self.home)?;
        let mut device = read(repo.join("config/device.json"))?;
        device["build_profile"] = json!("cpu");
        device["cameras"] = json!([0]);
        atomic(out.path().join("device.json"), &device)?;
        let mut runtime = read(repo.join("config/config.json"))?;
        runtime["deployment_fixture"] = json!(mode);
        atomic(out.path().join("runtime.json"), &runtime)?;
        let p = package::pack(
            repo,
            out.path(),
            &out.path().join("device.json"),
            Some(&out.path().join("runtime.json")),
            None,
            &controller(),
        )?;
        let identity = string(&p, "release_id")?.to_string();
        fs::create_dir_all(self.root.join("releases"))?;
        let dest = self.root.join("releases").join(&identity);
        package::extract(&out.path().join("release.tar.gz"), &dest, &identity)?;
        let fixture = std::env::var_os("VORTEX_SERVICE_FIXTURE")
            .map(PathBuf::from)
            .expect("Set VORTEX_SERVICE_FIXTURE to the built service_fixture example");
        copy(&fixture, &launcher::binary(&dest, launcher::MAIN), 0o755)?;
        atomic(
            dest.join("target/release/vortex-build.json"),
            &json!({"schema_version":1,"runtime":"main","profile":"cpu","source_sha256":launcher::source_digest(&dest)?,"binaries":{launcher::MAIN:hash(launcher::binary(&dest,launcher::MAIN))?}}),
        )?;
        atomic(
            dest.join("release.json"),
            &json!({"release_id":identity,"profile":"cpu"}),
        )?;
        checked(
            Command::new("chown")
                .arg("-R")
                .arg(format!("{}:{}", self.user, self.user))
                .arg(&self.root),
            20,
        )?;
        Ok(identity)
    }
}
#[test]
#[ignore = "Requires root and a real systemd host; creates only a dedicated vortex-test-* account/unit"]
fn actual_service_lifecycle() -> Result<()> {
    admin()?;
    checked(Command::new("systemctl").arg("show-environment"), 10)?;
    let user = format!("vortex-test-{}", &token()[..10]);
    let home = PathBuf::from(format!("/home/{user}"));
    ensure!(!home.exists(), "Test home already exists");
    checked(
        Command::new("useradd").args(["-m", "-s", "/bin/bash", &user]),
        20,
    )?;
    let s = Sandbox {
        root: home.join("vision.deployment"),
        unit: format!("{user}.service"),
        user,
        home,
    };
    fs::create_dir_all(s.home.join(".local/state/vortex"))?;
    fs::write(s.home.join(".local/state/vortex/deployment.lock"), "")?;
    fs::create_dir_all(&s.root)?;
    checked(
        Command::new("chown")
            .arg("-R")
            .arg(format!("{}:{}", s.user, s.user))
            .arg(&s.home),
        20,
    )?;
    let first = s.prepare("healthy")?;
    let second = s.prepare("healthy")?;
    let bad = s.prepare("fail")?;
    checked(
        Command::new(controller())
            .args(["install-service", "--root"])
            .arg(&s.root)
            .arg("--legacy-root")
            .arg(s.home.join("vision"))
            .args(["--user", &s.user, "--service", &s.unit]),
        30,
    )?;
    let account = checked(Command::new("id").args(["-u", &s.user]), 10)?;
    let uid = String::from_utf8(account.stdout)?.trim().to_string();
    checked(
        Command::new("systemctl").args(["start", &format!("user@{uid}.service")]),
        30,
    )?;
    checked(
        Command::new("runuser").args([
            "-u",
            &s.user,
            "--",
            "sudo",
            "-n",
            "systemctl",
            "stop",
            &s.unit,
        ]),
        20,
    )?;
    ensure!(
        !output(
            Command::new("runuser").args([
                "-u",
                &s.user,
                "--",
                "sudo",
                "-n",
                "systemctl",
                "stop",
                "vortex-unrelated.service"
            ]),
            10
        )?
        .status
        .success(),
        "Service permission was broader than expected"
    );
    ensure!(
        s.submit("first", Some(&first))?["state"] == "active",
        "First activation failed"
    );
    ensure!(
        s.submit("second", Some(&second))?["state"] == "active",
        "Second activation failed"
    );
    ensure!(
        s.submit("rollback", None)?["selection"]["release_id"] == first,
        "Manual rollback failed"
    );
    let failure = s.submit("bad", Some(&bad))?;
    ensure!(
        failure["state"] == "rolled_back" && failure["selection"]["release_id"] == first,
        "Candidate failure did not restore prior release: {failure}"
    );
    let health = read(s.root.join("health.json"))?;
    let stall = s.root.join("releases").join(&first).join(".fixture-stall");
    fs::write(&stall, "")?;
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let h = read(s.root.join("health.json"))?;
        if h["pid"] != health["pid"] {
            break;
        }
        ensure!(
            Instant::now() < deadline,
            "Watchdog failed to restart the stalled fixture"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    fs::remove_file(stall)?;
    checked(Command::new("systemctl").args(["stop", &s.unit]), 20)?;
    let old = read(s.root.join("selection.json"))?;
    let next = json!({"kind":"release","release_id":second,"launch_token":"interrupted"});
    fs::remove_file(s.root.join("current"))?;
    std::os::unix::fs::symlink(
        s.root.join("releases").join(&second),
        s.root.join("current"),
    )?;
    atomic(s.root.join("selection.json"), &next)?;
    atomic(
        s.root.join("activation.json"),
        &json!({"schema_version":1,"operation_id":"interrupted","phase":"verifying","old":old,"new":next,"previous_before":{"kind":"none"},"boot_id":"previous-boot","pid":0,"process_start":"0"}),
    )?;
    checked(Command::new("systemctl").args(["start", &s.unit]), 20)?;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if !s.root.join("activation.json").exists() {
            break;
        }
        ensure!(
            Instant::now() < deadline,
            "Service entry did not recover the interrupted switch"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    ensure!(
        read(s.root.join("selection.json"))?["release_id"] == first,
        "Startup selected unconfirmed candidate"
    );
    if let Some(bundle) = std::env::var_os("VORTEX_OFFLINE_BUNDLE") {
        let archive = PathBuf::from(bundle);
        let metadata = read(archive.parent().unwrap().join("offline.json"))?;
        let source = s.root.join("releases").join(&first);
        let dest = s.home.join("offline-inputs");
        offline::extract(
            &archive,
            string(&metadata, "archive_sha256")?,
            &dest,
            &source,
            "cpu",
        )?;
        let (mut update, mut install) = offline::package_commands(&dest, true)?;
        checked(&mut update, 120)?;
        checked(&mut install, 120)?;
        offline::install_user(&dest, &s.user, &s.home, "1.96.0")?;
        // Prove an empty user's restored compiler/cache works with networking disabled.
        checked(
            Command::new("unshare")
                .args(["--net", "--", "runuser", "-u", &s.user, "--", "env"])
                .arg(format!("HOME={}", s.home.display()))
                .arg(format!("CARGO_HOME={}/.cargo", s.home.display()))
                .arg(format!("RUSTUP_HOME={}/.rustup", s.home.display()))
                .arg("RUSTUP_AUTO_INSTALL=0")
                .arg(s.home.join(".cargo/bin/cargo"))
                .args([
                    "+1.96.0",
                    "fetch",
                    "--locked",
                    "--offline",
                    "--manifest-path",
                ])
                .arg(source.join("Cargo.toml")),
            120,
        )?;
        println!("OFFLINE PASS: full bundle hashes, local package resolution, empty-user toolchain/cache restore, Cargo fetch with networking disabled");
    }
    println!("REAL SYSTEMD PASS: install, restricted sudo, detached jobs, activation, rollback, watchdog restart, interrupted-start recovery");
    Ok(())
}

/// Exercise the production initialization worker with a real CPU build.
#[test]
#[ignore = "Installs packages and builds vision; requires a disposable offline rehearsal guest"]
fn actual_offline_initialization() -> Result<()> {
    admin()?;
    ensure!(
        Path::new("/etc/vortex-rehearsal-guest").is_file(),
        "Initialization rehearsal requires a disposable guest"
    );
    let user = format!("vortex-test-{}", &token()[..10]);
    checked(
        Command::new("useradd").args(["-m", "-s", "/bin/bash", &user]),
        20,
    )?;
    let home = PathBuf::from(format!("/home/{user}"));
    let s = Sandbox {
        root: home.join("vision.deployment"),
        unit: format!("{user}.service"),
        user,
        home,
    };
    let out = tempfile::tempdir_in(&s.home)?;
    let repo = repository();
    let mut device = read(repo.join("config/device.json"))?;
    device["build_profile"] = json!("cpu");
    device["cameras"] = json!([0]);
    atomic(out.path().join("device.json"), &device)?;
    let p = package::pack(
        &repo,
        out.path(),
        &out.path().join("device.json"),
        None,
        None,
        &controller(),
    )?;
    let release = string(&p, "release_id")?;
    let incoming = s.root.join("incoming").join(release);
    fs::create_dir_all(&incoming)?;
    copy(
        &out.path().join("release.tar.gz"),
        &incoming.join("release.tar.gz"),
        0o644,
    )?;
    checked(
        Command::new("chown")
            .arg("-R")
            .arg(format!("{}:{}", s.user, s.user))
            .arg(&s.home),
        20,
    )?;
    let bundle = PathBuf::from(std::env::var("VORTEX_OFFLINE_BUNDLE")?);
    let metadata = read(bundle.parent().unwrap().join("offline.json"))?;
    checked(
        Command::new(controller())
            .env("SUDO_USER", &s.user)
            .args(["initialize-submit", "--root"])
            .arg(&s.root)
            .arg("--legacy-root")
            .arg(s.home.join("vision"))
            .args([
                "--user",
                &s.user,
                "--service",
                &s.unit,
                "--job-id",
                "clean-init",
                "--release-id",
                release,
                "--sha256",
                string(&p, "archive_sha256")?,
                "--offline-bundle",
            ])
            .arg(&bundle)
            .args(["--offline-sha256", string(&metadata, "archive_sha256")?]),
        30,
    )?;
    let deadline = Instant::now() + Duration::from_secs(2400);
    let mut previous = String::new();
    loop {
        ensure!(
            Instant::now() < deadline,
            "Initialization timed out in {previous}"
        );
        if !s.root.join("jobs/clean-init.json").exists() {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let job = read(s.root.join("jobs/clean-init.json"))?;
        let state = string(&job, "state")?;
        if state != previous {
            println!("INITIALIZATION: {state}");
            previous = state.to_string();
        }
        if ["ready", "failed"].contains(&state) {
            let evidence = Path::new("/home/rehearsal/results");
            fs::create_dir_all(evidence)?;
            for entry in fs::read_dir(s.root.join("jobs"))? {
                let file = entry?.path();
                if file.is_file() {
                    let dest = evidence.join(file.file_name().unwrap());
                    fs::copy(&file, &dest)?;
                    fs::File::open(dest)?.sync_all()?;
                }
            }
            if state == "failed" {
                eprintln!(
                    "Initialization log: {}",
                    fs::read_to_string(s.root.join("jobs/clean-init.log"))?
                );
            }
            ensure!(
                state == "ready",
                "Clean offline initialization failed: {job}; see guest results"
            );
            ensure!(
                job["verification"] == "dependencies_build_smoke_service",
                "Initialization skipped validation"
            );
            ensure!(
                !s.root.join("selection.json").exists()
                    || read(s.root.join("selection.json"))?["kind"] == "none",
                "Initialization unexpectedly activated a camera runtime"
            );
            ensure!(
                launcher::binary(&s.root.join("releases").join(release), launcher::MAIN).is_file(),
                "Vision executable was not built"
            );
            println!("CLEAN OFFLINE INITIALIZATION PASS: empty account, actual package install, restored compiler/cache, real CPU vision build, saved-image smoke and service installation");
            break;
        }
        ensure!(
            Instant::now() < deadline,
            "Initialization timed out in {state}"
        );
        std::thread::sleep(Duration::from_secs(2));
    }
    Ok(())
}

/// The VM runner invokes phases in separate boots; state survives in the guest disk.
#[test]
#[ignore = "Requires a disposable rehearsal VM and orchestration across real reboots"]
fn actual_reboot_recovery() -> Result<()> {
    admin()?;
    ensure!(
        Path::new("/etc/vortex-rehearsal-guest").is_file(),
        "Reboot rehearsal requires a disposable guest"
    );
    let state = Path::new("/var/lib/vortex-rehearsal.json");
    let boot = fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    let phase = std::env::var("VORTEX_REBOOT_PHASE")?;
    if phase == "prepare" {
        ensure!(
            !state.exists(),
            "Existing rehearsal state must be inspected"
        );
        let user = format!("vortex-test-{}", &token()[..10]);
        checked(
            Command::new("useradd").args(["-m", "-s", "/bin/bash", &user]),
            20,
        )?;
        let home = PathBuf::from(format!("/home/{user}"));
        let s = Sandbox {
            root: home.join("vision.deployment"),
            unit: format!("{user}.service"),
            user,
            home,
        };
        fs::create_dir_all(&s.root)?;
        fs::create_dir_all(s.home.join(".local/state/vortex"))?;
        fs::write(s.home.join(".local/state/vortex/deployment.lock"), "")?;
        checked(
            Command::new("chown")
                .arg("-R")
                .arg(format!("{}:{}", s.user, s.user))
                .arg(s.home.join(".local")),
            20,
        )?;
        let first = s.prepare("healthy")?;
        let second = s.prepare("healthy")?;
        checked(
            Command::new(controller())
                .args(["install-service", "--root"])
                .arg(&s.root)
                .arg("--legacy-root")
                .arg(s.home.join("vision"))
                .args(["--user", &s.user, "--service", &s.unit]),
            30,
        )?;
        let account = checked(Command::new("id").args(["-u", &s.user]), 10)?;
        let uid = String::from_utf8(account.stdout)?.trim().to_string();
        checked(
            Command::new("systemctl").args(["start", &format!("user@{uid}.service")]),
            30,
        )?;
        ensure!(
            s.submit("boot-first", Some(&first))?["state"] == "active",
            "Initial activation failed"
        );
        atomic(
            state,
            &json!({"user":s.user,"home":s.home,"root":s.root,"unit":s.unit,"first":first,"second":second,"boot":boot,"phase":"prepared"}),
        )?;
        std::mem::forget(s);
    } else {
        let mut v = read(state)?;
        ensure!(
            string(&v, "boot")? != boot,
            "Guest has not actually rebooted"
        );
        let user = string(&v, "user")?.to_string();
        ensure!(
            user.starts_with("vortex-test-") && user.len() == 22,
            "Invalid rehearsal account"
        );
        let home = PathBuf::from(format!("/home/{user}"));
        ensure!(
            v["home"] == home.to_string_lossy().as_ref()
                && v["root"] == home.join("vision.deployment").to_string_lossy().as_ref()
                && v["unit"] == format!("{user}.service"),
            "Invalid rehearsal paths"
        );
        let s = Sandbox {
            root: home.join("vision.deployment"),
            unit: format!("{user}.service"),
            user,
            home,
        };
        let expected_phase = if phase == "interrupt" {
            "prepared"
        } else {
            "interrupted"
        };
        ensure!(v["phase"] == expected_phase, "Unexpected reboot phase");
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Ok(h) = read(s.root.join("health.json")) {
                let status = checked(
                    Command::new("systemctl").args([
                        "show",
                        "--property=MainPID",
                        "--value",
                        &s.unit,
                    ]),
                    10,
                )?;
                let pid = String::from_utf8(status.stdout)?
                    .trim()
                    .parse::<u64>()
                    .unwrap_or(0);
                if pid > 0 && h["pid"].as_u64() == Some(pid) && h["boot_id"] == boot.trim() {
                    break;
                }
            }
            ensure!(
                Instant::now() < deadline,
                "Fresh runtime health did not recover after reboot"
            );
            std::thread::sleep(Duration::from_millis(250));
        }
        ensure!(
            read(s.root.join("selection.json"))?["release_id"] == v["first"],
            "Boot selected an unconfirmed release"
        );
        ensure!(
            !s.root.join("activation.json").exists(),
            "Recovery journal was not cleared"
        );
        if phase == "interrupt" {
            checked(Command::new("systemctl").args(["stop", &s.unit]), 20)?;
            let old = read(s.root.join("selection.json"))?;
            let next = json!({"kind":"release","release_id":v["second"],"launch_token":"reboot-interrupted"});
            fs::remove_file(s.root.join("current"))?;
            std::os::unix::fs::symlink(
                s.root.join("releases").join(string(&v, "second")?),
                s.root.join("current"),
            )?;
            atomic(s.root.join("selection.json"), &next)?;
            atomic(
                s.root.join("activation.json"),
                &json!({"schema_version":1,"operation_id":"reboot-interrupted","phase":"verifying","old":old,"new":next,"previous_before":{"kind":"none"},"boot_id":boot.trim(),"pid":0,"process_start":"0"}),
            )?;
            v["boot"] = json!(boot);
            v["phase"] = json!("interrupted");
            atomic(state, &v)?;
            std::mem::forget(s);
            println!("REBOOT PASS: selected service and fresh runtime health recovered after a real guest reboot");
        } else {
            ensure!(phase == "recover", "Unknown reboot phase");
            fs::remove_file(state)?;
            println!("REBOOT RECOVERY PASS: interrupted activation restored the confirmed release after a second real guest reboot");
        }
    }
    Ok(())
}
