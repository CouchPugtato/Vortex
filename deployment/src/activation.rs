use crate::{common::*, launcher, package};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};
pub fn ready(root: &Path, identity: &str) -> Result<(PathBuf, Value)> {
    id(identity)?;
    let dir = child(root, &format!("releases/{identity}"))?;
    let m = read(dir.join("manifest.json"))?;
    package::validate_manifest(&m, identity)?;
    let meta = read(dir.join("release.json"))?;
    ensure!(
        meta["release_id"] == identity && meta["profile"] == m["build_profile"],
        "Invalid ready release"
    );
    Ok((dir, m))
}
pub fn preflight(root: &Path, identity: &str) -> Result<Value> {
    linux()?;
    let (dir, m) = ready(root, identity)?;
    package::verify(&dir, &m)?;
    let c = read(dir.join("config/device.json"))?;
    launcher::resources(&dir, &c)?;
    let binary = launcher::checked_binary(&dir, &c, launcher::MAIN)?;
    package::elf(&binary, Some(std::env::consts::ARCH))?;
    let libraries = output(Command::new("ldd").arg(&binary).current_dir(&dir), 30)?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&libraries.stdout),
        String::from_utf8_lossy(&libraries.stderr)
    );
    ensure!(
        !text.contains("not found")
            && (libraries.status.success()
                || text.contains("not a dynamic executable")
                || text.contains("statically linked")),
        "Runtime libraries unavailable: {text}"
    );
    let smoke = checked(
        Command::new(&binary)
            .arg("--deployment-smoke")
            .arg(dir.join("simulation_tests/reference/dim_noisy_tag.png"))
            .current_dir(&dir),
        30,
    )?;
    let s = String::from_utf8_lossy(&smoke.stdout);
    let reports: Vec<_> = s
        .lines()
        .filter_map(|l| l.strip_prefix("VORTEX_SMOKE "))
        .collect();
    ensure!(reports.len() == 1, "Expected exactly one smoke report");
    let report: Value = serde_json::from_str(reports[0])?;
    ensure!(
        report["passed"] == true
            && report["tag_id"].as_u64() == Some(1)
            && report["camera_error_m"]
                .as_f64()
                .is_some_and(|n| (0.0..0.2).contains(&n)),
        "Saved-image pose check failed"
    );
    for name in m["files"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .chain([
            "target/release/dumapril-taglocalization",
            "target/release/vortex-build.json",
            "release.json",
        ])
    {
        File::open(dir.join(name))?.sync_all()?;
    }
    let mut dirs: Vec<_> = walkdir::WalkDir::new(&dir)
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|e| e.file_type().is_dir())
        .map(|e| e.into_path())
        .collect();
    dirs.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for p in dirs {
        sync_dir(&p)?;
    }
    Ok(report)
}
pub trait Manager {
    fn info(&self) -> Result<Value>;
    fn stop(&self) -> Result<()>;
    fn start(&self) -> Result<()>;
}
pub struct Systemd {
    pub name: String,
}
impl Systemd {
    pub fn new(root: &Path) -> Result<Self> {
        linux()?;
        let c = read(root.join("managed-service.json"))?;
        ensure!(
            c["installed"] == true,
            "Managed service preparation is incomplete"
        );
        let name = string(&c, "service")?.to_owned();
        service(&name)?;
        Ok(Self { name })
    }
    fn control(&self, action: &str) -> Result<()> {
        let mut c = Command::new(if uid() == 0 { "systemctl" } else { "sudo" });
        if uid() != 0 {
            c.args(["-n", "systemctl"]);
        }
        checked(c.args([action, &self.name]), 30)?;
        Ok(())
    }
}
impl Manager for Systemd {
    fn info(&self) -> Result<Value> {
        let o = checked(
            Command::new("systemctl").args([
                "show",
                &self.name,
                "-p",
                "ActiveState",
                "-p",
                "MainPID",
                "-p",
                "NRestarts",
                "-p",
                "Result",
                "-p",
                "StatusText",
            ]),
            10,
        )?;
        let v = properties(&o.stdout);
        Ok(
            json!({"active":v["ActiveState"]=="active","pid":v["MainPID"].as_str().unwrap_or("0").parse::<u64>()?,"state":v["ActiveState"],"restarts":v["NRestarts"].as_str().unwrap_or("0").parse::<u64>()?,"result":v["Result"],"status":v["StatusText"]}),
        )
    }
    fn stop(&self) -> Result<()> {
        self.control("stop")?;
        let v = self.info()?;
        ensure!(v["active"] != true && v["pid"] == 0, "Service did not stop");
        Ok(())
    }
    fn start(&self) -> Result<()> {
        self.control("reset-failed")?;
        self.control("start")
    }
}
pub fn selection(root: &Path, v: Value) -> Result<Value> {
    match v["kind"].as_str() {
        Some("none") => {}
        Some("release") => {
            ready(root, string(&v, "release_id")?)?;
        }
        Some("legacy") => {
            let c = read(root.join("managed-service.json"))?;
            ensure!(
                v["path"] == c["legacy_root"] && Path::new(string(&v, "path")?).is_absolute(),
                "Invalid legacy selection"
            );
        }
        _ => anyhow::bail!("Invalid release selection"),
    };
    Ok(v)
}
pub fn selected(root: &Path) -> Result<Value> {
    selection(
        root,
        optional(root.join("selection.json"), json!({"kind":"none"}))?,
    )
}
pub fn publish(root: &Path, v: &Value) -> Result<()> {
    let pointer = root.join("current");
    if let Ok(m) = fs::symlink_metadata(&pointer) {
        ensure!(
            m.file_type().is_symlink(),
            "Current pointer is not a symlink"
        );
    }
    if v["kind"] == "release" {
        let dest = child(root, &format!("releases/{}", string(v, "release_id")?))?;
        let tmp = root.join(format!("current-{}.tmp", token()));
        #[cfg(unix)]
        std::os::unix::fs::symlink(dest, &tmp)?;
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(dest, &tmp)?;
        if let Err(e) = fs::rename(&tmp, &pointer) {
            let _ = fs::remove_file(tmp);
            return Err(e.into());
        }
    } else if fs::symlink_metadata(&pointer).is_ok() {
        fs::remove_file(pointer)?;
    }
    sync_dir(root)
}
pub fn switch(root: &Path, v: &Value) -> Result<()> {
    selection(root, v.clone())?;
    publish(root, v)?;
    atomic(root.join("selection.json"), v)
}
fn restore(root: &Path, old: &Value, previous: &Value) -> Result<()> {
    switch(root, old)?;
    atomic(root.join("active.json"), old)?;
    atomic(root.join("previous.json"), previous)
}
fn clear(root: &Path) -> Result<()> {
    let p = root.join("activation.json");
    if p.exists() {
        fs::remove_file(p)?;
    }
    sync_dir(root)
}
pub fn process_identity(pid: u64) -> Option<String> {
    let s = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    s.rsplit_once(')')?
        .1
        .split_whitespace()
        .nth(19)
        .map(str::to_owned)
}
fn boot_id() -> Result<String> {
    Ok(fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .into())
}
pub fn owner_alive(j: &Value) -> bool {
    boot_id().is_ok_and(|b| j["boot_id"] == b)
        && j["pid"]
            .as_u64()
            .and_then(process_identity)
            .is_some_and(|s| j["process_start"] == s)
}
pub fn health(root: &Path, sel: &Value, info: &Value, report: &Value) -> Result<()> {
    ensure!(
        info["active"] == true
            && info["pid"].as_u64().unwrap_or(0) > 0
            && report["pid"] == info["pid"],
        "Health process is not active service"
    );
    ensure!(
        report["schema_version"].as_u64() == Some(1)
            && report["release_id"] == sel["release_id"]
            && report["launch_token"] == sel["launch_token"],
        "Stale health identity"
    );
    let (dir, _) = ready(root, string(sel, "release_id")?)?;
    let c = read(dir.join("config/device.json"))?;
    let r = read(dir.join(string(&c, "runtime_config")?))?;
    let expected = launcher::required(&c, &r)?;
    let cams = report["cameras"]
        .as_array()
        .context("Missing camera health")?;
    let configured = c["cameras"].as_array().context("Missing camera config")?;
    ensure!(cams.len() == configured.len(), "Missing configured cameras");
    let mut seen = std::collections::HashSet::new();
    for cam in cams {
        let index = cam["index"].as_u64().context("Invalid camera index")?;
        ensure!(
            configured.contains(&json!(index)) && seen.insert(index),
            "Wrong/duplicate cameras"
        );
        ensure!(
            cam["frames"].as_u64().unwrap_or(0) > 0
                && cam["pipeline_ok"] == true
                && cam["last_frame_age_ms"]
                    .as_f64()
                    .is_some_and(|n| (0.0..=1500.).contains(&n)),
            "Stale camera/pipeline failure"
        );
        let backends = cam["backends"].as_array().context("Missing backends")?;
        ensure!(
            expected.iter().all(|s| backends.contains(&json!(s))),
            "Required detector unavailable"
        );
    }
    ensure!(
        report["uptime_ms"].as_f64().is_some_and(|n| n >= 0.),
        "Invalid health clock"
    );
    Ok(())
}
pub fn wait_healthy(root: &Path, sel: &Value, m: &dyn Manager) -> Result<Value> {
    if sel["kind"] == "none" {
        ensure!(
            m.info()?["active"] != true,
            "First-install failure left a running service"
        );
        return Ok(json!({"verification":"stopped_no_previous_release"}));
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut baseline: Option<Value> = None;
    let mut started = Instant::now();
    let mut published = Instant::now();
    let mut uptime = None;
    let mut error = "No health report".to_string();
    while Instant::now() < deadline {
        let attempt = (|| -> Result<Option<Value>> {
            let info = m.info()?;
            if sel["kind"] == "legacy" {
                ensure!(
                    info["active"] == true && info["pid"].as_u64().unwrap_or(0) > 0,
                    "Legacy process did not restart"
                );
                if baseline.as_ref().is_none_or(|b| b["pid"] != info["pid"]) {
                    baseline = Some(info);
                    started = Instant::now();
                }
                if started.elapsed() >= Duration::from_secs(3) {
                    return Ok(Some(json!({"verification":"legacy_process_only"})));
                }
                return Ok(None);
            }
            let report = read(root.join("health.json"))?;
            health(root, sel, &info, &report)?;
            let t = report["uptime_ms"].as_f64().unwrap();
            if baseline.as_ref().is_none_or(|b| b["pid"] != report["pid"])
                || uptime.is_some_and(|old| t < old)
            {
                baseline = Some(report.clone());
                started = Instant::now();
                published = Instant::now();
            }
            if uptime != Some(t) {
                uptime = Some(t);
                published = Instant::now();
            }
            ensure!(
                published.elapsed() < Duration::from_millis(1250),
                "Health reporter stopped advancing"
            );
            let b = baseline.as_ref().unwrap();
            if started.elapsed() >= Duration::from_secs(3)
                && t > b["uptime_ms"].as_f64().unwrap()
                && report["cameras"].as_array().unwrap().iter().all(|c| {
                    b["cameras"].as_array().unwrap().iter().any(|old| {
                        old["index"] == c["index"] && c["frames"].as_u64() > old["frames"].as_u64()
                    })
                })
            {
                return Ok(Some(
                    json!({"verification":"all_cameras_advancing","pid":report["pid"]}),
                ));
            }
            Ok(None)
        })();
        match attempt {
            Ok(Some(v)) => return Ok(v),
            Ok(None) => {}
            Err(e) => {
                baseline = None;
                uptime = None;
                error = e.to_string();
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    anyhow::bail!("Runtime health verification timed out: {error}")
}
pub fn recover(root: &Path, m: Option<&dyn Manager>) -> Result<Value> {
    let j = optional(root.join("activation.json"), Value::Null)?;
    if j.is_null() {
        return Ok(json!({"state":"no_recovery_needed"}));
    }
    ensure!(
        j["schema_version"] == 1
            && ["prepared", "verifying", "restoring", "confirmed"].contains(&string(&j, "phase")?),
        "Invalid activation journal"
    );
    for key in ["old", "new", "previous_before"] {
        selection(root, j[key].clone())?;
    }
    ensure!(!owner_alive(&j), "Activation is still running");
    if let Some(m) = m {
        m.stop()?;
    }
    let state = if j["phase"] == "confirmed" {
        switch(root, &j["new"])?;
        atomic(root.join("active.json"), &j["new"])?;
        atomic(root.join("previous.json"), &j["old"])?;
        "completed_confirmed_activation"
    } else {
        restore(root, &j["old"], &j["previous_before"])?;
        "restored_previous_selection"
    };
    let mut result =
        json!({"state":state,"operation_id":j["operation_id"],"runtime_health_verified":false});
    atomic(root.join("recovery.json"), &result)?;
    clear(root)?;
    if let Some(m) = m {
        let sel = selected(root)?;
        if sel["kind"] != "none" {
            m.start()?;
            let check = wait_healthy(root, &sel, m)?;
            result
                .as_object_mut()
                .unwrap()
                .extend(check.as_object().unwrap().clone());
            result["runtime_health_verified"] =
                json!(result["verification"] == "all_cameras_advancing");
            atomic(root.join("recovery.json"), &result)?;
        }
    }
    Ok(result)
}
pub fn activate(
    root: &Path,
    identity: Option<&str>,
    rollback: bool,
    operation: &str,
    m: &dyn Manager,
) -> Result<Value> {
    activate_with(
        root,
        identity,
        rollback,
        operation,
        m,
        &preflight,
        &wait_healthy,
        &lock_path()?,
    )
}
// Explicit dependencies keep transaction failure tests isolated from host services.
#[allow(clippy::too_many_arguments)]
pub fn activate_with(
    root: &Path,
    identity: Option<&str>,
    rollback: bool,
    operation: &str,
    m: &dyn Manager,
    preflight: &dyn Fn(&Path, &str) -> Result<Value>,
    check: &dyn Fn(&Path, &Value, &dyn Manager) -> Result<Value>,
    lock: &Path,
) -> Result<Value> {
    id(operation)?;
    claim(root, operation, "claim")?;
    let path = root.join("jobs").join(format!("{operation}.json"));
    let mut job = json!({"operation_id":operation,"state":"queued","action":if rollback{"rollback"}else{"activate"}});
    atomic(&path, &job)?;
    let result = (|| -> Result<Value> {
        let _lock = Lock::acquire(lock)?;
        if root.join("activation.json").exists() {
            recover(root, Some(m))?;
        }
        let old = selected(root)?;
        ensure!(
            old == optional(root.join("active.json"), json!({"kind":"none"}))?,
            "Selection differs from committed state; recover before activation"
        );
        let previous = selection(
            root,
            optional(root.join("previous.json"), json!({"kind":"none"}))?,
        )?;
        let mut new = if rollback {
            ensure!(previous["kind"] != "none", "No previous release available");
            previous.clone()
        } else {
            json!({"kind":"release","release_id":id(identity.unwrap_or(""))?})
        };
        if new["kind"] == "release" {
            ensure!(
                old["release_id"] != new["release_id"],
                "Release already selected"
            );
            job["state"] = json!("validating");
            job["release_id"] = new["release_id"].clone();
            atomic(&path, &job)?;
            preflight(root, string(&new, "release_id")?)?;
        }
        new["launch_token"] = json!(token());
        let mut journal = json!({"schema_version":1,"operation_id":operation,"phase":"prepared","old":old,"new":new,"previous_before":previous,"boot_id":boot_id()?,"pid":std::process::id(),"process_start":process_identity(std::process::id() as u64)});
        atomic(root.join("activation.json"), &journal)?;
        let mut committed = false;
        let mut verified = Value::Null;
        let transaction = (|| -> Result<()> {
            job["state"] = json!("stopping");
            atomic(&path, &job)?;
            m.stop()?;
            job["state"] = json!("switching");
            atomic(&path, &job)?;
            switch(root, &new)?;
            journal["phase"] = json!("verifying");
            atomic(root.join("activation.json"), &journal)?;
            m.start()?;
            job["state"] = json!("verifying");
            atomic(&path, &job)?;
            verified = check(root, &new, m)?;
            atomic(root.join("previous.json"), &old)?;
            atomic(root.join("active.json"), &new)?;
            journal["phase"] = json!("confirmed");
            atomic(root.join("activation.json"), &journal)?;
            committed = true;
            job["state"] = json!("active");
            job["selection"] = new.clone();
            job.as_object_mut()
                .unwrap()
                .extend(verified.as_object().unwrap().clone());
            atomic(&path, &job)?;
            clear(root)
        })();
        if let Err(failure) = transaction {
            if committed {
                job["state"] = json!("active");
                job["selection"] = new;
                job["warning"] = json!(format!("{failure:#}"));
                job.as_object_mut()
                    .unwrap()
                    .extend(verified.as_object().unwrap().clone());
                return Ok(job.clone());
            }
            job["error"] = json!(format!("{failure:#}"));
            let rollback = (|| -> Result<()> {
                let unchanged = selected(root)? == old && m.info()?["active"] == true;
                let mut restored = old.clone();
                if !unchanged {
                    m.stop()?;
                    restored["launch_token"] = json!(token());
                }
                restore(root, &restored, &previous)?;
                journal["old"] = restored.clone();
                journal["phase"] = json!("restoring");
                atomic(root.join("activation.json"), &journal)?;
                if !unchanged && restored["kind"] != "none" {
                    m.start()?;
                }
                let checked = check(root, &restored, m)?;
                clear(root)?;
                job["state"] = json!("rolled_back");
                job["selection"] = restored;
                job.as_object_mut()
                    .unwrap()
                    .extend(checked.as_object().unwrap().clone());
                Ok(())
            })();
            if let Err(e) = rollback {
                job["state"] = json!("rollback_failed");
                job["rollback_error"] = json!(format!("{e:#}"));
            }
            atomic(&path, &job)?;
        }
        Ok(job.clone())
    })();
    match result {
        Ok(v) => Ok(v),
        Err(e) => {
            job["state"] = json!("failed");
            job["error"] = json!(format!("{e:#}"));
            atomic(&path, &job)?;
            Err(e)
        }
    }
}
pub fn service_run(root: &Path) -> Result<()> {
    linux()?;
    let j = optional(root.join("activation.json"), Value::Null)?;
    if !j.is_null() && !owner_alive(&j) {
        let _lock = Lock::acquire(&lock_path()?)?;
        recover(root, None)?;
    }
    let sel = selected(root)?;
    ensure!(sel["kind"] != "none", "No selected release");
    if !root.join("activation.json").exists() {
        ensure!(
            sel == read(root.join("active.json"))?,
            "Uncommitted release selection"
        );
    }
    if sel["kind"] == "release" {
        let (dir, _) = ready(root, string(&sel, "release_id")?)?;
        ensure!(
            fs::symlink_metadata(root.join("current"))?
                .file_type()
                .is_symlink()
                && fs::canonicalize(root.join("current"))? == fs::canonicalize(&dir)?,
            "Release pointer mismatch"
        );
        let c = read(dir.join("config/device.json"))?;
        let r = launcher::resources(&dir, &c)?;
        let mut cmd = launcher::command(&dir, &c, None)?;
        cmd.env("VORTEX_HEALTH_PATH", root.join("health.json"))
            .env("VORTEX_DIAGNOSTICS_DIR", root.join("diagnostics"))
            .env("VORTEX_SERVICE_WATCHDOG", "1")
            .env("VORTEX_RELEASE_ID", string(&sel, "release_id")?)
            .env("VORTEX_LAUNCH_TOKEN", string(&sel, "launch_token")?)
            .env(
                "VORTEX_REQUIRED_BACKENDS",
                launcher::required(&c, &r)?.join(","),
            );
        launcher::exec(cmd)
    } else {
        let dir = Path::new(string(&sel, "path")?);
        let mut cmd = Command::new("/bin/bash");
        cmd.arg(dir.join("run_vortex.sh")).current_dir(dir);
        launcher::exec(cmd)
    }
}
pub fn overview(root: &Path) -> Result<Value> {
    let mut releases = Vec::new();
    if root.join("releases").exists() {
        for entry in fs::read_dir(root.join("releases"))? {
            let e = entry?;
            let name = e.file_name().to_string_lossy().to_string();
            if ready(root, &name).is_ok() {
                releases.push(name);
            }
        }
    }
    releases.sort_by(|a, b| b.cmp(a));
    let current = selected(root)?;
    let cameras = if current["kind"] == "release" {
        let (dir, _) = ready(root, string(&current, "release_id")?)?;
        read(dir.join("config/device.json"))?["cameras"].clone()
    } else {
        json!([])
    };
    let installed = root.join("vortexctl");
    Ok(
        json!({"ready":releases,"active":optional(root.join("active.json"),json!({"kind":"none"}))?,"previous":optional(root.join("previous.json"),json!({"kind":"none"}))?,"pending":optional(root.join("activation.json"),Value::Null)?,"configured_cameras":cameras,"service_configured":optional(root.join("managed-service.json"),json!({}))?["installed"]==true,"controller_current":installed.is_file() && hash(installed)?==hash(executable()?)?}),
    )
}
pub fn operation_status(root: &Path, identity: &str) -> Result<Value> {
    id(identity)?;
    let mut job = optional(
        root.join("jobs").join(format!("{identity}.json")),
        json!({"operation_id":identity,"state":"starting"}),
    )?;
    let recovery = optional(root.join("recovery.json"), Value::Null)?;
    if recovery["operation_id"] == identity
        && [
            "queued",
            "stopping",
            "switching",
            "verifying",
            "rollback_failed",
        ]
        .contains(&job["state"].as_str().unwrap_or(""))
    {
        job["state"] = json!("recovered");
        job["recovery"] = recovery;
        job["error"] =
            json!("Interrupted activation recovered; refresh and inspect runtime health");
    }
    Ok(job)
}
