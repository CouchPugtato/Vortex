use crate::{activation, common::*};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
pub struct Account {
    pub uid: u32,
    pub gid: u32,
    pub home: PathBuf,
}
pub fn account(user: &str) -> Result<Account> {
    simple(user)?;
    let o = checked(Command::new("getent").args(["passwd", user]), 10)?;
    let text = String::from_utf8(o.stdout)?;
    let p: Vec<_> = text.trim().split(':').collect();
    ensure!(p.len() == 7 && p[0] == user, "Invalid runtime account");
    Ok(Account {
        uid: p[2].parse()?,
        gid: p[3].parse()?,
        home: PathBuf::from(p[5]),
    })
}
pub fn chown(path: &Path, a: &Account) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let p = std::ffi::CString::new(path.as_os_str().as_bytes())?;
        ensure!(
            unsafe { libc::chown(p.as_ptr(), a.uid, a.gid) } == 0,
            "Cannot set runtime ownership: {}",
            path.display()
        );
    }
    let _ = (path, a);
    Ok(())
}
pub fn install(root: &Path, legacy: &Path, name: &str, user: &str) -> Result<Value> {
    admin()?;
    service(name)?;
    for s in [
        root.to_str().context("Root path")?,
        legacy.to_str().context("Legacy path")?,
        user,
    ] {
        simple(s)?;
    }
    let a = account(user)?;
    ensure!(a.uid != 0, "Runtime account must not be root");
    let lock = a.home.join(".local/state/vortex/deployment.lock");
    ensure!(
        lock.exists(),
        "Stage a release before installing the service"
    );
    let _lock = Lock::acquire(&lock)?;
    let configured = optional(root.join("managed-service.json"), Value::Null)?;
    ensure!(
        configured.is_null()
            || (configured["service"] == name
                && configured["user"] == user
                && configured["legacy_root"] == legacy.to_str().unwrap()),
        "Managed service identity cannot change"
    );
    if root.join("activation.json").exists() {
        ensure!(
            !configured.is_null(),
            "Recover unfinished activation before installing controller"
        );
        let mut c = configured;
        c["controller_update_deferred"] = json!(true);
        return Ok(c);
    }
    let dropins = checked(
        Command::new("systemctl").args(["show", name, "-p", "DropInPaths", "--value"]),
        15,
    )?;
    ensure!(
        String::from_utf8_lossy(&dropins.stdout).trim().is_empty(),
        "Existing service has drop-ins; migrate them explicitly"
    );
    let mut baseline = json!({"kind":"none"});
    if configured.is_null() {
        let running = output(
            Command::new("systemctl").args(["is-active", "--quiet", name]),
            15,
        )?
        .status
        .success();
        if running {
            let o = checked(
                Command::new("systemctl").args(["show", name, "-p", "ExecStart", "--value"]),
                15,
            )?;
            ensure!(
                String::from_utf8_lossy(&o.stdout)
                    .contains(&format!("path={}/run_vortex.sh ;", legacy.display())),
                "Existing service is not the expected legacy launcher"
            );
            baseline = json!({"kind":"legacy","path":legacy});
        }
    }
    let unit_path = Path::new("/etc/systemd/system").join(name);
    let original = if unit_path.exists() {
        Some(fs::read(&unit_path)?)
    } else {
        None
    };
    if configured.is_null() {
        if let Some(bytes) = &original {
            atomic_bytes(&root.join("legacy-service.unit"), bytes, 0o644)?;
        }
    }
    copy(&executable()?, &root.join("vortexctl"), 0o755)?;
    chown(&root.join("vortexctl"), &a)?;
    let mut config = json!({"schema_version":1,"service":name,"user":user,"legacy_root":legacy,"installed":false});
    atomic(root.join("managed-service.json"), &config)?;
    if configured.is_null() {
        atomic(root.join("selection.json"), &baseline)?;
        atomic(root.join("active.json"), &baseline)?;
        atomic(root.join("previous.json"), &json!({"kind":"none"}))?;
    }
    for n in [
        "managed-service.json",
        "selection.json",
        "active.json",
        "previous.json",
    ] {
        chown(&root.join(n), &a)?;
    }
    let unit=format!("[Unit]\nDescription=Vortex managed vision release\nAfter=network-online.target\nWants=network-online.target\nStartLimitIntervalSec=60\nStartLimitBurst=5\n\n[Service]\nType=simple\nNotifyAccess=main\nWatchdogSec=0\nWatchdogSignal=SIGTERM\nTimeoutAbortSec=5\nUser={user}\nGroup={}\nExecStart={}/vortexctl service-run --root {}\nRestart=on-failure\nRestartSec=2\nKillMode=control-group\nKillSignal=SIGINT\nTimeoutStopSec=10\n\n[Install]\nWantedBy=multi-user.target\n",a.gid,root.display(),root.display());
    atomic_bytes(&unit_path, unit.as_bytes(), 0o644)?;
    let result = (|| -> Result<()> {
        let sudoers = Path::new("/etc/sudoers.d").join(format!(
            "vortex-{}",
            name.trim_end_matches(".service").replace('.', "_")
        ));
        let temp = sudoers.with_extension("new");
        let systemctl = checked(Command::new("which").arg("systemctl"), 10)?;
        let executable = String::from_utf8(systemctl.stdout)?.trim().to_string();
        simple(&executable)?;
        let permissions = format!(
            "{user} ALL=(root) NOPASSWD: {}\n",
            ["start", "stop", "reset-failed"]
                .iter()
                .map(|a| format!("{executable} {a} {name}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        atomic_bytes(&temp, permissions.as_bytes(), 0o440)?;
        let validation = checked(Command::new("visudo").arg("-cf").arg(&temp), 15);
        if validation.is_err() {
            let _ = fs::remove_file(&temp);
        }
        validation?;
        fs::rename(temp, sudoers)?;
        checked(Command::new("systemctl").arg("daemon-reload"), 20)?;
        checked(Command::new("systemctl").args(["enable", name]), 20)?;
        checked(Command::new("loginctl").args(["enable-linger", user]), 20)?;
        Ok(())
    })();
    if let Err(e) = result {
        if let Some(bytes) = original {
            atomic_bytes(&unit_path, &bytes, 0o644)?;
        } else {
            fs::remove_file(&unit_path)?;
        }
        checked(Command::new("systemctl").arg("daemon-reload"), 20)?;
        return Err(e);
    }
    config["installed"] = json!(true);
    atomic(root.join("managed-service.json"), &config)?;
    chown(&root.join("managed-service.json"), &a)?;
    Ok(config)
}
pub fn diagnostics(
    root: &Path,
    camera: u64,
    client: &str,
    stop: bool,
    m: &dyn activation::Manager,
) -> Result<Value> {
    diagnostics_options(root, camera, client, stop, m, false)
}
pub fn diagnostics_options(
    root: &Path,
    camera: u64,
    client: &str,
    stop: bool,
    m: &dyn activation::Manager,
    radio: bool,
) -> Result<Value> {
    ensure!(
        camera <= 63
            && !client.is_empty()
            && client.len() <= 96
            && client
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "Invalid diagnostics camera/client"
    );
    let dir = child(root, "diagnostics")?;
    let request = dir.join("request.json");
    if stop {
        if optional(&request, json!({}))?["client_id"] == client {
            fs::remove_file(&request)?;
        }
        return Ok(json!({"schema_version":1,"detached":true}));
    }
    let info = m.info()?;
    let mut result = json!({"schema_version":1,"ready":false,"camera_index":camera,"service":info,"reason":"Service is not running"});
    if info["active"] != true || info["pid"].as_u64().unwrap_or(0) == 0 {
        return Ok(result);
    }
    let sel = activation::selected(root)?;
    if sel["kind"] != "release" {
        result["reason"] = json!("Activate a managed release to use service diagnostics");
        return Ok(result);
    }
    let report = optional(root.join("health.json"), json!({}))?;
    let now = (now() * 1000.) as u64;
    let fresh = |v: &Value| {
        v["timestamp_ms"]
            .as_u64()
            .is_some_and(|t| t <= now && now - t <= 1500)
    };
    if report["schema_version"] != 1
        || report["release_id"] != sel["release_id"]
        || report["launch_token"] != sel["launch_token"]
        || report["pid"] != info["pid"]
        || report["diagnostics_version"] != 1
        || !fresh(&report)
    {
        result["reason"]=json!("Waiting for current runtime health; older releases need rebuilding with service diagnostics");
        return Ok(result);
    }
    let camera_health = report["cameras"]
        .as_array()
        .context("Missing camera health")?
        .iter()
        .find(|c| c["index"].as_u64() == Some(camera))
        .context("Selected camera is not configured")?
        .clone();
    result["health"] = report.clone();
    result["camera_health"] = camera_health.clone();
    result["reason"] = json!("Waiting for a processed preview");
    fs::create_dir_all(&dir)?;
    let previous = optional(&request, json!({}))?;
    if previous["client_id"] != client
        && previous["pid"] == info["pid"]
        && previous["launch_token"] == sel["launch_token"]
        && previous["expires_ms"]
            .as_u64()
            .is_some_and(|t| t > now && t <= now + 5000)
    {
        result["reason"] = json!("Another diagnostics client has the preview lease");
        return Ok(result);
    }
    atomic(
        &request,
        &json!({"schema_version":1,"release_id":sel["release_id"],"launch_token":sel["launch_token"],"pid":info["pid"],"camera_index":camera,"client_id":client,"expires_ms":now+3000,"radio":radio}),
    )?;
    let mut state = optional(dir.join("preview.json"), json!({}))?;
    if state["schema_version"] != 1
        || state["pid"] != info["pid"]
        || state["release_id"] != sel["release_id"]
        || state["launch_token"] != sel["launch_token"]
        || state["camera_index"].as_u64() != Some(camera)
        || state["client_id"] != client
        || !fresh(&state)
        || !state["latency_ms"].as_f64().is_some_and(|n| {
            n >= 0. && n + (now - state["timestamp_ms"].as_u64().unwrap()) as f64 <= 1500.
        })
    {
        return Ok(result);
    }
    let name = string(&state, "frame_id")?;
    let parts: Vec<_> = name
        .strip_prefix("frame-")
        .and_then(|s| s.strip_suffix(".jpg"))
        .context("Invalid preview filename")?
        .split('-')
        .collect();
    ensure!(
        parts.len() == 2
            && parts
                .iter()
                .all(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())),
        "Invalid preview filename"
    );
    let image = child(&dir, name)?;
    match fs::metadata(&image) {
        Ok(m) => ensure!(
            m.len() > 0 && m.len() <= 512 * 1024,
            "Preview image exceeds limits"
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(result),
        Err(e) => return Err(e.into()),
    }
    let bytes = match fs::read(&image) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(result),
        Err(e) => return Err(e.into()),
    };
    ensure!(
        bytes.len() <= 512 * 1024 && bytes.starts_with(b"\xff\xd8"),
        "Invalid or oversized preview JPEG"
    );
    if activation::selected(root)? != sel || m.info()?["pid"] != info["pid"] {
        result["reason"] = json!("Runtime changed while reading preview");
        return Ok(result);
    }
    state["fps"] = camera_health["fps"].clone();
    state["frame_age_ms"] = json!(
        state["latency_ms"].as_f64().unwrap()
            + (now - state["timestamp_ms"].as_u64().unwrap()) as f64
    );
    state["watchdog"] = report["watchdog"].clone();
    use base64::Engine;
    result["jpeg_base64"] = json!(base64::engine::general_purpose::STANDARD.encode(bytes));
    result["state"] = state;
    result["ready"] = json!(true);
    result["reason"] = json!("Production service attached");
    Ok(result)
}
