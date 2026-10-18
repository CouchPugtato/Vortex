use crate::common::*;
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{fs, path::Path, process::Command};
pub const TERMINAL: [&str; 7] = [
    "ready",
    "active",
    "failed",
    "rolled_back",
    "rollback_failed",
    "recovered",
    "interrupted",
];
pub fn unit(root: &Path, identity: &str) -> String {
    format!(
        "vortex-job-{}.service",
        &hash_bytes(format!("{}/{identity}", root.display()).as_bytes())[..32]
    )
}
pub fn submit(
    root: &Path,
    identity: &str,
    kind: &str,
    release: Option<&str>,
    sha: Option<&str>,
) -> Result<Value> {
    linux()?;
    id(identity)?;
    simple(root.to_str().unwrap_or(""))?;
    ensure!(
        ["stage", "activate", "rollback"].contains(&kind),
        "Invalid job kind"
    );
    let mut check = Command::new("loginctl");
    user_env(&mut check);
    let o = checked(
        check.args(["show-user", &uid().to_string(), "-p", "Linger", "--value"]),
        15,
    )?;
    ensure!(
        String::from_utf8_lossy(&o.stdout).trim() == "yes",
        "Persistent jobs need user lingering; use Initialize Jetson"
    );
    let mut check = Command::new("systemctl");
    user_env(&mut check);
    checked(check.args(["--user", "show-environment"]), 15)?;
    let executable = if kind == "stage" {
        ensure!(release == Some(identity), "Stage identity mismatch");
        checksum(sha.unwrap_or(""))?;
        let expected = child(root, &format!("incoming/{identity}/vortexctl"))?;
        ensure!(
            fs::canonicalize(executable()?)? == fs::canonicalize(&expected)?,
            "Stage must use its verified uploaded controller"
        );
        expected
    } else {
        root.join("vortexctl")
    };
    let mut args = vec![
        kind.to_string(),
        "--root".into(),
        root.display().to_string(),
    ];
    if kind == "stage" {
        args.extend([
            "--archive".into(),
            root.join("incoming")
                .join(identity)
                .join("release.tar.gz")
                .display()
                .to_string(),
            "--sha256".into(),
            sha.unwrap().into(),
            "--release-id".into(),
            identity.into(),
            "--with-debug-bridge".into(),
        ]);
    } else {
        args.extend(["--operation-id".into(), identity.into()]);
        if kind == "activate" {
            args.extend(["--release-id".into(), id(release.unwrap_or(""))?.into()]);
        }
    }
    dispatch(root, identity, kind, false, &executable, &args)
}
pub fn dispatch(
    root: &Path,
    identity: &str,
    kind: &str,
    system: bool,
    exe: &Path,
    args: &[String],
) -> Result<Value> {
    claim(root, identity, "dispatch")?;
    let unit = unit(root, identity);
    let path = root.join("jobs").join(format!("{identity}.unit.json"));
    let mut data = json!({"schema_version":1,"job_id":identity,"kind":kind,"unit":unit,"submitted_at":now(),"dispatch":"submitting","manager":if system{"system"}else{"user"}});
    atomic(&path, &data)?;
    let log = root.join("jobs").join(format!("{identity}.controller.log"));
    if system {
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&log)?;
        mode(&log, 0o644)?;
    }
    let mut cmd = Command::new("systemd-run");
    if !system {
        cmd.arg("--user");
        user_env(&mut cmd);
    }
    cmd.args([
        format!("--unit={unit}"),
        format!("--description=Vortex {kind}"),
        "--property=Type=exec".into(),
        "--property=Restart=no".into(),
        "--property=KillMode=control-group".into(),
        "--property=TimeoutStopSec=20".into(),
        format!(
            "--property=RuntimeMaxSec={}",
            if system {
                10800
            } else if kind == "stage" {
                7200
            } else {
                300
            }
        ),
        "--property=Nice=10".into(),
        format!("--property=WorkingDirectory={}", root.display()),
        "--property=StandardInput=null".into(),
        format!("--property=StandardOutput=append:{}", log.display()),
        "--property=StandardError=inherit".into(),
        "--".into(),
    ])
    .arg(exe)
    .args(args);
    checked(&mut cmd, 30)?;
    data["dispatch"] = json!("submitted");
    atomic(path, &data)?;
    Ok(data)
}
pub fn status(root: &Path, identity: &str, mut job: Value) -> Result<Value> {
    id(identity)?;
    let path = child(root, &format!("jobs/{identity}.unit.json"))?;
    if path.exists() {
        let d = read(path)?;
        let expected = unit(root, identity);
        ensure!(
            d["unit"] == expected
                && d["job_id"] == identity
                && [Some("user"), Some("system"), None].contains(&d["manager"].as_str())
                && (d["manager"] != "system" || d["kind"] == "initialize"),
            "Invalid job manager descriptor"
        );
        let mut cmd = Command::new("systemctl");
        if d["manager"] != "system" {
            cmd.arg("--user");
            user_env(&mut cmd);
        }
        let o = output(
            cmd.args([
                "show",
                &expected,
                "-p",
                "ActiveState",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
                "-p",
                "LoadState",
            ]),
            15,
        )?;
        let v = properties(&o.stdout);
        ensure!(
            o.status.success() || v["LoadState"] == "not-found",
            "Cannot inspect job manager; reconnect before retrying"
        );
        job["unit"] = json!(expected);
        job["unit_state"] = v["ActiveState"].clone();
        if !TERMINAL.contains(&job["state"].as_str().unwrap_or(""))
            && !["active", "activating", "reloading", "deactivating"]
                .contains(&v["ActiveState"].as_str().unwrap_or(""))
            && (d["dispatch"] == "submitted"
                || now() - d["submitted_at"].as_f64().unwrap_or(now()) > 30.)
        {
            job["state"] = json!("interrupted");
            job["error"] = json!(
                "Deployment worker stopped before completing; inspect logs and recovery state"
            );
            job["unit_result"] = v["Result"].clone();
            job["exit_status"] = v["ExecMainStatus"].clone();
        }
    }
    let stage_log = if job["action"] == "initialize" && job["state"] == "building" {
        job["release_id"]
            .as_str()
            .filter(|s| id(s).is_ok())
            .map(|s| format!("jobs/{s}.log"))
    } else {
        None
    };
    for relative in stage_log.into_iter().chain([
        format!("jobs/{identity}.log"),
        format!("jobs/{identity}.controller.log"),
    ]) {
        let p = child(root, &relative)?;
        if p.is_file() {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = fs::File::open(p)?;
            f.seek(SeekFrom::Start(f.metadata()?.len().saturating_sub(16384)))?;
            let mut b = Vec::new();
            f.take(16384).read_to_end(&mut b)?;
            job["log_tail"] = json!(String::from_utf8_lossy(&b));
            break;
        }
    }
    Ok(job)
}
