//! Transactional robot Ethernet setup. NetworkManager owns the rollback timer.
use crate::common::*;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

fn nm(args: &[&str]) -> Result<String> {
    Ok(String::from_utf8(
        checked(Command::new("nmcli").env("LC_ALL", "C").args(args), 30)
            .context(
                "NetworkManager command failed; Initialize Jetson over USB first if it is missing",
            )?
            .stdout,
    )?
    .trim()
    .into())
}
fn field(device: &str, key: &str) -> Result<String> {
    nm(&["--get-values", key, "device", "show", device])
}
pub fn address(team: u32, device: u32) -> Result<String> {
    ensure!(
        (1..=25599).contains(&team) && (6..=19).contains(&device),
        "Use team 1–25599 and an unused device address 6–19"
    );
    Ok(format!("10.{}.{}.{device}", team / 100, team % 100))
}
pub fn interface(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.len() <= 15
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
        "Invalid Ethernet interface"
    );
    ensure!(
        !["usb0", "usb1", "rndis0", "l4tbr0", "lo"].contains(&name),
        "USB recovery and loopback interfaces cannot be changed"
    );
    Ok(())
}
pub fn info() -> Result<Value> {
    linux()?;
    let rows = nm(&[
        "--terse",
        "--escape",
        "no",
        "--fields",
        "DEVICE,TYPE,STATE",
        "device",
        "status",
    ])?;
    let mut devices = Vec::new();
    for row in rows.lines() {
        let parts: Vec<_> = row.splitn(3, ':').collect();
        if parts.len() != 3 || parts[1] != "ethernet" {
            continue;
        }
        let ips = field(parts[0], "IP4.ADDRESS")?;
        let eligible = parts[2] != "unmanaged"
            && interface(parts[0]).is_ok()
            && !ips.lines().any(|a| a.starts_with("192.168.55."));
        devices.push(json!({"interface":parts[0],"state":parts[2],"addresses":ips.lines().collect::<Vec<_>>(),"eligible":eligible}));
    }
    let hostname = String::from_utf8(checked(&mut Command::new("hostname"), 10)?.stdout)?
        .trim()
        .to_string();
    Ok(json!({"schema_version":1,"hostname":hostname,"interfaces":devices}))
}
fn base() -> Result<PathBuf> {
    let p = PathBuf::from("/var/lib/vortex-network");
    if p.exists() {
        regular_directory(&p)?;
    } else {
        fs::create_dir(&p)?;
        mode(&p, 0o755)?;
    }
    Ok(p)
}
fn regular_directory(p: &Path) -> Result<()> {
    ensure!(
        fs::symlink_metadata(p)?.is_dir(),
        "Network state directory cannot be a symlink"
    );
    Ok(())
}
fn directory(operation: &str) -> Result<PathBuf> {
    id(operation)?;
    let p = child(Path::new("/var/lib/vortex-network"), operation)?;
    regular_directory(&p)?;
    Ok(p)
}
fn state(dir: &Path, status: &str, extra: Value) -> Result<Value> {
    let mut v = read(dir.join("status.json"))?;
    v["state"] = json!(status);
    v.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    atomic(dir.join("status.json"), &v)?;
    Ok(v)
}
pub fn status(operation: &str) -> Result<Value> {
    read(directory(operation)?.join("status.json"))
}

fn install_radio_ssh(user: &str, port: u16) -> Result<()> {
    crate::service::account(user)?;
    ensure!(
        user != "root" && (5800..=5810).contains(&port),
        "Use a non-root account and unused TCP team port 5800–5810"
    );
    let config = Path::new("/etc/ssh/vortex-radio.conf");
    let unit = Path::new("/etc/systemd/system/vortex-radio-ssh.service");
    // A separate listener preserves the image's existing SSH service and USB port 22.
    let content = format!("Port {port}\nListenAddress 0.0.0.0\nPidFile /run/vortex-radio-sshd.pid\nUsePAM yes\nPermitRootLogin no\nAllowUsers {user}\nPasswordAuthentication yes\nPubkeyAuthentication yes\nAuthorizedKeysFile .ssh/authorized_keys\nX11Forwarding no\nAllowTcpForwarding no\nPermitTunnel no\nSubsystem sftp internal-sftp\n");
    for path in [config, unit] {
        ensure!(
            !path.exists() || fs::symlink_metadata(path)?.is_file(),
            "SSH configuration must be a regular file"
        );
    }
    let previous = fs::read(config).ok();
    let previous_unit = fs::read(unit).ok();
    atomic_bytes(config, content.as_bytes(), 0o600)?;
    let result = (|| -> Result<()> {
        checked(
            Command::new("/usr/sbin/sshd")
                .args(["-t", "-f"])
                .arg(config),
            10,
        )?;
        atomic_bytes(unit, b"[Unit]\nDescription=Vortex robot network SSH\nAfter=network.target\n[Service]\nExecStartPre=/usr/bin/install -d -m0755 /run/sshd\nExecStart=/usr/sbin/sshd -D -e -f /etc/ssh/vortex-radio.conf\nRestart=on-failure\nKillMode=process\n[Install]\nWantedBy=multi-user.target\n", 0o644)?;
        checked(Command::new("systemctl").arg("daemon-reload"), 15)?;
        checked(
            Command::new("systemctl").args(["enable", "vortex-radio-ssh.service"]),
            15,
        )?;
        checked(
            Command::new("systemctl").args(["restart", "vortex-radio-ssh.service"]),
            15,
        )?;
        Ok(())
    })();
    if result.is_err() {
        if let Some(bytes) = previous {
            atomic_bytes(config, &bytes, 0o600)?;
        } else {
            let _ = fs::remove_file(config);
        }
        if let Some(bytes) = previous_unit {
            atomic_bytes(unit, &bytes, 0o644)?;
            let _ = Command::new("systemctl").arg("daemon-reload").output();
            let _ = Command::new("systemctl")
                .args(["restart", "vortex-radio-ssh.service"])
                .output();
        } else {
            let _ = Command::new("systemctl")
                .args(["disable", "--now", "vortex-radio-ssh.service"])
                .output();
            let _ = fs::remove_file(unit);
            let _ = Command::new("systemctl").arg("daemon-reload").output();
        }
    }
    result
}
pub fn submit(device: &str, team: u32, number: u32, user: &str, port: u16) -> Result<Value> {
    admin()?;
    interface(device)?;
    let ip = address(team, number)?;
    let report = info()?;
    ensure!(
        report["interfaces"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["interface"] == device && d["eligible"] == true),
        "Select a managed Ethernet device, excluding USB recovery"
    );
    let root = base()?;
    let _lock = Lock::acquire(&root.join("setup.lock"))?;
    for e in fs::read_dir(&root)? {
        let p = e?.path();
        if p.is_dir() && !fs::symlink_metadata(&p)?.file_type().is_symlink() {
            let s = optional(p.join("status.json"), json!({}))?;
            ensure!(
                ![
                    Some("queued"),
                    Some("applying"),
                    Some("awaiting_confirmation")
                ]
                .contains(&s["state"].as_str())
                    || now() - s["started_at"].as_f64().unwrap_or(now()) > 180.,
                "A network change is already pending"
            );
        }
    }
    install_radio_ssh(user, port)?;
    let operation = token();
    let dir = root.join(&operation);
    fs::create_dir(&dir)?;
    mode(&dir, 0o755)?;
    let secret = token();
    let plan = json!({"interface":device,"address":ip,"token":secret,"profile":format!("vortex-radio-{operation}"),"old":field(device,"GENERAL.CON-UUID")?});
    atomic_bytes(&dir.join("plan.json"), &serde_json::to_vec(&plan)?, 0o600)?;
    let initial = json!({"schema_version":1,"operation_id":operation,"state":"queued","interface":device,"address":ip,"started_at":now(),"rollback_seconds":90});
    atomic(dir.join("status.json"), &initial)?;
    copy(&executable()?, &dir.join("vortexctl"), 0o755)?;
    let launched = checked(
        Command::new("systemd-run")
            .args([
                "--collect",
                &format!("--unit=vortex-network-{operation}"),
                "--property=Type=exec",
                "--property=RuntimeMaxSec=150",
                "--property=KillMode=control-group",
            ])
            .arg(dir.join("vortexctl"))
            .args(["network-worker", "--operation-id", &operation]),
        20,
    );
    if let Err(e) = launched {
        state(&dir, "failed", json!({"error":format!("{e:#}")}))?;
        return Err(e);
    }
    let mut response = initial;
    response["token"] = json!(secret);
    response["hostname"] = report["hostname"].clone();
    Ok(response)
}
fn checkpoint(method: &str, path: &str) -> Result<()> {
    checked(
        Command::new("busctl").args([
            "call",
            "org.freedesktop.NetworkManager",
            "/org/freedesktop/NetworkManager",
            "org.freedesktop.NetworkManager",
            method,
            "o",
            path,
        ]),
        15,
    )?;
    Ok(())
}
fn restore(dir: &Path, plan: &Value) -> Result<()> {
    if let Some(cp) = plan["checkpoint"].as_str() {
        let _ = checkpoint("CheckpointRollback", cp);
        let _ = checkpoint("CheckpointDestroy", cp);
    }
    // The checkpoint can expire before this worker wakes; never leave an unconfirmed
    // static profile eligible for autoconnect. Reboot before confirmation uses the old profile.
    let profile = string(plan, "profile")?;
    let _ = nm(&["connection", "delete", "id", profile]);
    let device = string(plan, "interface")?;
    let old = plan["old"].as_str().unwrap_or("");
    if !old.is_empty() && old != "--" {
        // CheckpointRollback schedules reactivation asynchronously. Wait until
        // the old profile is fully connected before reporting recovery.
        nm(&["--wait", "20", "connection", "up", "uuid", old])?;
        ensure!(
            field(device, "GENERAL.CON-UUID")? == old,
            "Previous Ethernet profile could not be restored; use USB recovery"
        );
    } else if old.is_empty() || old == "--" {
        ensure!(
            field(device, "GENERAL.CONNECTION")? != profile,
            "Unconfirmed Ethernet profile is still active; use USB recovery"
        );
    }
    state(
        dir,
        "rolled_back",
        json!({"message":"Network change was not confirmed; previous configuration restored"}),
    )?;
    Ok(())
}
pub fn worker(operation: &str) -> Result<Value> {
    admin()?;
    let dir = directory(operation)?;
    // The submitter holds the same lock until systemd acknowledges this worker.
    // Give its SSH response time to arrive before trying to change connectivity.
    std::thread::sleep(Duration::from_secs(3));
    let deadline = Instant::now() + Duration::from_secs(10);
    let _global = loop {
        match Lock::acquire(&base()?.join("setup.lock")) {
            Ok(lock) => break lock,
            Err(error) if Instant::now() >= deadline => {
                state(&dir, "failed", json!({"error":format!("{error:#}")}))?;
                return Err(error);
            }
            Err(_) => std::thread::sleep(Duration::from_millis(100)),
        }
    };
    let mut plan = read(dir.join("plan.json"))?;
    let result = (|| -> Result<Value> {
        let device = string(&plan, "interface")?.to_string();
        let device_path = field(&device, "GENERAL.DBUS-PATH")?;
        let response = checked(
            Command::new("busctl").args([
                "call",
                "org.freedesktop.NetworkManager",
                "/org/freedesktop/NetworkManager",
                "org.freedesktop.NetworkManager",
                "CheckpointCreate",
                "aouu",
                "1",
                &device_path,
                "90",
                "2",
            ]),
            15,
        )?;
        let output = String::from_utf8(response.stdout)?;
        let cp = output
            .split('"')
            .nth(1)
            .context("Missing NetworkManager checkpoint")?;
        ensure!(
            cp.starts_with("/org/freedesktop/NetworkManager/Checkpoint/"),
            "Invalid NetworkManager checkpoint"
        );
        plan["checkpoint"] = json!(cp);
        atomic_bytes(&dir.join("plan.json"), &serde_json::to_vec(&plan)?, 0o600)?;
        state(&dir, "applying", json!({}))?;
        nm(&[
            "connection",
            "add",
            "type",
            "ethernet",
            "ifname",
            &device,
            "con-name",
            string(&plan, "profile")?,
            "connection.autoconnect",
            "no",
            "connection.autoconnect-priority",
            "999",
            "ipv4.method",
            "manual",
            "ipv4.addresses",
            &format!("{}/24", string(&plan, "address")?),
            "ipv4.never-default",
            "yes",
            "ipv4.dad-timeout",
            "3000",
            "ipv6.never-default",
            "yes",
        ])?;
        nm(&[
            "--wait",
            "20",
            "connection",
            "up",
            "id",
            string(&plan, "profile")?,
        ])?;
        state(&dir, "awaiting_confirmation", json!({}))?;
        let deadline = Instant::now() + Duration::from_secs(75);
        while Instant::now() < deadline {
            if read(dir.join("status.json"))?["state"] == "confirmed" {
                return status(operation);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        let _commit = Lock::acquire(&dir.join("confirm.lock"))?;
        if read(dir.join("status.json"))?["state"] != "confirmed" {
            restore(&dir, &plan)?;
        }
        status(operation)
    })();
    if let Err(ref error) = result {
        let _commit = Lock::acquire(&dir.join("confirm.lock"))?;
        match restore(&dir, &plan) {
            Ok(()) => {
                state(&dir, "rolled_back", json!({"error":format!("{error:#}")}))?;
            }
            Err(recovery) => {
                state(
                    &dir,
                    "rollback_failed",
                    json!({"error":format!("{error:#}; {recovery:#}. Use USB recovery")}),
                )?;
            }
        }
    }
    result
}
pub fn confirm(operation: &str, secret: &str) -> Result<Value> {
    admin()?;
    let dir = directory(operation)?;
    let _lock = Lock::acquire(&dir.join("confirm.lock"))?;
    let plan = read(dir.join("plan.json"))?;
    ensure!(
        plan["token"] == secret
            && read(dir.join("status.json"))?["state"] == "awaiting_confirmation",
        "Network confirmation is invalid or expired"
    );
    let ips = field(string(&plan, "interface")?, "IP4.ADDRESS")?;
    ensure!(
        ips.lines()
            .any(|s| s == format!("{}/24", plan["address"].as_str().unwrap())),
        "Requested address is not active"
    );
    checkpoint("CheckpointDestroy", string(&plan, "checkpoint")?)?;
    if let Err(e) = nm(&[
        "connection",
        "modify",
        "id",
        string(&plan, "profile")?,
        "connection.autoconnect",
        "yes",
    ]) {
        let old = plan["old"].as_str().unwrap_or("");
        if !old.is_empty() && old != "--" {
            let _ = nm(&["connection", "up", "uuid", old]);
        }
        restore(&dir, &plan)?;
        return Err(e);
    }
    state(
        &dir,
        "confirmed",
        json!({"message":"Robot Ethernet address confirmed and saved"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn team_addresses_and_reserved_interfaces() {
        for (team, expected) in [
            (1, "10.0.1.12"),
            (509, "10.5.9.12"),
            (916, "10.9.16.12"),
            (9106, "10.91.6.12"),
            (25599, "10.255.99.12"),
        ] {
            assert_eq!(address(team, 12).unwrap(), expected);
        }
        for (team, device) in [(0, 12), (25600, 12), (509, 5), (509, 20), (509, 200)] {
            assert!(address(team, device).is_err());
        }
        for name in [
            "usb0",
            "usb1",
            "rndis0",
            "l4tbr0",
            "lo",
            "eth0;reboot",
            "../eth0",
            "",
        ] {
            assert!(interface(name).is_err());
        }
        for name in ["eth0", "enp1s0", "enx12345"] {
            interface(name).unwrap();
        }
    }
}
