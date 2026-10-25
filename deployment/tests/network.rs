//! Real NetworkManager tests run only inside the disposable two-NIC rehearsal VM.
#![cfg(target_os = "linux")]
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::{
    fs,
    io::Read,
    net::TcpStream,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};
use vortexctl::common::*;
fn controller() -> PathBuf {
    std::env::var_os("VORTEX_TEST_CONTROLLER")
        .map(PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_vortexctl").into())
}
fn call(args: &[&str]) -> Result<Value> {
    Ok(serde_json::from_slice(
        &checked(Command::new(controller()).args(args), 30)?.stdout,
    )?)
}
fn nm(args: &[&str]) -> Result<String> {
    Ok(
        String::from_utf8(checked(Command::new("nmcli").args(args), 30)?.stdout)?
            .trim()
            .into(),
    )
}
fn wait(operation: &str, wanted: &str, seconds: u64) -> Result<Value> {
    let end = Instant::now() + Duration::from_secs(seconds);
    loop {
        let value = call(&["network-status", "--operation-id", operation])?;
        if value["state"] == wanted {
            return Ok(value);
        }
        ensure!(
            value["state"] != "failed"
                && (wanted == "rolled_back" || value["state"] != "rolled_back"),
            "Network change failed: {value}"
        );
        ensure!(Instant::now() < end, "Network test timed out: {value}");
        std::thread::sleep(Duration::from_millis(250));
    }
}
#[test]
#[ignore = "Requires a disposable rehearsal guest with a second Ethernet NIC"]
fn real_network_confirmation_and_rollback() -> Result<()> {
    admin()?;
    ensure!(
        Path::new("/etc/vortex-rehearsal-guest").is_file(),
        "Only run inside the disposable rehearsal guest"
    );
    let device = fs::read_dir("/sys/class/net")?
        .filter_map(|entry| entry.ok())
        .find(|entry| {
            fs::read_to_string(entry.path().join("address"))
                .is_ok_and(|s| s.trim() == "52:54:00:12:34:56")
        })
        .context("Missing isolated rehearsal Ethernet NIC")?
        .file_name()
        .to_string_lossy()
        .to_string();
    // Cloud-init initially brings up both NICs with networkd. Remove the empty
    // radio definition after NetworkManager is installed; otherwise networkd's
    // IPv6 link-local state can be assumed as an external profile at the next boot.
    // Production setup accepts an already managed NIC and never takes ownership
    // from a competing network manager.
    fs::write("/etc/netplan/50-cloud-init.yaml", "network:\n  version: 2\n  ethernets:\n    management:\n      match: {macaddress: '52:54:00:65:43:21'}\n      set-name: eth0\n      dhcp4: true\n")?;
    fs::write(
        "/etc/cloud/cloud.cfg.d/99-vortex-rehearsal-network.cfg",
        "network: {config: disabled}\n",
    )?;
    fs::create_dir_all("/etc/systemd/network")?;
    fs::write(
        "/etc/systemd/network/10-vortex-rehearsal-radio.link",
        include_str!("fixtures/radio.link"),
    )?;
    checked(Command::new("netplan").arg("generate"), 20)?;
    fs::create_dir_all("/etc/NetworkManager/conf.d")?;
    fs::write(
        "/etc/NetworkManager/conf.d/90-vortex-rehearsal.conf",
        format!("[keyfile]\nunmanaged-devices=*,except:interface-name:{device}\n"),
    )?;
    checked(
        Command::new("systemctl").args(["restart", "NetworkManager"]),
        20,
    )?;
    nm(&["device", "set", &device, "managed", "yes"])?;
    nm(&[
        "connection",
        "add",
        "type",
        "ethernet",
        "ifname",
        &device,
        "con-name",
        "rehearsal-original",
        "ipv4.method",
        "manual",
        "ipv4.addresses",
        "10.5.9.13/24",
        "ipv4.never-default",
        "yes",
        "ipv6.method",
        "disabled",
    ])?;
    nm(&["connection", "up", "id", "rehearsal-original"])?;
    let op = call(&[
        "network-submit",
        "--interface",
        &device,
        "--team",
        "509",
        "--device",
        "12",
        "--user",
        "rehearsal",
    ])?;
    let identity = string(&op, "operation_id")?;
    wait(identity, "awaiting_confirmation", 30)?;
    ensure!(
        !output(
            Command::new(controller()).args([
                "network-confirm",
                "--operation-id",
                identity,
                "--token",
                "wrong"
            ]),
            10
        )?
        .status
        .success(),
        "Bad confirmation token was accepted"
    );
    ensure!(
        !output(
            Command::new(controller()).args([
                "network-submit",
                "--interface",
                &device,
                "--team",
                "509",
                "--device",
                "14",
                "--user",
                "rehearsal"
            ]),
            10
        )?
        .status
        .success(),
        "Concurrent network mutation was accepted"
    );
    let confirmed = call(&[
        "network-confirm",
        "--operation-id",
        identity,
        "--token",
        string(&op, "token")?,
    ])?;
    ensure!(
        confirmed["state"] == "confirmed",
        "Confirmation did not persist"
    );
    let ips = nm(&["--get-values", "IP4.ADDRESS", "device", "show", &device])?;
    ensure!(
        ips.contains("10.5.9.12/24"),
        "Wrong committed address: {ips}"
    );
    for port in [22, 5808] {
        let mut stream = TcpStream::connect(("127.0.0.1", port))?;
        stream.set_read_timeout(Some(Duration::from_secs(3)))?;
        let mut bytes = [0; 256];
        let count = stream.read(&mut bytes)?;
        ensure!(
            String::from_utf8_lossy(&bytes[..count]).starts_with("SSH-2.0-"),
            "SSH listener {port} is unavailable"
        );
    }
    // Let the confirmed worker release its global mutation lock.
    std::thread::sleep(Duration::from_secs(1));
    let unconfirmed = call(&[
        "network-submit",
        "--interface",
        &device,
        "--team",
        "509",
        "--device",
        "14",
        "--user",
        "rehearsal",
    ])?;
    let identity = string(&unconfirmed, "operation_id")?;
    wait(identity, "awaiting_confirmation", 30)?;
    wait(identity, "rolled_back", 110)?;
    let ips = nm(&["--get-values", "IP4.ADDRESS", "device", "show", &device])?;
    ensure!(
        ips.contains("10.5.9.12/24") && !ips.contains("10.5.9.14/24"),
        "Unconfirmed address did not roll back: {ips}"
    );
    ensure!(
        !output(
            Command::new(controller()).args([
                "network-submit",
                "--interface",
                "usb0",
                "--user",
                "rehearsal"
            ]),
            10
        )?
        .status
        .success(),
        "USB recovery interface was accepted"
    );
    println!("NETWORK PASS: real NetworkManager checkpoint, confirmed address, rejected bad token/concurrent change/USB mutation, both SSH listeners, automatic rollback after client disappears");
    fs::write(
        "/var/lib/vortex-network/rehearsal-boot",
        fs::read("/proc/sys/kernel/random/boot_id")?,
    )?;
    Ok(())
}

#[test]
#[ignore = "Requires a rehearsal guest reboot after the network transaction test"]
fn real_network_survives_reboot() -> Result<()> {
    admin()?;
    ensure!(
        Path::new("/etc/vortex-rehearsal-guest").is_file(),
        "Disposable guest required"
    );
    ensure!(
        fs::read("/var/lib/vortex-network/rehearsal-boot")?
            != fs::read("/proc/sys/kernel/random/boot_id")?,
        "Guest has not rebooted"
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let report = vortexctl::network::info()?;
        let address = report["interfaces"].as_array().unwrap().iter().any(|d| {
            d["addresses"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a == "10.5.9.12/24")
        });
        if address && TcpStream::connect(("127.0.0.1", 5808)).is_ok() {
            break;
        }
        ensure!(
            Instant::now() < deadline,
            "Confirmed robot address or SSH listener did not survive reboot"
        );
        std::thread::sleep(Duration::from_millis(250));
    }
    println!("NETWORK REBOOT PASS: confirmed robot Ethernet address and separate SSH listener survived a real guest reboot");
    Ok(())
}
