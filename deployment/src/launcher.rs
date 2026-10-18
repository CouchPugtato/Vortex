use crate::common::*;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};
pub const MAIN: &str = "dumapril-taglocalization";
pub const BRIDGE: &str = "orin_bridge";
pub fn features(profile: &str) -> Result<Vec<&'static str>> {
    Ok(match profile {
        "cpu" => vec![],
        "gpu" => vec!["gpu"],
        "tensorrt" => vec!["tensorrt"],
        "gpu-tensorrt" => vec!["gpu", "tensorrt"],
        _ => anyhow::bail!("Unknown build profile"),
    })
}
fn text(v: &Value, empty: bool) -> Result<&str> {
    let s = v.as_str().context("Expected text")?;
    ensure!(
        (empty || !s.trim().is_empty()) && !s.chars().any(char::is_control),
        "Invalid text setting"
    );
    Ok(s)
}
fn integer(v: &Value, lo: u64, hi: u64) -> bool {
    v.as_u64().is_some_and(|n| (lo..=hi).contains(&n))
}
pub fn config(v: &Value) -> Result<()> {
    ensure!(
        v["schema_version"].as_u64() == Some(1) && v["runtime"] == "main",
        "Invalid device schema/runtime"
    );
    features(string(v, "build_profile")?)?;
    let cams = v["cameras"].as_array().context("Missing cameras")?;
    let mut seen = HashSet::new();
    ensure!(
        !cams.is_empty()
            && cams
                .iter()
                .all(|n| integer(n, 0, 63) && seen.insert(n.as_u64().unwrap())),
        "Camera indices must be unique integers from 0 to 63"
    );
    text(&v["runtime_config"], false)?;
    text(&v["tag_map"], false)?;
    let nt = &v["networktables"];
    let udp = &v["udp"];
    ensure!(
        nt["enabled"].is_boolean() && udp["enabled"].is_boolean(),
        "Invalid network enable flag"
    );
    ensure!(
        ["team", "custom", "local"].contains(&string(nt, "mode")?)
            && integer(&nt["team"], 1, 99999),
        "Invalid NetworkTables settings"
    );
    text(
        &nt["server"],
        nt["enabled"] != true || nt["mode"] != "custom",
    )?;
    ensure!(
        text(&nt["table"], false)?.starts_with('/'),
        "NT table must start with /"
    );
    text(&udp["target"], udp["enabled"] != true)?;
    ensure!(integer(&udp["port"], 1, 65535), "Invalid UDP port");
    Ok(())
}
pub fn resources(root: &Path, c: &Value) -> Result<Value> {
    config(c)?;
    let r = read(root.join(string(c, "runtime_config")?))?;
    ensure!(r["camera"].is_object(), "Missing camera object");
    let mut cameras = vec![&r["camera"]];
    if let Some(profiles) = r.get("camera_profiles") {
        cameras.extend(
            profiles
                .as_object()
                .context("Invalid camera profiles")?
                .values(),
        );
    }
    for cam in cameras {
        ensure!(cam.is_object(), "Invalid camera profile");
        for k in [
            "fx",
            "fy",
            "cx",
            "cy",
            "k1",
            "k2",
            "k3",
            "p1",
            "p2",
            "tag_size_m",
        ] {
            ensure!(
                cam[k].as_f64().is_some_and(f64::is_finite),
                "Invalid camera.{k}"
            );
        }
        for k in ["fx", "fy", "tag_size_m"] {
            ensure!(cam[k].as_f64().unwrap() > 0., "Camera {k} must be positive");
        }
        for k in [
            "x_offset",
            "y_offset",
            "z_offset",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        ] {
            ensure!(
                cam.get(k)
                    .is_none_or(|n| n.as_f64().is_some_and(f64::is_finite)),
                "Invalid camera {k}"
            );
        }
    }
    for section in ["processing", "object_detection"] {
        if let Some(v) = r.get(section) {
            for (k, n) in v.as_object().context("Invalid processing section")? {
                if k == "use_nn" && section == "object_detection" {
                    ensure!(n.is_boolean(), "use_nn must be boolean");
                } else if [
                    "smoothing_alpha",
                    "resolution_scale_factor",
                    "gpu_scale_factor",
                    "black_level_offset",
                    "sensor_gain",
                    "red_balance",
                    "blue_balance",
                    "yolo_obj_width_m",
                    "yolo_obj_height_m",
                    "processing_yolo_obj_width_m",
                    "processing_yolo_obj_height_m",
                    "confidence_threshold",
                ]
                .contains(&k.as_str())
                {
                    ensure!(
                        n.as_f64().is_some_and(f64::is_finite),
                        "Invalid {section}.{k}"
                    );
                }
            }
        }
    }
    let field = read(root.join(string(c, "tag_map")?))?;
    let tags = field["tags"].as_array().context("Missing tag map")?;
    ensure!(!tags.is_empty(), "Empty tag map");
    let mut ids = HashSet::new();
    for tag in tags {
        ensure!(
            integer(&tag["ID"], 1, i32::MAX as u64) && ids.insert(tag["ID"].as_u64().unwrap()),
            "Invalid/duplicate tag ID"
        );
        for k in ["x", "y", "z"] {
            ensure!(
                tag["pose"]["translation"][k]
                    .as_f64()
                    .is_some_and(f64::is_finite),
                "Invalid tag translation"
            );
        }
        let mut norm = 0.;
        for k in ["W", "X", "Y", "Z"] {
            let n = tag["pose"]["rotation"]["quaternion"][k]
                .as_f64()
                .context("Invalid tag quaternion")?;
            ensure!(n.is_finite(), "Invalid quaternion");
            norm += n * n;
        }
        ensure!(norm > 0., "Zero quaternion");
    }
    Ok(r)
}
pub fn source_digest(root: &Path) -> Result<String> {
    let mut names: Vec<String> = vec![
        "Cargo.toml".into(),
        "Cargo.lock".into(),
        "build.rs".into(),
        "rust-toolchain.toml".into(),
    ];
    let mut sources = Vec::new();
    for e in walkdir::WalkDir::new(root.join("src")) {
        let e = e?;
        ensure!(
            !e.file_type().is_symlink(),
            "Source symlinks are not supported"
        );
        if e.file_type().is_file()
            && ["rs", "cpp", "h", "c"]
                .contains(&e.path().extension().and_then(|s| s.to_str()).unwrap_or(""))
        {
            sources.push(
                e.path()
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    sources.sort();
    names.extend(sources);
    let mut h = Sha256::new();
    for name in names {
        let digest = Sha256::digest(fs::read(root.join(&name))?);
        h.update(name.as_bytes());
        h.update([0]);
        h.update(digest);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn pin(root: &Path) -> Result<String> {
    let s = fs::read_to_string(root.join("rust-toolchain.toml"))?;
    for line in s.lines() {
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == "channel" {
                let v = v.trim().trim_matches('"');
                let parts: Vec<_> = v.split('.').collect();
                ensure!(
                    parts.len() == 3
                        && parts
                            .iter()
                            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())),
                    "Pin an exact Rust release"
                );
                return Ok(v.into());
            }
        }
    }
    anyhow::bail!("Missing exact Rust toolchain pin")
}
pub fn binary(root: &Path, name: &str) -> PathBuf {
    root.join("target/release")
        .join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}
pub fn checked_binary(root: &Path, c: &Value, name: &str) -> Result<PathBuf> {
    let p = binary(root, name);
    regular(&p)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            fs::metadata(&p)?.permissions().mode() & 0o111 != 0,
            "Runtime is not executable"
        );
    }
    let s = read(root.join("target/release/vortex-build.json"))?;
    ensure!(
        s["schema_version"] == 1
            && s["runtime"] == "main"
            && s["profile"] == c["build_profile"]
            && s["source_sha256"] == source_digest(root)?
            && s["binaries"][name] == hash(&p)?,
        "Stale executable or incorrect profile; run vortexctl launch --build"
    );
    Ok(p)
}
pub fn build(
    root: &Path,
    c: &Value,
    online: bool,
    debug: bool,
    cache: Option<&Path>,
    log: Option<&std::fs::File>,
) -> Result<()> {
    let cargo = home()?
        .join(".cargo/bin")
        .join(format!("cargo{}", std::env::consts::EXE_SUFFIX));
    let cargo = if cargo.is_file() {
        cargo
    } else {
        PathBuf::from("cargo")
    };
    let mut cmd = Command::new(cargo);
    cmd.arg(format!("+{}", pin(root)?)).args([
        "build",
        "--release",
        "--locked",
        "--no-default-features",
        "--bin",
        MAIN,
    ]);
    if !online {
        cmd.arg("--offline");
    }
    let mut bins = vec![MAIN];
    if debug {
        cmd.args(["--bin", BRIDGE]);
        bins.push(BRIDGE);
    }
    let f = features(string(c, "build_profile")?)?;
    if !f.is_empty() {
        cmd.args(["--features", &f.join(",")]);
    }
    if let Some(cache) = cache {
        cmd.arg("--target-dir").arg(cache);
    }
    for key in ["CARGO_TARGET_DIR", "CARGO_BUILD_TARGET", "RUSTUP_TOOLCHAIN"] {
        cmd.env_remove(key);
    }
    cmd.env("RUSTUP_AUTO_INSTALL", if online { "1" } else { "0" })
        .env("CARGO_BUILD_JOBS", "2")
        .current_dir(root);
    let before = source_digest(root)?;
    let stamp = root.join("target/release/vortex-build.json");
    if stamp.exists() {
        fs::remove_file(&stamp)?;
    }
    if let Some(log) = log {
        logged(&mut cmd, log, 7200)?;
    } else {
        let status = cmd.status()?;
        ensure!(status.success(), "Cargo build failed");
    }
    ensure!(
        source_digest(root)? == before,
        "Source changed during build"
    );
    let mut hashes = BTreeMap::new();
    for name in bins {
        let dest = binary(root, name);
        if let Some(cache) = cache {
            let src = cache.join("release").join(dest.file_name().unwrap());
            if src != dest {
                copy(&src, &dest, 0o755)?;
            }
        }
        hashes.insert(name, hash(dest)?);
    }
    atomic(
        stamp,
        &json!({"schema_version":1,"runtime":"main","profile":c["build_profile"],"source_sha256":before,"binaries":hashes}),
    )
}
pub fn required(c: &Value, r: &Value) -> Result<Vec<&'static str>> {
    let f = features(string(c, "build_profile")?)?;
    let mut out = vec![if f.contains(&"gpu") {
        "apriltag-gpu"
    } else {
        "apriltag-cpu"
    }];
    if f.contains(&"tensorrt") && r["object_detection"]["use_nn"] != false {
        out.push("yolo-tensorrt");
    }
    Ok(out)
}
pub fn command(root: &Path, c: &Value, debug: Option<u64>) -> Result<Command> {
    let mut cmd = Command::new(checked_binary(
        root,
        c,
        if debug.is_some() { BRIDGE } else { MAIN },
    )?);
    cmd.current_dir(root);
    for k in [
        "VORTEX_RUNTIME_CONFIG",
        "VORTEX_TAG_MAP",
        "VORTEX_NT_ENABLE",
        "VORTEX_NT_TEAM",
        "VORTEX_NT_SERVER",
        "VORTEX_NT_TABLE",
        "VORTEX_UDP_ENABLE",
        "VORTEX_UDP_TARGET",
        "VORTEX_UDP_PORT",
        "VORTEX_BRIDGE_CONFIG",
        "VORTEX_BRIDGE_CAMERA",
        "VORTEX_BRIDGE_FRAME",
        "VORTEX_BRIDGE_STATE",
    ] {
        cmd.env_remove(k);
    }
    let nt = &c["networktables"];
    let udp = &c["udp"];
    cmd.env(
        "VORTEX_RUNTIME_CONFIG",
        root.join(string(c, "runtime_config")?),
    )
    .env("VORTEX_TAG_MAP", root.join(string(c, "tag_map")?))
    .env(
        "VORTEX_MAIN_GPU",
        if features(string(c, "build_profile")?)?.contains(&"gpu") {
            "1"
        } else {
            "0"
        },
    );
    for (key, value) in [
        (
            "VORTEX_NT_ENABLE",
            if nt["enabled"] == true {
                "1".into()
            } else {
                "0".into()
            },
        ),
        ("VORTEX_NT_TABLE", string(nt, "table")?.into()),
        (
            "VORTEX_UDP_ENABLE",
            if udp["enabled"] == true {
                "1".into()
            } else {
                "0".into()
            },
        ),
        ("VORTEX_UDP_TARGET", string(udp, "target")?.into()),
        ("VORTEX_UDP_PORT", udp["port"].to_string()),
    ] {
        cmd.env(key, value);
    }
    if nt["enabled"] == true {
        if nt["mode"] == "team" {
            cmd.env("VORTEX_NT_TEAM", nt["team"].to_string());
        } else {
            cmd.env(
                "VORTEX_NT_SERVER",
                if nt["mode"] == "local" {
                    "127.0.0.1"
                } else {
                    string(nt, "server")?
                },
            );
        }
    }
    if let Some(camera) = debug {
        ensure!(camera <= 5, "Debug camera must be 0-5");
        cmd.env("VORTEX_BRIDGE_CAMERA", camera.to_string())
            .env(
                "VORTEX_BRIDGE_CONFIG",
                root.join(string(c, "runtime_config")?),
            )
            .env("VORTEX_BRIDGE_FRAME", "/tmp/vortex_bridge_frame.jpg")
            .env("VORTEX_BRIDGE_STATE", "/tmp/vortex_bridge_state.json");
    } else {
        cmd.arg(
            c["cameras"]
                .as_array()
                .unwrap()
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    Ok(cmd)
}
pub fn exec(mut cmd: Command) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        Err(cmd.exec().into())
    }
    #[cfg(not(unix))]
    {
        ensure!(cmd.status()?.success(), "Runtime failed");
        Ok(())
    }
}
