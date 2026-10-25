use crate::{
    activation::{self, Manager},
    common::*,
    launcher, service,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    time::{Duration, Instant},
};
pub fn validate(root: &Path, seconds: u64) -> Result<Value> {
    ensure!(
        (5..=600).contains(&seconds),
        "Validation duration must be 5-600 seconds"
    );
    let manager = activation::Systemd::new(root)?;
    let selection = activation::selected(root)?;
    ensure!(
        selection["kind"] == "release",
        "Activate a managed release first"
    );
    let identity = string(&selection, "release_id")?;
    let smoke = activation::preflight(root, identity)?;
    let (dir, _) = activation::ready(root, identity)?;
    let device = read(dir.join("config/device.json"))?;
    let runtime = launcher::resources(&dir, &device)?;
    let baseline = read(root.join("health.json"))?;
    let info = manager.info()?;
    activation::health(root, &selection, &info, &baseline)?;
    let cameras = device["cameras"]
        .as_array()
        .context("Camera settings missing")?;
    let client = format!("validation-{}", token());
    let result = (|| -> Result<Value> {
        let started = Instant::now();
        let mut previews = BTreeMap::<u64, (u64, f64)>::new();
        let mut last = baseline.clone();
        let mut samples = 0;
        let mut maximum_age = 0f64;
        while started.elapsed() < Duration::from_secs(seconds) {
            ensure!(
                activation::selected(root)? == selection,
                "Release changed during validation"
            );
            let current = manager.info()?;
            ensure!(
                current["pid"] == info["pid"] && current["restarts"] == info["restarts"],
                "Service restarted during validation"
            );
            let report = read(root.join("health.json"))?;
            let timestamp = report["timestamp_ms"]
                .as_f64()
                .context("Missing runtime timestamp")?;
            ensure!(
                (0.0..=1500.0).contains(&(now() * 1000. - timestamp)),
                "Runtime health publication is stale"
            );
            activation::health(root, &selection, &current, &report)?;
            ensure!(
                report["uptime_ms"].as_f64() >= last["uptime_ms"].as_f64(),
                "Runtime clock moved backwards"
            );
            for c in report["cameras"].as_array().unwrap() {
                maximum_age = maximum_age.max(c["last_frame_age_ms"].as_f64().unwrap());
            }
            let camera = cameras[(started.elapsed().as_secs() as usize / 2) % cameras.len()]
                .as_u64()
                .unwrap();
            let preview = service::diagnostics(root, camera, &client, false, &manager)?;
            ensure!(
                preview["reason"] != "Another diagnostics client has the preview lease",
                "Detach the other diagnostics client before validation"
            );
            if preview["ready"] == true {
                let state = &preview["state"];
                let entry = previews.entry(camera).or_default();
                entry.0 += 1;
                entry.1 = entry.1.max(state["frame_age_ms"].as_f64().unwrap_or(1500.));
            }
            samples += 1;
            last = report;
            std::thread::sleep(Duration::from_millis(250));
        }
        for c in last["cameras"].as_array().unwrap() {
            let old = baseline["cameras"]
                .as_array()
                .unwrap()
                .iter()
                .find(|b| b["index"] == c["index"])
                .context("Camera set changed")?;
            ensure!(
                c["frames"].as_u64() > old["frames"].as_u64(),
                "A configured camera did not advance"
            );
        }
        ensure!(cameras.iter().all(|c|previews.get(&c.as_u64().unwrap()).is_some_and(|v|v.0>0)),"Not every configured camera produced a fresh preview; allow at least two seconds per camera");
        let hardware = fs::read("/proc/device-tree/model")
            .ok()
            .map(|b| String::from_utf8_lossy(&b).trim_matches('\0').to_string());
        Ok(
            json!({"schema_version":1,"passed":true,"verification":"device_health_and_previews","release_id":identity,"pid":info["pid"],"duration_seconds":seconds,"samples":samples,"architecture":std::env::consts::ARCH,"hardware":hardware,"required_backends":launcher::required(&device,&runtime)?,"maximum_camera_frame_age_ms":maximum_age,"preview_samples_and_max_age_ms":previews,"smoke":smoke,"boot_id":fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim()}),
        )
    })();
    let _ = service::diagnostics(root, 0, &client, true, &manager);
    result
}
