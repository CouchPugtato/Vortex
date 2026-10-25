//! Test-only native runtime for exercising real systemd supervision without cameras.
use serde_json::{json, Value};
use std::{
    env, fs,
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
fn write(path: &Path, value: &Value) {
    let temp = path.with_extension("tmp");
    fs::write(&temp, value.to_string()).unwrap();
    fs::rename(temp, path).unwrap();
}
#[cfg(target_os = "linux")]
fn notify(message: &str) {
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixDatagram};
    let Some(path) = env::var_os("NOTIFY_SOCKET") else {
        return;
    };
    let path = path.to_string_lossy();
    let socket = UnixDatagram::unbound().unwrap();
    if let Some(name) = path.strip_prefix('@') {
        socket
            .send_to_addr(
                message.as_bytes(),
                &SocketAddr::from_abstract_name(name).unwrap(),
            )
            .unwrap();
    } else {
        socket.send_to(message.as_bytes(), path.as_ref()).unwrap();
    }
}
#[cfg(not(target_os = "linux"))]
fn notify(_: &str) {}
fn main() {
    if env::args().any(|a| a == "--deployment-smoke") {
        println!(
            "VORTEX_SMOKE {}",
            json!({"passed":true,"tag_id":1,"camera_error_m":0.01})
        );
        return;
    }
    let config: Value =
        serde_json::from_slice(&fs::read(env::var("VORTEX_RUNTIME_CONFIG").unwrap()).unwrap())
            .unwrap();
    if config["deployment_fixture"] == "fail" {
        std::process::exit(2)
    }
    let started = Instant::now();
    let mut frames = 0;
    notify("WATCHDOG_USEC=3000000\nWATCHDOG=1");
    loop {
        if !(Path::new(".fixture-stall").exists() && started.elapsed() > Duration::from_secs(4)) {
            frames += 1;
            notify("WATCHDOG=1");
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();
            let report = json!({"schema_version":1,"pid":std::process::id(),"boot_id":fs::read_to_string("/proc/sys/kernel/random/boot_id").unwrap().trim(),"release_id":env::var("VORTEX_RELEASE_ID").unwrap(),"launch_token":env::var("VORTEX_LAUNCH_TOKEN").unwrap(),"uptime_ms":started.elapsed().as_millis(),"timestamp_ms":now,"diagnostics_version":1,"watchdog":"healthy","cameras":[{"index":0,"frames":frames,"last_frame_age_ms":0,"pipeline_ok":true,"backends":["apriltag-cpu"],"fps":10}]});
            write(Path::new(&env::var("VORTEX_HEALTH_PATH").unwrap()), &report);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}
