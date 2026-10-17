//! Small, atomic health snapshots for deployment verification. No camera access here.
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf, time::{Duration, Instant}};

#[derive(Default)]
struct Camera {
    frames: u64,
    last: Option<Instant>,
    backends: Vec<String>,
    pipeline_ok: bool,
    fps: f64,
    processing_ms: f64,
    latency_ms: f64,
}

pub struct Reporter {
    path: PathBuf,
    release: String,
    token: String,
    started: Instant,
    written: Option<Instant>,
    cameras: BTreeMap<usize, Camera>,
    required: Vec<String>,
    watchdog: Option<Watchdog>,
}

struct Watchdog { socket: PathBuf, normal: bool, last: Instant }

#[cfg(unix)]
fn notify(path: &std::path::Path, message: &str) -> std::io::Result<()> {
    use std::os::unix::{ffi::OsStrExt, net::UnixDatagram};
    let socket = UnixDatagram::unbound()?;
    socket.set_nonblocking(true)?;
    #[cfg(target_os = "linux")]
    if let Some(name) = path.as_os_str().as_bytes().strip_prefix(b"@") {
        use std::os::linux::net::SocketAddrExt;
        let address = std::os::unix::net::SocketAddr::from_abstract_name(name)?;
        socket.send_to_addr(message.as_bytes(), &address)?;
        return Ok(())
    }
    socket.send_to(message.as_bytes(), path)?;
    Ok(())
}
#[cfg(not(unix))]
fn notify(_path: &std::path::Path, _message: &str) -> std::io::Result<()> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "systemd notifications require Unix"))
}

#[derive(Serialize)]
struct CameraSnapshot<'a> {
    index: usize,
    frames: u64,
    last_frame_age_ms: Option<u128>,
    pipeline_ok: bool,
    backends: &'a [String],
    fps: f64,
    processing_ms: f64,
    latency_ms: f64,
}

impl Reporter {
    pub fn from_env(indices: &[usize]) -> anyhow::Result<Option<Self>> {
        let Some(path) = std::env::var_os("VORTEX_HEALTH_PATH") else { return Ok(None) };
        let release = std::env::var("VORTEX_RELEASE_ID")?;
        let token = std::env::var("VORTEX_LAUNCH_TOKEN")?;
        anyhow::ensure!(!release.is_empty() && !token.is_empty(), "Missing release health identity");
        let mut reporter = Self::new(PathBuf::from(path), release, token, indices);
        reporter.required = std::env::var("VORTEX_REQUIRED_BACKENDS").unwrap_or_default().split(',')
            .filter(|s| !s.is_empty()).map(str::to_string).collect();
        if std::env::var("VORTEX_SERVICE_WATCHDOG").as_deref() == Ok("1") {
            if let Some(socket) = std::env::var_os("NOTIFY_SOCKET") {
                let socket = PathBuf::from(socket);
                // Opt in per process, preserving rollback to older non-notifying binaries.
                notify(&socket, "WATCHDOG_USEC=40000000\nWATCHDOG=1\nSTATUS=Starting camera pipelines")?;
                reporter.watchdog = Some(Watchdog { socket, normal: false, last: Instant::now() });
            }
        }
        Ok(Some(reporter))
    }

    fn new(path: PathBuf, release: String, token: String, indices: &[usize]) -> Self {
        Self { path, release, token, started: Instant::now(), written: None,
            cameras: indices.iter().map(|i| (*i, Camera::default())).collect(), required: Vec::new(), watchdog: None }
    }

    pub fn record(&mut self, index: usize, timestamp: Instant, backends: Vec<String>, pipeline_ok: bool) {
        if let Some(camera) = self.cameras.get_mut(&index) {
            if let Some(last) = camera.last {
                let elapsed = timestamp.saturating_duration_since(last).as_secs_f64();
                if elapsed > 0.0 {
                    let fps = 1.0/elapsed;
                    camera.fps = if camera.fps == 0.0 { fps } else { 0.2*fps+0.8*camera.fps };
                }
            }
            camera.frames += 1;
            camera.last = Some(timestamp);
            camera.backends = backends;
            camera.pipeline_ok = pipeline_ok;
        }
    }

    pub fn record_metrics(&mut self, index: usize, timestamp: Instant, backends: Vec<String>, pipeline_ok: bool,
                          processing_ms: f64, latency_ms: f64) {
        self.record(index, timestamp, backends, pipeline_ok);
        if let Some(camera) = self.cameras.get_mut(&index) {
            camera.processing_ms = processing_ms;
            camera.latency_ms = latency_ms;
        }
    }

    fn healthy(&self, now: Instant) -> bool {
        !self.cameras.is_empty() && self.cameras.values().all(|camera| camera.pipeline_ok &&
            camera.last.map_or(false, |last| now.saturating_duration_since(last) <= Duration::from_secs(3)) &&
            self.required.iter().all(|backend| camera.backends.contains(backend)))
    }

    fn watchdog_message(&mut self, now: Instant) -> Option<String> {
        let healthy = self.healthy(now);
        let uptime = now.saturating_duration_since(self.started);
        let watchdog = self.watchdog.as_mut()?;
        if now.saturating_duration_since(watchdog.last) < Duration::from_secs(1) { return None }
        watchdog.last = now;
        let mut message = String::new();
        if !watchdog.normal && (healthy || uptime >= Duration::from_secs(30)) {
            watchdog.normal = true;
            message.push_str("WATCHDOG_USEC=10000000\n");
        }
        if healthy || !watchdog.normal { message.push_str("WATCHDOG=1\n"); }
        message.push_str(if healthy { "STATUS=All configured cameras processing" } else if !watchdog.normal {
            "STATUS=Starting camera pipelines"
        } else { "STATUS=Camera processing stalled or required detector unavailable" });
        Some(message)
    }

    fn snapshot(&self, now: Instant) -> serde_json::Value {
        let cameras: Vec<_> = self.cameras.iter().map(|(index, camera)| CameraSnapshot {
            index: *index, frames: camera.frames,
            last_frame_age_ms: camera.last.map(|last| now.saturating_duration_since(last).as_millis()),
            pipeline_ok: camera.pipeline_ok, backends: &camera.backends,
            fps: camera.fps, processing_ms: camera.processing_ms, latency_ms: camera.latency_ms,
        }).collect();
        serde_json::json!({"schema_version": 1, "release_id": self.release, "launch_token": self.token,
            "timestamp_ms": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(),
            "pid": std::process::id(), "uptime_ms": now.saturating_duration_since(self.started).as_millis(),
            "cameras": cameras, "diagnostics_version": 1,
            "watchdog": if self.watchdog.is_none() { "disabled" } else if self.healthy(now) { "healthy" }
                else if self.watchdog.as_ref().map_or(false, |w| !w.normal) { "starting" } else { "stalled" }})
    }

    pub fn flush(&mut self) -> anyhow::Result<()> {
        let now = Instant::now();
        if self.written.map_or(false, |last| now.duration_since(last) < Duration::from_millis(250)) { return Ok(()) }
        self.written = Some(now);
        let parent = self.path.parent().ok_or_else(|| anyhow::anyhow!("Invalid health path"))?;
        fs::create_dir_all(parent)?;
        let temporary = self.path.with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&temporary, serde_json::to_vec(&self.snapshot(now))?)?;
        fs::rename(temporary, &self.path)?;
        let was_normal = self.watchdog.as_ref().map_or(false, |w| w.normal);
        if let Some(message) = self.watchdog_message(now) {
            if let Err(error) = notify(&self.watchdog.as_ref().unwrap().socket, &message) {
                self.watchdog.as_mut().unwrap().normal = was_normal;
                return Err(error.into())
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn supervised(indices: &[usize], now: Instant) -> Reporter {
        let mut reporter = Reporter::new("unused".into(), "release".into(), "nonce".into(), indices);
        reporter.started = now;
        reporter.watchdog = Some(Watchdog { socket: "unused".into(), normal: false, last: now });
        reporter
    }
    #[test]
    fn startup_grace_ends_once_and_stalls_cannot_extend_the_watchdog() {
        let now = Instant::now();
        let mut reporter = supervised(&[0], now);
        assert!(reporter.watchdog_message(now+Duration::from_secs(1)).unwrap().contains("WATCHDOG=1"));
        let ended = reporter.watchdog_message(now+Duration::from_secs(31)).unwrap();
        assert!(ended.contains("WATCHDOG_USEC=10000000"));
        assert!(!ended.contains("WATCHDOG=1"));
        let stalled = reporter.watchdog_message(now+Duration::from_secs(32)).unwrap();
        assert!(!stalled.contains("WATCHDOG_USEC="));
        assert!(!stalled.contains("WATCHDOG=1"));
    }
    #[test]
    fn empty_scenes_keep_all_cameras_healthy_and_one_stall_withholds_heartbeats() {
        let now = Instant::now();
        let mut reporter = supervised(&[0, 2], now);
        for index in [0, 2] { reporter.record(index, now, vec!["apriltag-cpu".into()], true); }
        assert!(reporter.watchdog_message(now+Duration::from_secs(1)).unwrap().contains("WATCHDOG=1"));
        reporter.record(0, now+Duration::from_secs(4), vec!["apriltag-cpu".into()], true);
        assert!(!reporter.watchdog_message(now+Duration::from_secs(4)).unwrap().contains("WATCHDOG=1"));
        reporter.record(2, now+Duration::from_secs(5), vec!["apriltag-cpu".into()], true);
        assert!(reporter.watchdog_message(now+Duration::from_secs(5)).unwrap().contains("WATCHDOG=1"));
    }
    #[test]
    fn wrong_backend_and_detector_failures_are_not_healthy() {
        let now = Instant::now();
        let mut reporter = supervised(&[0], now);
        reporter.required = vec!["apriltag-gpu".into()];
        reporter.record(0, now, vec!["apriltag-cpu".into()], true);
        assert!(!reporter.healthy(now));
        reporter.record(0, now, vec!["apriltag-gpu".into()], false);
        assert!(!reporter.healthy(now));
        reporter.record(0, now, vec!["apriltag-gpu".into()], true);
        assert!(reporter.healthy(now));
    }
    #[test]
    fn watchdog_is_optional_and_notifications_are_rate_limited() {
        let now = Instant::now();
        let mut reporter = supervised(&[0], now);
        assert!(reporter.watchdog_message(now+Duration::from_millis(100)).is_none());
        reporter.watchdog = None;
        assert!(reporter.watchdog_message(now+Duration::from_secs(2)).is_none());
    }
    #[test]
    fn processing_metrics_and_frame_rate_come_from_processed_frames() {
        let now = Instant::now();
        let mut reporter = supervised(&[0], now);
        reporter.record_metrics(0, now, vec!["apriltag-cpu".into()], true, 8.0, 11.0);
        reporter.record_metrics(0, now+Duration::from_millis(20), vec!["apriltag-cpu".into()], true, 9.0, 12.0);
        let snapshot = reporter.snapshot(now+Duration::from_millis(21));
        assert_eq!(snapshot["cameras"][0]["fps"], 50.0);
        assert_eq!(snapshot["cameras"][0]["processing_ms"], 9.0);
        assert_eq!(snapshot["cameras"][0]["latency_ms"], 12.0);
    }
    #[test]
    fn missing_camera_and_empty_detections_are_distinct() {
        let mut reporter = Reporter::new("unused".into(), "release".into(), "nonce".into(), &[0, 2]);
        let now = Instant::now();
        reporter.record(0, now, vec!["apriltag-cpu".into()], true);
        let snapshot = reporter.snapshot(now + Duration::from_millis(500));
        assert_eq!(snapshot["cameras"][0]["frames"], 1);
        assert_eq!(snapshot["cameras"][0]["last_frame_age_ms"], 500);
        assert_eq!(snapshot["cameras"][0]["pipeline_ok"], true);
        assert_eq!(snapshot["cameras"][1]["frames"], 0);
        assert!(snapshot["cameras"][1]["last_frame_age_ms"].is_null());
    }
    #[test]
    fn failed_frame_does_not_claim_success() {
        let mut reporter = Reporter::new("unused".into(), "release".into(), "nonce".into(), &[0]);
        reporter.record(0, Instant::now(), vec!["apriltag-cpu".into()], false);
        assert_eq!(reporter.snapshot(Instant::now())["cameras"][0]["pipeline_ok"], false);
    }
    #[test]
    fn file_snapshots_are_replaced_and_parse_as_complete_json() {
        let suffix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let directory = std::env::temp_dir().join(format!("vortex-health-{}-{suffix}", std::process::id()));
        let path = directory.join("health.json");
        let mut reporter = Reporter::new(path.clone(), "release".into(), "token".into(), &[0]);
        reporter.flush().unwrap();
        reporter.record(0, Instant::now(), vec!["apriltag-cpu".into()], true);
        reporter.written = None;
        reporter.flush().unwrap();
        let value: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(value["cameras"][0]["frames"], 1);
        assert_eq!(value["launch_token"], "token");
        fs::remove_dir_all(directory).unwrap();
    }
}
