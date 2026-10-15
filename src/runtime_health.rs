//! Small, atomic health snapshots for deployment verification. No camera access here.
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf, time::{Duration, Instant}};

#[derive(Default)]
struct Camera {
    frames: u64,
    last: Option<Instant>,
    backends: Vec<String>,
    pipeline_ok: bool,
}

pub struct Reporter {
    path: PathBuf,
    release: String,
    token: String,
    started: Instant,
    written: Option<Instant>,
    cameras: BTreeMap<usize, Camera>,
}

#[derive(Serialize)]
struct CameraSnapshot<'a> {
    index: usize,
    frames: u64,
    last_frame_age_ms: Option<u128>,
    pipeline_ok: bool,
    backends: &'a [String],
}

impl Reporter {
    pub fn from_env(indices: &[usize]) -> anyhow::Result<Option<Self>> {
        let Some(path) = std::env::var_os("VORTEX_HEALTH_PATH") else { return Ok(None) };
        let release = std::env::var("VORTEX_RELEASE_ID")?;
        let token = std::env::var("VORTEX_LAUNCH_TOKEN")?;
        anyhow::ensure!(!release.is_empty() && !token.is_empty(), "Missing release health identity");
        Ok(Some(Self::new(PathBuf::from(path), release, token, indices)))
    }

    fn new(path: PathBuf, release: String, token: String, indices: &[usize]) -> Self {
        Self { path, release, token, started: Instant::now(), written: None,
            cameras: indices.iter().map(|i| (*i, Camera::default())).collect() }
    }

    pub fn record(&mut self, index: usize, timestamp: Instant, backends: Vec<String>, pipeline_ok: bool) {
        if let Some(camera) = self.cameras.get_mut(&index) {
            camera.frames += 1;
            camera.last = Some(timestamp);
            camera.backends = backends;
            camera.pipeline_ok = pipeline_ok;
        }
    }

    fn snapshot(&self, now: Instant) -> serde_json::Value {
        let cameras: Vec<_> = self.cameras.iter().map(|(index, camera)| CameraSnapshot {
            index: *index, frames: camera.frames,
            last_frame_age_ms: camera.last.map(|last| now.saturating_duration_since(last).as_millis()),
            pipeline_ok: camera.pipeline_ok, backends: &camera.backends,
        }).collect();
        serde_json::json!({"schema_version": 1, "release_id": self.release, "launch_token": self.token,
            "pid": std::process::id(), "uptime_ms": now.saturating_duration_since(self.started).as_millis(),
            "cameras": cameras})
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
