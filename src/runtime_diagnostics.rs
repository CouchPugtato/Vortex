//! On-demand previews of the production detector's own pixels and results.
use serde_json::{json, Value};
use std::{fs, path::PathBuf, sync::{mpsc, Arc, Mutex}, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};

pub fn now_ms() -> u128 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() }

#[derive(Clone)]
struct Lease { camera: usize, client: String, expires: u128, radio: bool }
struct Control { lease: Option<Lease>, offered: Option<Instant> }
struct Frame { pixels: Vec<u8>, width: usize, height: usize, captured: Instant, metadata: Value, lease: Lease }

#[derive(Clone)]
pub struct Hub { tx: mpsc::SyncSender<Frame>, control: Arc<Mutex<Control>> }

impl Hub {
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        let Some(directory) = std::env::var_os("VORTEX_DIAGNOSTICS_DIR") else { return Ok(None) };
        let release = std::env::var("VORTEX_RELEASE_ID")?;
        let token = std::env::var("VORTEX_LAUNCH_TOKEN")?;
        Ok(Some(Self::start(PathBuf::from(directory), release, token)?))
    }

    fn start(directory: PathBuf, release: String, token: String) -> anyhow::Result<Self> {
        fs::create_dir_all(&directory)?;
        // One managed main process owns this directory. Remove only our old previews.
        for item in fs::read_dir(&directory)? {
            let path = item?.path();
            if path.file_name().and_then(|n| n.to_str()).map_or(false, |n| n.starts_with("frame-") && n.ends_with(".jpg")) {
                let _ = fs::remove_file(path);
            }
        }
        let (tx, rx) = mpsc::sync_channel::<Frame>(1);
        let control = Arc::new(Mutex::new(Control { lease: None, offered: None }));
        let shared = control.clone();
        std::thread::spawn(move || {
            let mut sequence = 0u64;
            let mut retained = std::collections::VecDeque::new();
            loop {
                let lease = read_lease(&directory, &release, &token, now_ms());
                if let Ok(mut state) = shared.lock() { state.lease = lease; }
                match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(frame) => {
                        if frame.lease.expires <= now_ms() { continue }
                        sequence += 1;
                        if let Err(error) = write_frame(&directory, &release, &token, sequence, frame, &mut retained) {
                            eprintln!("Diagnostics preview failed: {error}");
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {},
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });
        Ok(Self { tx, control })
    }

    pub fn offer<F: FnOnce() -> Value>(&self, camera: usize, pixels: &[u8], width: usize, height: usize,
                                       captured: Instant, metadata: F) {
        if width == 0 || height == 0 || width.checked_mul(height) != Some(pixels.len()) || pixels.len() > 32*1024*1024 { return }
        let Ok(mut control) = self.control.try_lock() else { return };
        let Some(lease) = control.lease.clone() else { return };
        let now = Instant::now();
        if lease.camera != camera || lease.expires <= now_ms() ||
            control.offered.map_or(false, |last| now.saturating_duration_since(last) < Duration::from_millis(if lease.radio {1000} else {200})) { return }
        control.offered = Some(now);
        drop(control);
        // At most one frame waits for the worker. Preview never waits on disk/JPEG I/O.
        let _ = self.tx.try_send(Frame { pixels: pixels.to_vec(), width, height, captured, metadata: metadata(), lease });
    }
}

fn read_lease(directory: &std::path::Path, release: &str, token: &str, now: u128) -> Option<Lease> {
    let path = directory.join("request.json");
    if fs::metadata(&path).ok()?.len() > 4096 { return None }
    let value: Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    let expires = value["expires_ms"].as_u64()? as u128;
    let client = value["client_id"].as_str()?;
    if value["release_id"] != release || value["launch_token"] != token ||
        value["pid"] != std::process::id() || expires <= now || expires > now+5000 ||
        client.is_empty() || client.len() > 96 || !client.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') { return None }
    Some(Lease { camera: usize::try_from(value["camera_index"].as_u64()?).ok()?, client: client.into(), expires, radio: value["radio"].as_bool().unwrap_or(false) })
}

fn write_frame(directory: &std::path::Path, release: &str, token: &str, sequence: u64, mut frame: Frame,
               retained: &mut std::collections::VecDeque<PathBuf>) -> anyhow::Result<()> {
    let gray = image::GrayImage::from_raw(frame.width as u32, frame.height as u32, frame.pixels)
        .ok_or_else(|| anyhow::anyhow!("Invalid preview pixels"))?;
    let (max_width, max_height) = if frame.lease.radio { (320.0, 240.0) } else { (640.0, 480.0) };
    let scale = (max_width/frame.width as f64).min(max_height/frame.height as f64).min(1.0);
    let width = ((frame.width as f64*scale).round() as u32).max(1);
    let height = ((frame.height as f64*scale).round() as u32).max(1);
    let resized = image::imageops::resize(&gray, width, height, image::imageops::FilterType::Nearest);
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, if frame.lease.radio {50} else {65}).encode(resized.as_raw(), width, height, image::ColorType::L8)?;
    let name = format!("frame-{}-{sequence}.jpg", std::process::id());
    let path = directory.join(&name);
    frame.metadata["schema_version"] = json!(1);
    frame.metadata["release_id"] = json!(release);
    frame.metadata["launch_token"] = json!(token);
    frame.metadata["pid"] = json!(std::process::id());
    frame.metadata["client_id"] = json!(frame.lease.client);
    frame.metadata["camera_index"] = json!(frame.lease.camera);
    frame.metadata["frame_id"] = json!(name);
    frame.metadata["source_width"] = json!(frame.width);
    frame.metadata["source_height"] = json!(frame.height);
    frame.metadata["preview_width"] = json!(width);
    frame.metadata["preview_height"] = json!(height);
    frame.metadata["timestamp_ms"] = json!(now_ms());
    frame.metadata["latency_ms"] = json!(frame.captured.elapsed().as_secs_f64()*1000.0);
    let temporary = directory.join("preview.tmp");
    let published: anyhow::Result<()> = (|| {
        fs::write(&path, jpeg)?;
        fs::write(&temporary, serde_json::to_vec(&frame.metadata)?)?;
        fs::rename(&temporary, directory.join("preview.json"))?;
        Ok(())
    })();
    if let Err(error) = published {
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&temporary);
        return Err(error)
    }
    retained.push_back(path);
    while retained.len() > 3 { if let Some(old) = retained.pop_front() { let _ = fs::remove_file(old); } }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;
    #[test]
    fn radio_lease_limits_work_and_image_size() {
        let directory = std::env::temp_dir().join(format!("vortex-radio-preview-{}-{}",std::process::id(),now_ms()));
        fs::create_dir_all(&directory).unwrap();
        let lease = Lease { camera: 0, client: "radio".into(), expires: now_ms()+3000, radio: true };
        let (tx,rx) = mpsc::sync_channel(1);
        let hub = Hub { tx, control: Arc::new(Mutex::new(Control {lease:Some(lease.clone()),offered:Some(Instant::now()-Duration::from_millis(500))})) };
        hub.offer(0,&[1],1,1,Instant::now(),||panic!("radio throttle must skip metadata and frame copying"));
        assert!(rx.try_recv().is_err());
        let frame = Frame {pixels:vec![70;1280*720],width:1280,height:720,captured:Instant::now(),metadata:json!({}),lease};
        let mut retained = std::collections::VecDeque::new();
        write_frame(&directory,"r","t",1,frame,&mut retained).unwrap();
        let state:Value=serde_json::from_slice(&fs::read(directory.join("preview.json")).unwrap()).unwrap();
        let image=image::open(directory.join(state["frame_id"].as_str().unwrap())).unwrap();
        assert_eq!(image.dimensions(),(320,180));
        assert_eq!(state["source_width"],1280);
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn no_subscription_wrong_camera_expired_lease_and_throttle_do_not_copy_frames() {
        let (tx, rx) = mpsc::sync_channel(1);
        let control = Arc::new(Mutex::new(Control { lease: None, offered: None }));
        let hub = Hub { tx, control: control.clone() };
        let calls = std::cell::Cell::new(0);
        let metadata = || { calls.set(calls.get()+1); json!({}) };
        hub.offer(0, &[1], 1, 1, Instant::now(), metadata);
        control.lock().unwrap().lease = Some(Lease { camera: 2, client: "c".into(), expires: now_ms()+2000, radio: false });
        hub.offer(0, &[1], 1, 1, Instant::now(), metadata);
        hub.offer(2, &[1], 1, 1, Instant::now(), metadata);
        hub.offer(2, &[1], 1, 1, Instant::now(), metadata);
        assert_eq!(calls.get(), 1);
        assert!(rx.try_recv().is_ok());
        assert!(rx.try_recv().is_err());
        control.lock().unwrap().lease.as_mut().unwrap().expires = 0;
        hub.offer(2, &[1], 1, 1, Instant::now(), metadata);
        assert_eq!(calls.get(), 1);
    }
    #[test]
    fn images_and_results_share_a_frame_and_storage_is_bounded() {
        let directory = std::env::temp_dir().join(format!("vortex-preview-{}-{}", std::process::id(), now_ms()));
        fs::create_dir_all(&directory).unwrap();
        let mut retained = std::collections::VecDeque::new();
        for sequence in 1..=5 {
            let frame = Frame { pixels: vec![70; 1280*720], width: 1280, height: 720, captured: Instant::now(),
                metadata: json!({"apriltags": [{"id": sequence}], "objects": []}),
                lease: Lease { camera: 2, client: "client".into(), expires: now_ms()+3000, radio: false } };
            write_frame(&directory, "release", "token", sequence, frame, &mut retained).unwrap();
        }
        let state: Value = serde_json::from_slice(&fs::read(directory.join("preview.json")).unwrap()).unwrap();
        assert_eq!(state["apriltags"][0]["id"], 5);
        let image = image::open(directory.join(state["frame_id"].as_str().unwrap())).unwrap();
        assert_eq!((image.width(), image.height()), (640, 360));
        assert_eq!(retained.len(), 3);
        assert_eq!(fs::read_dir(&directory).unwrap().filter(|p| p.as_ref().unwrap().path().extension().and_then(|s| s.to_str()) == Some("jpg")).count(), 3);
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn lease_requires_live_current_process_and_bounded_expiry() {
        let directory = std::env::temp_dir().join(format!("vortex-lease-{}-{}", std::process::id(), now_ms()));
        fs::create_dir_all(&directory).unwrap();
        let base = json!({"release_id": "r", "launch_token": "t", "pid": std::process::id(), "camera_index": 0,
                          "client_id": "client", "expires_ms": 4000});
        for (field, value) in [("launch_token", json!("old")), ("pid", json!(0)), ("expires_ms", json!(1000)), ("expires_ms", json!(99999))] {
            let mut bad = base.clone(); bad[field] = value;
            fs::write(directory.join("request.json"), bad.to_string()).unwrap();
            assert!(read_lease(&directory, "r", "t", 2000).is_none());
        }
        fs::write(directory.join("request.json"), base.to_string()).unwrap();
        assert!(read_lease(&directory, "r", "t", 2000).is_some());
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn failed_metadata_publication_does_not_accumulate_orphan_images() {
        let directory = std::env::temp_dir().join(format!("vortex-preview-failure-{}-{}", std::process::id(), now_ms()));
        fs::create_dir_all(directory.join("preview.json")).unwrap();
        let mut retained = std::collections::VecDeque::new();
        for sequence in 1..=4 {
            let frame = Frame { pixels: vec![1; 16], width: 4, height: 4, captured: Instant::now(), metadata: json!({}),
                lease: Lease { camera: 0, client: "c".into(), expires: now_ms()+2000, radio: false } };
            assert!(write_frame(&directory, "r", "t", sequence, frame, &mut retained).is_err());
        }
        assert!(retained.is_empty());
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }
}
