#[path = "../config.rs"]
mod config;
#[path = "../detector.rs"]
mod detector;
#[path = "../pose.rs"]
mod pose;
#[path = "../undistort.rs"]
mod undistort;
#[path = "../yolo_detector.rs"]
mod yolo_detector;

use std::fs;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::Serialize;
use turbojpeg::{Compressor, Decompressor, Image, PixelFormat, Subsamp};
use v4l::buffer::Type;
use v4l::capability::Flags;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::{Device, FourCC};

use config::RuntimeConfig;
use detector::{CpuDetector, Detection};

#[derive(Serialize)]
struct TagOut {
    id: usize,
    x: f64,
    y: f64,
    z: f64,
    corners: [[f64; 2]; 4],
    seen_fps: f64,
    seen_pct: f64,
}

#[derive(Serialize)]
struct ObjectOut {
    class_name: String,
    confidence: f64,
    bbox: [f64; 4],
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Serialize)]
struct StateOut {
    timestamp_ms: u128,
    camera_index: usize,
    fps: f64,
    apriltags: Vec<TagOut>,
    objects: Vec<ObjectOut>,
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis()
}

fn set_px(img: &mut [u8], w: usize, h: usize, x: i32, y: i32, v: u8) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    if x >= w || y >= h {
        return;
    }
    img[y * w + x] = v;
}

fn draw_line(img: &mut [u8], w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        set_px(img, w, h, x0, y0, 255);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x0 == x1 {
                break;
            }
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            if y0 == y1 {
                break;
            }
            err += dx;
            y0 += sy;
        }
    }
}

fn draw_rect(img: &mut [u8], w: usize, h: usize, x: i32, y: i32, rw: i32, rh: i32) {
    draw_line(img, w, h, x, y, x + rw, y);
    draw_line(img, w, h, x + rw, y, x + rw, y + rh);
    draw_line(img, w, h, x + rw, y + rh, x, y + rh);
    draw_line(img, w, h, x, y + rh, x, y);
}

fn apply_processing(gray: &mut [u8], cfg: &config::ProcessingConfig) {
    let gain = cfg.sensor_gain.max(0.01);
    let black_offset = cfg.black_level_offset;
    for p in gray.iter_mut() {
        let mut v = (*p as f64) / 255.0;
        v *= gain;
        v += black_offset / 255.0;
        v = v.clamp(0.0, 1.0);
        *p = (v * 255.0).clamp(0.0, 255.0) as u8;
    }
}

fn write_atomic(path: &str, bytes: &[u8]) {
    let tmp = format!("{}.tmp", path);
    if fs::write(&tmp, bytes).is_ok() {
        let _ = fs::rename(&tmp, path);
    }
}

fn main() -> Result<()> {
    let requested_camera = std::env::var("VORTEX_BRIDGE_CAMERA")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let requested_camera = if requested_camera > 5 { 0 } else { requested_camera };
    let frame_path = std::env::var("VORTEX_BRIDGE_FRAME")
        .unwrap_or_else(|_| "/tmp/vortex_bridge_frame.jpg".to_string());
    let state_path = std::env::var("VORTEX_BRIDGE_STATE")
        .unwrap_or_else(|_| "/tmp/vortex_bridge_state.json".to_string());
    let cfg_path = std::env::var("VORTEX_BRIDGE_CONFIG")
        .unwrap_or_else(|_| "config/config.json".to_string());

    let mut runtime_config = RuntimeConfig::load(Path::new(&cfg_path))?;
    let mut cfg_last_modified = fs::metadata(&cfg_path).and_then(|m| m.modified()).ok();
    let mut cfg_last_check = Instant::now();
    let mut cpu = CpuDetector::new(2)?;
    #[cfg(feature = "tensorrt")]
    let mut yolo = yolo_detector::YoloDetector::new().ok();

    let mut selected_camera = None;
    let mut selected_dev = None;
    for step in 0..=5 {
        let idx = (requested_camera + step) % 6;
        let dev = match Device::new(idx) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let caps = match dev.query_caps() {
            Ok(c) => c,
            Err(_) => continue,
        };
        if caps.capabilities.contains(Flags::VIDEO_CAPTURE) {
            selected_camera = Some(idx);
            selected_dev = Some(dev);
            break;
        }
    }
    let camera_index =
        selected_camera.ok_or_else(|| anyhow::anyhow!("no valid video capture camera found in indexes 0..5"))?;
    if camera_index != requested_camera {
        eprintln!(
            "Requested camera {} unavailable; using next valid camera {}",
            requested_camera, camera_index
        );
    }
    let dev = selected_dev.expect("selected camera must have a device");
    let mut fmt = dev.format()?;
    fmt.fourcc = FourCC::new(b"MJPG");
    dev.set_format(&fmt)?;
    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4)?;
    let mut decompressor = Decompressor::new()?;
    let mut compressor = Compressor::new()?;
    compressor.set_subsamp(Subsamp::Gray);

    let mut fps_count = 0u64;
    let mut tag_seen_counts: HashMap<usize, u64> = HashMap::new();
    let mut fps_last = Instant::now();
    let mut fps = 0.0f64;

    loop {
        if cfg_last_check.elapsed() >= Duration::from_millis(500) {
            cfg_last_check = Instant::now();
            let modified = fs::metadata(&cfg_path).and_then(|m| m.modified()).ok();
            let changed = match (cfg_last_modified, modified) {
                (Some(old), Some(new)) => new > old,
                (None, Some(_)) => true,
                _ => false,
            };
            if changed {
                if let Ok(new_cfg) = RuntimeConfig::load(Path::new(&cfg_path)) {
                    runtime_config = new_cfg;
                    cfg_last_modified = modified;
                    eprintln!("Runtime config reloaded: {}", cfg_path);
                }
            }
        }

        let (buf, _) = match stream.next() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let bytes = buf.to_vec();
        let header = match decompressor.read_header(&bytes) {
            Ok(h) => h,
            Err(_) => continue,
        };
        let width = header.width;
        let height = header.height;
        let mut gray = vec![0u8; width * height];
        let img = Image {
            pixels: gray.as_mut_slice(),
            width,
            pitch: width,
            height,
            format: PixelFormat::GRAY,
        };
        if decompressor.decompress(&bytes, img).is_err() {
            continue;
        }
        apply_processing(&mut gray, &runtime_config.processing);

        let mut tags = Vec::new();
        let mut objects = Vec::new();

        let dets = cpu.detect(&gray, width, height).unwrap_or_default();
        for det in dets {
            if let Detection::AprilTag(apr) = det {
                let corners_raw = [
                    (apr.corners[0][0], apr.corners[0][1]),
                    (apr.corners[1][0], apr.corners[1][1]),
                    (apr.corners[2][0], apr.corners[2][1]),
                    (apr.corners[3][0], apr.corners[3][1]),
                ];
                let corners = undistort::undistort_points(&corners_raw, &runtime_config.camera);
                let tag_size = runtime_config.camera.tag_size_m;
                let (x, y, z) = if let Some(p) = pose::estimate_pose(
                    &corners,
                    tag_size,
                    runtime_config.camera.fx,
                    runtime_config.camera.fy,
                    runtime_config.camera.cx,
                    runtime_config.camera.cy,
                ) {
                    (p.translation.x, p.translation.y, p.translation.z)
                } else {
                    (0.0, 0.0, 0.0)
                };
                for i in 0..4 {
                    let a = corners_raw[i];
                    let b = corners_raw[(i + 1) % 4];
                    draw_line(
                        &mut gray,
                        width,
                        height,
                        a.0 as i32,
                        a.1 as i32,
                        b.0 as i32,
                        b.1 as i32,
                    );
                }
                let frame_count_for_metrics = fps_count + 1;
                let c = tag_seen_counts.entry(apr.id).or_insert(0);
                *c += 1;
                let elapsed_for_metrics = fps_last.elapsed().as_secs_f64().max(1e-6);
                let seen_fps = (*c as f64) / elapsed_for_metrics;
                let seen_pct = ((*c as f64) / (frame_count_for_metrics as f64)) * 100.0;
                tags.push(TagOut {
                    id: apr.id,
                    x,
                    y,
                    z,
                    corners: apr.corners,
                    seen_fps,
                    seen_pct,
                });
            }
        }

        #[cfg(feature = "tensorrt")]
        if let Some(y) = &mut yolo {
            let yolo_dets = y.detect(&gray, width, height).unwrap_or_default();
            for det in yolo_dets {
                if let Detection::Yolo(o) = det {
                    let bbox = o.bbox;
                    draw_rect(
                        &mut gray,
                        width,
                        height,
                        bbox[0] as i32,
                        bbox[1] as i32,
                        bbox[2] as i32,
                        bbox[3] as i32,
                    );
                    let u = bbox[0] + bbox[2] / 2.0;
                    let v = bbox[1] + bbox[3] / 2.0;
                    let obj_w_m = runtime_config.object_detection.yolo_obj_width_m;
                    let obj_h_m = runtime_config.object_detection.yolo_obj_height_m;
                    let mut z_candidates = Vec::new();
                    if bbox[2] > 1.0 {
                        z_candidates.push((runtime_config.camera.fx * obj_w_m) / bbox[2]);
                    }
                    if bbox[3] > 1.0 {
                        z_candidates.push((runtime_config.camera.fy * obj_h_m) / bbox[3]);
                    }
                    if z_candidates.is_empty() {
                        continue;
                    }
                    let z = z_candidates.iter().sum::<f64>() / z_candidates.len() as f64;
                    let x = (u - runtime_config.camera.cx) * z / runtime_config.camera.fx;
                    let y = (v - runtime_config.camera.cy) * z / runtime_config.camera.fy;
                    objects.push(ObjectOut {
                        class_name: o.class_name,
                        confidence: o.confidence,
                        bbox,
                        x,
                        y,
                        z,
                    });
                }
            }
        }

        fps_count += 1;
        let elapsed = fps_last.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            fps = fps_count as f64 / elapsed;
            fps_count = 0;
            tag_seen_counts.clear();
            fps_last = Instant::now();
        }

        let state = StateOut {
            timestamp_ms: now_ms(),
            camera_index,
            fps,
            apriltags: tags,
            objects,
        };
        if let Ok(json) = serde_json::to_string(&state) {
            write_atomic(&state_path, json.as_bytes());
        }

        let out_img = Image {
            pixels: gray.as_slice(),
            width,
            pitch: width,
            height,
            format: PixelFormat::GRAY,
        };
        if let Ok(jpg) = compressor.compress_to_owned(out_img) {
            write_atomic(&frame_path, &jpg);
        }

        println!(
            "Camera {}: {:.2} FPS | Last Detections: {}",
            camera_index,
            fps,
            state.apriltags.len() + state.objects.len()
        );
        for t in &state.apriltags {
            println!(
                "  - Tag ID: {} | Dist: {:.2}m | X: {:.2}m | Y: {:.2}m",
                t.id, t.z, t.x, t.y
            );
        }
        for o in &state.objects {
            println!(
                "  - Object: {} ({:.2}) | Dist: {:.2}m | X: {:.2}m | Y: {:.2}m | BBox: [{:.0},{:.0},{:.0},{:.0}]",
                o.class_name, o.confidence, o.z, o.x, o.y, o.bbox[0], o.bbox[1], o.bbox[2], o.bbox[3]
            );
        }
    }
}
