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
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
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
    floor_z_error: f64,
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
    field: Option<FieldOut>,
    robot_pose: Option<RobotPoseOut>,
    apriltags: Vec<TagOut>,
    objects: Vec<ObjectOut>,
}

#[derive(Serialize)]
struct FieldOut {
    length: f64,
    width: f64,
}

#[derive(Serialize)]
struct RobotPoseOut {
    x: f64,
    y: f64,
    tags_used: usize,
    floor_z_error_avg: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct TagMap {
    tags: Vec<TagMapTag>,
    field: Option<TagMapField>,
}

#[derive(Clone, Debug, Deserialize)]
struct TagMapField {
    length: f64,
    width: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct TagMapTag {
    #[serde(rename = "ID", alias = "id")]
    id: usize,
    pose: TagMapPose,
}

#[derive(Clone, Debug, Deserialize)]
struct TagMapPose {
    translation: TagMapTranslation,
    rotation: TagMapRotation,
}

#[derive(Clone, Debug, Deserialize)]
struct TagMapTranslation {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct TagMapRotation {
    quaternion: TagMapQuaternion,
}

#[derive(Clone, Debug, Deserialize)]
struct TagMapQuaternion {
    #[serde(rename = "W", alias = "w")]
    w: f64,
    #[serde(rename = "X", alias = "x")]
    x: f64,
    #[serde(rename = "Y", alias = "y")]
    y: f64,
    #[serde(rename = "Z", alias = "z")]
    z: f64,
}

#[derive(Clone, Debug)]
struct FieldTagPose {
    pos: Vector3<f64>,
    rot_field_from_tag: Matrix3<f64>,
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

fn load_tag_map(path: &str) -> Option<(HashMap<usize, FieldTagPose>, Option<FieldOut>)> {
    let raw = fs::read_to_string(path).ok()?;
    let parsed: TagMap = serde_json::from_str(&raw).ok()?;
    let mut by_id = HashMap::new();
    for tag in parsed.tags {
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            tag.pose.rotation.quaternion.w,
            tag.pose.rotation.quaternion.x,
            tag.pose.rotation.quaternion.y,
            tag.pose.rotation.quaternion.z,
        ));
        by_id.insert(
            tag.id,
            FieldTagPose {
                pos: Vector3::new(
                    tag.pose.translation.x,
                    tag.pose.translation.y,
                    tag.pose.translation.z,
                ),
                rot_field_from_tag: q.to_rotation_matrix().into_inner().transpose(),
            },
        );
    }
    let field = parsed.field.map(|f| FieldOut {
        length: f.length,
        width: f.width,
    });
    Some((by_id, field))
}

fn camera_to_robot_rotation(camera_cfg: &config::CameraConfig) -> Matrix3<f64> {
    let r_yaw = Rotation3::from_axis_angle(&Vector3::y_axis(), camera_cfg.yaw_deg.to_radians());
    let r_pitch = Rotation3::from_axis_angle(&Vector3::x_axis(), camera_cfg.pitch_deg.to_radians());
    let r_roll = Rotation3::from_axis_angle(&Vector3::z_axis(), camera_cfg.roll_deg.to_radians());
    (r_yaw * r_pitch * r_roll).into_inner()
}

fn estimate_robot_field_from_tag(
    pose: &pose::Pose,
    tag_field: &FieldTagPose,
    camera_cfg: &config::CameraConfig,
) -> (Vector3<f64>, f64) {
    // pose gives tag frame -> camera frame.
    let r_ct = pose.rotation;
    let r_tc = r_ct.transpose();
    let p_tc = -r_tc * pose.translation;
    let p_fc = tag_field.pos + tag_field.rot_field_from_tag * p_tc;
    let r_fc = tag_field.rot_field_from_tag * r_tc;

    let r_rc = camera_to_robot_rotation(camera_cfg);
    let r_cr = r_rc.transpose();
    let cam_offset_robot = Vector3::new(camera_cfg.x_offset, camera_cfg.y_offset, camera_cfg.z_offset);
    let robot_origin_in_camera = -r_cr * cam_offset_robot;
    let p_fr = p_fc + r_fc * robot_origin_in_camera;
    let floor_z_error = p_fr.z.abs();
    (p_fr, floor_z_error)
}

fn robust_fuse_field_pose(candidates: &[(f64, f64, f64, f64)]) -> Option<(f64, f64, usize, f64)> {
    // (x, y, floor_z_error, cam_depth)
    const MAX_FLOOR_ERR_M: f64 = 3.00;
    const HUBER_DELTA_M: f64 = 0.35;

    let finite: Vec<(f64, f64, f64, f64)> = candidates
        .iter()
        .copied()
        .filter(|(x, y, z_err, d)| x.is_finite() && y.is_finite() && z_err.is_finite() && d.is_finite())
        .collect();
    if finite.is_empty() {
        return None;
    }

    let mut kept: Vec<(f64, f64, f64, f64)> = finite
        .iter()
        .copied()
        .filter(|(_, _, z_err, _)| *z_err <= MAX_FLOOR_ERR_M)
        .collect();
    // fail safe if gate rejects all, keep finite candidates instead of dropping pose entirely
    if kept.is_empty() {
        kept = finite;
    }

    let mut weights = Vec::with_capacity(kept.len());
    for (_x, _y, z_err, depth) in &kept {
        let z_w = 1.0 / (1.0 + 4.0 * z_err.max(0.0));
        let d_w = 1.0 / (0.3 + depth.abs().max(0.05));
        weights.push((z_w * d_w).max(1e-6));
    }

    let weighted_mean = |ws: &[f64]| -> (f64, f64) {
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sw = 0.0;
        for (i, (x, y, _z, _d)) in kept.iter().enumerate() {
            let w = ws[i];
            sx += x * w;
            sy += y * w;
            sw += w;
        }
        if sw <= 1e-9 {
            (0.0, 0.0)
        } else {
            (sx / sw, sy / sw)
        }
    };

    let (mut cx, mut cy) = weighted_mean(&weights);
    for _ in 0..2 {
        let mut huber_weights = Vec::with_capacity(kept.len());
        for (i, (x, y, _z, _d)) in kept.iter().enumerate() {
            let r = ((*x - cx).hypot(*y - cy)).max(1e-9);
            let huber = if r <= HUBER_DELTA_M { 1.0 } else { HUBER_DELTA_M / r };
            huber_weights.push(weights[i] * huber);
        }
        let (nx, ny) = weighted_mean(&huber_weights);
        cx = nx;
        cy = ny;
    }

    let avg_z = kept.iter().map(|v| v.2).sum::<f64>() / kept.len() as f64;
    Some((cx, cy, kept.len(), avg_z))
}

fn write_atomic(path: &str, bytes: &[u8]) {
    let tmp = format!("{}.tmp", path);
    if fs::write(&tmp, bytes).is_ok() {
        let _ = fs::rename(&tmp, path);
    }
}

fn resolve_tag_map_path() -> String {
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("VORTEX_TAG_MAP") {
        if !p.trim().is_empty() {
            candidates.push(p);
        }
    }
    candidates.push("config/apriltag_map.json".to_string());
    candidates.push("../config/apriltag_map.json".to_string());
    candidates.push("/home/vortex/deployments/Vortex/config/apriltag_map.json".to_string());
    candidates.push("/home/jetson/deployments/Vortex/config/apriltag_map.json".to_string());

    for p in candidates {
        if fs::metadata(&p).map(|m| m.is_file()).unwrap_or(false) {
            return p;
        }
    }
    "config/apriltag_map.json".to_string()
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
    let tag_map_path = resolve_tag_map_path();

    let mut runtime_config = RuntimeConfig::load(Path::new(&cfg_path))?;
    let mut cfg_last_modified = fs::metadata(&cfg_path).and_then(|m| m.modified()).ok();
    let mut tag_map_loaded = load_tag_map(&tag_map_path);
    let mut tag_map_last_modified = fs::metadata(&tag_map_path).and_then(|m| m.modified()).ok();
    eprintln!("Tag map path: {}", tag_map_path);
    let mut cfg_last_check = Instant::now();
    let mut cpu = CpuDetector::new(2)?;
    #[cfg(feature = "tensorrt")]
    let mut yolo = if runtime_config.object_detection.use_nn {
        match yolo_detector::YoloDetector::new() {
            Ok(det) => {
                eprintln!("orin_bridge: TensorRT YOLO enabled");
                Some(det)
            }
            Err(err) => {
                eprintln!("orin_bridge: TensorRT YOLO unavailable: {}", err);
                None
            }
        }
    } else {
        eprintln!("orin_bridge: TensorRT YOLO disabled by config");
        None
    };
    #[cfg(not(feature = "tensorrt"))]
    eprintln!("orin_bridge: built without tensorrt feature; object detection disabled");

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
    let mut camera_cfg = runtime_config.camera_for_index(camera_index);
    eprintln!("Using camera profile for id {}", camera_index);
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
    let mut filtered_robot_xy: Option<(f64, f64)> = None;

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
                    #[cfg(feature = "tensorrt")]
                    {
                        let prev_use_nn = runtime_config.object_detection.use_nn;
                        let next_use_nn = new_cfg.object_detection.use_nn;
                        if prev_use_nn != next_use_nn {
                            if next_use_nn {
                                match yolo_detector::YoloDetector::new() {
                                    Ok(det) => {
                                        yolo = Some(det);
                                        eprintln!("orin_bridge: TensorRT YOLO enabled by config reload");
                                    }
                                    Err(err) => {
                                        yolo = None;
                                        eprintln!("orin_bridge: TensorRT YOLO enable failed: {}", err);
                                    }
                                }
                            } else {
                                yolo = None;
                                eprintln!("orin_bridge: TensorRT YOLO disabled by config reload");
                            }
                        }
                    }
                    runtime_config = new_cfg;
                    camera_cfg = runtime_config.camera_for_index(camera_index);
                    cfg_last_modified = modified;
                    eprintln!("Runtime config reloaded: {}", cfg_path);
                    eprintln!("Using camera profile for id {}", camera_index);
                }
            }
            let map_modified = fs::metadata(&tag_map_path).and_then(|m| m.modified()).ok();
            let map_changed = match (tag_map_last_modified, map_modified) {
                (Some(old), Some(new)) => new > old,
                (None, Some(_)) => true,
                _ => false,
            };
            if map_changed {
                tag_map_loaded = load_tag_map(&tag_map_path);
                tag_map_last_modified = map_modified;
                eprintln!("Tag map reloaded: {}", tag_map_path);
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
        let mut field_candidates: Vec<(f64, f64, f64, f64)> = Vec::new();

        let dets = cpu.detect(&gray, width, height).unwrap_or_default();
        for det in dets {
            if let Detection::AprilTag(apr) = det {
                // apriltag corners are not in the solver's expected TL,BL,BR,TR order.
                let c = apr.corners;
                let corners_raw = [
                    (c[3][0], c[3][1]), // TL
                    (c[0][0], c[0][1]), // BL
                    (c[1][0], c[1][1]), // BR
                    (c[2][0], c[2][1]), // TR
                ];
                let (x, y, z, floor_z_error) = if let Some(p) = pose::estimate_pose(
                    &corners_raw,
                    &camera_cfg,
                    true,
                ) {
                    if let Some((map_by_id, _field_meta)) = &tag_map_loaded {
                        if let Some(tag_field) = map_by_id.get(&apr.id) {
                            let (robot_field, z_err) =
                                estimate_robot_field_from_tag(&p, tag_field, &camera_cfg);
                            field_candidates.push((robot_field.x, robot_field.y, z_err, p.translation.z.abs()));
                            (robot_field.x, robot_field.y, 0.0, z_err)
                        } else {
                            (p.translation.x, p.translation.y, p.translation.z, 0.0)
                        }
                    } else {
                        (p.translation.x, p.translation.y, p.translation.z, 0.0)
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0)
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
                    floor_z_error,
                    corners: apr.corners,
                    seen_fps,
                    seen_pct,
                });
            }
        }

        #[cfg(feature = "tensorrt")]
        if runtime_config.object_detection.use_nn {
            if let Some(y) = &mut yolo {
            let yolo_dets = y.detect(&gray, width, height).unwrap_or_default();
            let conf_threshold = runtime_config
                .object_detection
                .confidence_threshold
                .clamp(0.0, 1.0);
            for det in yolo_dets {
                if let Detection::Yolo(o) = det {
                    if o.confidence < conf_threshold {
                        continue;
                    }
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
                        z_candidates.push((camera_cfg.fx * obj_w_m) / bbox[2]);
                    }
                    if bbox[3] > 1.0 {
                        z_candidates.push((camera_cfg.fy * obj_h_m) / bbox[3]);
                    }
                    if z_candidates.is_empty() {
                        continue;
                    }
                    let z = z_candidates.iter().sum::<f64>() / z_candidates.len() as f64;
                    let x = (u - camera_cfg.cx) * z / camera_cfg.fx;
                    let y = (v - camera_cfg.cy) * z / camera_cfg.fy;
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
        }

        fps_count += 1;
        let elapsed = fps_last.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            fps = fps_count as f64 / elapsed;
            fps_count = 0;
            tag_seen_counts.clear();
            fps_last = Instant::now();
        }

        let robot_pose = if let Some((raw_x, raw_y, used, avg_z_err)) =
            robust_fuse_field_pose(&field_candidates)
        {
            // temporal smoothing with jump clamp to reduce jitter in UI/telemetry
            const MAX_STEP_M: f64 = 0.60;
            const SMOOTH_ALPHA: f64 = 0.18;
            let (sx, sy) = if let Some((px, py)) = filtered_robot_xy {
                let mut tx = raw_x;
                let mut ty = raw_y;
                let dx = tx - px;
                let dy = ty - py;
                let dist = dx.hypot(dy);
                if dist > MAX_STEP_M {
                    let scale = MAX_STEP_M / dist;
                    tx = px + dx * scale;
                    ty = py + dy * scale;
                }
                (
                    px + SMOOTH_ALPHA * (tx - px),
                    py + SMOOTH_ALPHA * (ty - py),
                )
            } else {
                (raw_x, raw_y)
            };
            filtered_robot_xy = Some((sx, sy));
            Some(RobotPoseOut {
                x: sx,
                y: sy,
                tags_used: used,
                floor_z_error_avg: avg_z_err,
            })
        } else {
            filtered_robot_xy.map(|(sx, sy)| RobotPoseOut {
                x: sx,
                y: sy,
                tags_used: 0,
                floor_z_error_avg: 0.0,
            })
        };
        let field_meta = tag_map_loaded
            .as_ref()
            .and_then(|(_, f)| f.as_ref().map(|x| FieldOut { length: x.length, width: x.width }));
        let state = StateOut {
            timestamp_ms: now_ms(),
            camera_index,
            fps,
            field: field_meta,
            robot_pose,
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
                "  - Tag ID: {} | Field X: {:.2}m | Field Y: {:.2}m | FloorErr: {:.3}m",
                t.id, t.x, t.y, t.floor_z_error
            );
        }
        if let Some(rp) = &state.robot_pose {
            println!(
                "  - Robot Pose Avg | X: {:.2}m | Y: {:.2}m | Tags: {} | FloorErr: {:.3}m",
                rp.x, rp.y, rp.tags_used, rp.floor_z_error_avg
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
