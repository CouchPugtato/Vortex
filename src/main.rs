mod detector;
mod gpu_detector;
mod vpi;
mod config;
mod undistort;
mod preprocess;
mod pose;
mod yolo_detector;

use nalgebra as _;
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::mpsc;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;

use v4l::{Device, FourCC};
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::video::Capture;
use v4l::io::traits::CaptureStream;
use v4l::capability::Flags;
use turbojpeg::{Decompressor, Compressor, Image, PixelFormat, Subsamp};
use serde::Deserialize;

use crate::config::{CameraConfig, ObjectDetectionConfig, ProcessingConfig, RuntimeConfig};
use crate::detector::Detection;




#[derive(Debug, Clone)]
pub struct AprilTagPose {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub floor_z_error: f64,
}

#[derive(Debug, Clone)]
pub struct YoloBBox {
    pub class_name: String,
    pub confidence: f64,
    pub bbox: [f64; 4],
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone)]
enum ProcessedDetection {
    AprilTag(AprilTagPose),
    Yolo(YoloBBox),
}

#[derive(Debug)]
struct ProcessedDetections(Vec<ProcessedDetection>);

#[derive(Debug)]
struct PipelineStats {
    camera_index: usize,
    detections: ProcessedDetections,
    timestamp: Instant,
}

#[derive(Clone, Debug, Deserialize)]
struct TagMap {
    tags: Vec<TagMapTag>,
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    
    // parse camera indices from args
    // examples: "0", "0,1", "0 2", "0, 2"
    // create output directory for debug images
    if let Err(e) = std::fs::create_dir_all("output") {
        eprintln!("Failed to create output directory: {}", e);
    }

    let mut camera_indices: Vec<usize> = Vec::new();
    let mut output_dir_base = PathBuf::from("output");

    // skip program name
    for arg in args.iter().skip(1) {
        if arg.chars().all(|c| c.is_ascii_digit() || c == ',') {
            for part in arg.split(',') {
                if let Ok(idx) = part.trim().parse::<usize>() {
                    if !camera_indices.contains(&idx) {
                        camera_indices.push(idx);
                    }
                }
            }
        } else {
            // assume it's an output dir if not a number list
            output_dir_base = PathBuf::from(arg);
        }
    }

    if camera_indices.is_empty() { camera_indices.push(0); }

    println!("Starting Multi-Camera AprilTag Detector");
    println!("Cameras: {:?}", camera_indices);
    println!("Output Dir: {:?}", output_dir_base);

    fs::create_dir_all(&output_dir_base)?;

    let config_path = Path::new("config/config.json");
    let runtime_config = match RuntimeConfig::load(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load config from {:?}: {}", config_path, e);
            return Err(anyhow::anyhow!("Config load failed"));
        }
    };
    let tag_map_path = resolve_tag_map_path();
    let tag_map_by_id = tag_map_path
        .as_ref()
        .map(|p| load_tag_map(p.as_path()))
        .unwrap_or_default();
    if let Some(p) = &tag_map_path {
        println!("Loaded {} field tags from {}", tag_map_by_id.len(), p.display());
    } else {
        println!("No apriltag_map.json found in known locations; running without field map.");
    }

    // dynamic thread allocation
    // Jetson Orin Nano has 6 CPU cores.
    // for 4 cameras, we want:
    // - 1 Main Thread (Monitoring)
    // - 4 Capture Threads (Low CPU, mostly IO)
    // - 4 Decode Threads (TurboJPEG)
    // - 4 Detector Threads (VPI - CPU Backend)
    
    // if cameras <= 2: Allow multiple detector threads per camera to max out performance
    // if cameras > 2: Limit to 1 detector thread per camera to prevent thrashing
    let total_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(6);
    // reserve 2 cores for system overhead + capture/decode threads
    let reserved_cores = 2; 
    let threads_per_cam = if camera_indices.len() <= 2 {
        let available = total_cores as i32 - reserved_cores;
        std::cmp::max(1, available / camera_indices.len() as i32)
    } else {
        1
    };
    
    println!("Allocating {} detector threads per camera (Total Cameras: {}, Total Cores: {})", 
        threads_per_cam, camera_indices.len(), total_cores);

    let (tx_stats, rx_stats) = mpsc::channel();

    for &idx in &camera_indices {
        spawn_camera_pipeline(
            idx,
            threads_per_cam,
            tx_stats.clone(),
            runtime_config.camera_for_index(idx),
            runtime_config.processing,
            runtime_config.object_detection,
            tag_map_by_id.clone(),
        );
    }

    // monitor loop
    let mut cam_stats: HashMap<usize, (u64, Instant)> = HashMap::new(); // (frame_count, last_report)
    let mut cam_fps: HashMap<usize, f64> = HashMap::new();
    let mut cam_detections: HashMap<usize, Vec<ProcessedDetection>> = HashMap::new();
    let mut filtered_robot_xy: HashMap<usize, (f64, f64)> = HashMap::new();
    
    // simple exponential smoothing for pose
    // store last known pose for each tag id per camera
    let mut pose_filters: HashMap<(usize, usize), (f64, f64, f64)> = HashMap::new(); // (cam_idx, tag_id) -> (x, y, z)
    let alpha = runtime_config.processing.smoothing_alpha;

    // init stats
    let start_time = Instant::now();
    for &idx in &camera_indices {
        cam_stats.insert(idx, (0, start_time));
        cam_fps.insert(idx, 0.0);
        cam_detections.insert(idx, Vec::new());
    }

    loop {
        if let Ok(stat) = rx_stats.recv() {
            let mut smoothed_detections = Vec::new();

            for det in stat.detections.0 {
                match det {
                    ProcessedDetection::AprilTag(apr) => {
                        let key = (stat.camera_index, apr.id);
                        let (s_x, s_y, s_z) = if let Some(&(last_x, last_y, last_z)) = pose_filters.get(&key) {
                            (
                                last_x + alpha * (apr.x - last_x),
                                last_y + alpha * (apr.y - last_y),
                                last_z + alpha * (apr.z - last_z),
                            )
                        } else {
                            (apr.x, apr.y, apr.z)
                        };
                        
                        pose_filters.insert(key, (s_x, s_y, s_z));
                        
                        smoothed_detections.push(ProcessedDetection::AprilTag(AprilTagPose {
                            id: apr.id,
                            x: s_x,
                            y: s_y,
                            z: s_z,
                            floor_z_error: apr.floor_z_error,
                        }));
                    }
                    ProcessedDetection::Yolo(yolo) => {
                        // no smoothing for YOLO detections
                        smoothed_detections.push(ProcessedDetection::Yolo(yolo));
                    }
                }
            }

            cam_detections.insert(stat.camera_index, smoothed_detections);

            if let Some((count, last_time)) = cam_stats.get_mut(&stat.camera_index) {
                *count += 1;
                
                let now = Instant::now();
                let duration = now.duration_since(*last_time);
                
                if duration.as_secs() >= 1 {
                    let fps = *count as f64 / duration.as_secs_f64();
                    cam_fps.insert(stat.camera_index, fps);
                    *count = 0;
                    *last_time = now;
                    
                    // print summary
                    print!("\x1B[2J\x1B[1;1H"); // clear screen
                    println!("=== Multi-Camera Status ===");
                    for &idx in &camera_indices {
                        let detections = cam_detections.get(&idx).unwrap();
                        let tag_count = detections
                            .iter()
                            .filter(|d| matches!(d, ProcessedDetection::AprilTag(_)))
                            .count();
                        let yolo_count = detections
                            .iter()
                            .filter(|d| matches!(d, ProcessedDetection::Yolo(_)))
                            .count();
                        println!("Camera {}: {:.2} FPS | Last Detections: {}", 
                            idx, 
                            cam_fps.get(&idx).unwrap_or(&0.0),
                            detections.len()
                        );
                        println!("  - Counts: AprilTag={} YOLO={}", tag_count, yolo_count);
                        let mut field_candidates: Vec<(f64, f64, f64, f64)> = Vec::new();
                        for det in detections {
                            match det {
                                ProcessedDetection::AprilTag(a) => {
                                    field_candidates.push((a.x, a.y, a.floor_z_error, a.z.abs()));
                                    println!(
                                        "  - Tag ID: {} | Field X: {:.2}m | Field Y: {:.2}m | FloorErr: {:.3}m",
                                        a.id, a.x, a.y, a.floor_z_error
                                    );
                                }
                                ProcessedDetection::Yolo(y) => {
                                    println!(
                                        "  - Object: {} ({:.2}) | Dist: {:.2}m | X: {:.2}m | Y: {:.2}m | BBox: [{:.0},{:.0},{:.0},{:.0}]",
                                        y.class_name,
                                        y.confidence,
                                        y.z,
                                        y.x,
                                        y.y,
                                        y.bbox[0],
                                        y.bbox[1],
                                        y.bbox[2],
                                        y.bbox[3]
                                    );
                                }
                            }
                        }
                        if let Some((raw_x, raw_y, used, avg_z_err)) = robust_fuse_field_pose(&field_candidates) {
                            const MAX_STEP_M: f64 = 0.60;
                            const SMOOTH_ALPHA: f64 = 0.18;
                            let prev = filtered_robot_xy.get(&idx).copied();
                            let (sx, sy) = if let Some((px, py)) = prev {
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
                            filtered_robot_xy.insert(idx, (sx, sy));
                            println!(
                                "  - Robot Pose Avg | Field X: {:.2}m | Field Y: {:.2}m | Tags: {} | FloorErr: {:.3}m",
                                sx, sy, used, avg_z_err
                            );
                        }
                    }
                }
            }
        }
    }
}

fn spawn_camera_pipeline(
    camera_index: usize, 
    detector_threads: i32, 
    tx_stats: mpsc::Sender<PipelineStats>,
    camera_config: CameraConfig,
    processing_config: ProcessingConfig,
    object_detection_config: ObjectDetectionConfig,
    tag_map_by_id: HashMap<usize, FieldTagPose>,
) {
    println!("Spawning pipeline for Camera {}...", camera_index);

    let (tx_capture, rx_capture) = mpsc::sync_channel(1); 
    let (tx_decode, rx_decode) = mpsc::sync_channel(1);

    // 1. capture thread
    std::thread::spawn(move || {
        loop {
            let dev = match Device::new(camera_index) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Error opening camera {}: {}. Retrying in 2s...", camera_index, e);
                    std::thread::sleep(Duration::from_secs(2));
                    continue;
                }
            };

            let caps = match dev.query_caps() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error querying capabilities for camera {}: {}", camera_index, e);
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };

            if !caps.capabilities.contains(Flags::VIDEO_CAPTURE) {
                eprintln!("Camera {} ({}) is not a video capture device (missing VIDEO_CAPTURE capability). It might be a metadata node.", camera_index, caps.card);
                eprintln!("Try using indices 0, 2, 4, 6 instead of 0, 1, 2, 3.");
                std::thread::sleep(Duration::from_secs(5));
                continue;
            }

            println!("Camera {} initialized: {} ({})", camera_index, caps.card, caps.bus);

            let mut fmt = match dev.format() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Error getting format for camera {}: {}", camera_index, e);
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };
            
            fmt.fourcc = FourCC::new(b"MJPG");
            
            if let Err(e) = dev.set_format(&fmt) {
                eprintln!("Error setting format for camera {}: {}", camera_index, e);
                std::thread::sleep(Duration::from_secs(1));
                continue;
            }
            
            let mut stream = match MmapStream::with_buffers(&dev, Type::VideoCapture, 4) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error creating stream for camera {}: {}", camera_index, e);
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };
            
            use std::os::unix::io::AsRawFd;
            let fd = dev.handle().fd();

            loop {
                // use poll to implement a timeout (2000ms) to prevent freezing
                let mut fds = [libc::pollfd {
                    fd,
                    events: libc::POLLIN,
                    revents: 0,
                }];

                let ret = unsafe { libc::poll(fds.as_mut_ptr(), 1, 2000) };

                if ret == 0 {
                    eprintln!("Camera {} timeout (no frame for 2s). Resetting connection...", camera_index);
                    break; // break inner loop to trigger full reconnection
                } else if ret < 0 {
                    continue;
                }

                match stream.next() {
                    Ok((buf, _meta)) => {
                        let buf_vec = buf.to_vec();
                        match tx_capture.try_send(buf_vec) {
                            Ok(_) => {},
                            Err(mpsc::TrySendError::Full(_)) => {},
                            Err(mpsc::TrySendError::Disconnected(_)) => return, // exit outer loop, pipeline dead
                        }
                    }
                    Err(_e) => {
                        eprintln!("Stream error cam {}: restarting stream...", camera_index);
                        break; // break inner loop to trigger full reconnection
                    }
                }
            }
            // clean up old device/stream (Dropping them does this) and retry
            std::thread::sleep(Duration::from_millis(500));
        }
    });

    // 2. decode thread
    std::thread::spawn(move || {
        let mut decompressor = match Decompressor::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error creating decompressor for cam {}: {}", camera_index, e);
                return;
            }
        };

        loop {
            if let Ok(buf) = rx_capture.recv() {
                let header = match decompressor.read_header(&buf) {
                    Ok(h) => h,
                    Err(_) => continue,
                };
                
                let width = header.width;
                let height = header.height;
                let mut pixels = vec![0u8; width * height];
                let image = Image {
                    pixels: pixels.as_mut_slice(),
                    width,
                    pitch: width,
                    height,
                    format: PixelFormat::GRAY,
                };
                
                if decompressor.decompress(&buf, image).is_ok() {
                    match tx_decode.try_send((pixels, width, height)) {
                        Ok(_) => {},
                        Err(mpsc::TrySendError::Full(_)) => {}, // drop if detector is busy
                        Err(mpsc::TrySendError::Disconnected(_)) => break,
                    }
                }
            }
        }
    });

    // 3. detect thread
    std::thread::spawn(move || {
        enum DetectorWrapper {
            Cpu(detector::CpuDetector),
            #[cfg(feature = "gpu")]
            Gpu(gpu_detector::GpuDetector),
            #[cfg(feature = "tensorrt")]
            Yolo(yolo_detector::YoloDetector),
        }

        impl DetectorWrapper {
            fn name(&self) -> &'static str {
                match self {
                    DetectorWrapper::Cpu(_) => "apriltag-cpu",
                    #[cfg(feature = "gpu")]
                    DetectorWrapper::Gpu(_) => "apriltag-gpu",
                    #[cfg(feature = "tensorrt")]
                    DetectorWrapper::Yolo(_) => "yolo-tensorrt",
                }
            }

            fn detect(
                &mut self,
                data: &[u8],
                width: usize,
                height: usize,
                processing_config: &ProcessingConfig,
            ) -> anyhow::Result<Vec<crate::detector::Detection>> {
                match self {
                    DetectorWrapper::Cpu(d) => {
                        let scale = processing_config.resolution_scale_factor.clamp(0.1, 1.0);
                        if (scale - 1.0).abs() < f32::EPSILON {
                            return d.detect(data, width, height);
                        }

                        let scaled_width = ((width as f32) * scale).round().max(1.0) as usize;
                        let scaled_height = ((height as f32) * scale).round().max(1.0) as usize;
                        let scaled = resize_gray_nearest(data, width, height, scaled_width, scaled_height);
                        let mut out = d.detect(&scaled, scaled_width, scaled_height)?;
                        rescale_apriltag_detections(&mut out, 1.0 / scale as f64);
                        Ok(out)
                    }
                    #[cfg(feature = "gpu")]
                    DetectorWrapper::Gpu(d) => d.detect(data, width, height),
                    #[cfg(feature = "tensorrt")]
                    DetectorWrapper::Yolo(d) => d.detect(data, width, height),
                }
            }

            fn get_effective_config(&self, original_config: &CameraConfig) -> CameraConfig {
                match self {
                    DetectorWrapper::Cpu(_) => original_config.clone(),
                    #[cfg(feature = "gpu")]
                    DetectorWrapper::Gpu(d) => d.scaled_config.clone(),
                    #[cfg(feature = "tensorrt")]
                    DetectorWrapper::Yolo(_) => {
                        original_config.clone()
                    }
                }
            }
            
            fn requires_undistort(&self) -> bool {
                match self {
                    DetectorWrapper::Cpu(_) => true,
                    #[cfg(feature = "gpu")]
                    DetectorWrapper::Gpu(_) => false,
                    #[cfg(feature = "tensorrt")]
                    DetectorWrapper::Yolo(_) => true,
                }
            }
        }

        let mut detectors: Vec<DetectorWrapper> = Vec::new();
        let mut saved_debug_frame = false;

        loop {
            if let Ok((mut pixels, width, height)) = rx_decode.recv() {
                apply_processing(&mut pixels, &processing_config);
                if !saved_debug_frame {
                    match Compressor::new() {
                        Ok(mut compressor) => {
                            let image = Image {
                                pixels: pixels.as_slice(),
                                width,
                                pitch: width,
                                height,
                                format: PixelFormat::GRAY,
                            };
                            compressor.set_subsamp(Subsamp::Gray);
                            
                            match compressor.compress_to_owned(image) {
                                Ok(jpg_data) => {
                                    let filename = format!("output/debug_cam{}.jpg", camera_index);
                                    println!("DEBUG: Attempting to save frame to {}", filename);
                                    if let Err(e) = std::fs::write(&filename, jpg_data) {
                                        eprintln!("Failed to write debug frame: {}", e);
                                    } else {
                                        let abs_path = std::fs::canonicalize(&filename).unwrap_or(PathBuf::from(&filename));
                                        println!("Saved debug frame to {}", abs_path.display());
                                        saved_debug_frame = true;
                                    }
                                },
                                Err(e) => eprintln!("Failed to compress debug frame: {}", e),
                            }
                        },
                        Err(e) => eprintln!("Failed to create compressor: {}", e),
                    }
                }

                if detectors.is_empty() {
                    let mut tag_initialized = false;

                    #[cfg(feature = "gpu")]
                    {
                        let gpu_enabled = std::env::var("VORTEX_MAIN_GPU")
                            .ok()
                            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                            .unwrap_or(false);
                        if gpu_enabled {
                            let scale_factor = processing_config.resolution_scale_factor;
                            if let Ok(d) = gpu_detector::GpuDetector::new(width, height, &camera_config, scale_factor) {
                                println!("Initialized GPU Detector for Camera {} (Scale: {})", camera_index, scale_factor);
                                detectors.push(DetectorWrapper::Gpu(d));
                                tag_initialized = true;
                            } else {
                                eprintln!("Error building GPU detector for cam {}: Falling back to CPU.", camera_index);
                            }
                        } else {
                            eprintln!("GPU detector disabled for main pipeline (set VORTEX_MAIN_GPU=1 to enable). Using CPU.");
                        }
                    }

                    if !tag_initialized {
                        if let Ok(d) = detector::CpuDetector::new(detector_threads) {
                            println!("Initialized CPU Detector (Fallback) for Camera {}", camera_index);
                            detectors.push(DetectorWrapper::Cpu(d));
                            tag_initialized = true;
                        } else {
                            eprintln!("Error building CPU detector for cam {}: No AprilTag detector available.", camera_index);
                        }
                    }

                    #[cfg(feature = "tensorrt")]
                    {
                        if object_detection_config.use_nn {
                            if let Ok(d) = yolo_detector::YoloDetector::new() {
                                println!("Initialized YoloDetector for Camera {}", camera_index);
                                detectors.push(DetectorWrapper::Yolo(d));
                            } else {
                                eprintln!("Error building YoloDetector for cam {}: Skipping YOLO detection.", camera_index);
                            }
                        } else {
                            println!("YOLO disabled by config for Camera {}", camera_index);
                        }
                    }

                    if detectors.is_empty() {
                        eprintln!("No detectors initialized for cam {}. Exiting pipeline.", camera_index);
                        return;
                    }
                }

                let mut processed_detections = Vec::new();

                for detector in &mut detectors {
                    let raw_dets = match detector.detect(&pixels, width, height, &processing_config) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!("Detector {} failed on cam {}: {}", detector.name(), camera_index, e);
                            continue;
                        }
                    };

                    let effective_config = detector.get_effective_config(&camera_config);
                    let needs_undistort = detector.requires_undistort();

                    for det in raw_dets {
                        match det {
                            Detection::AprilTag(apr_det) => {
                                // Normalize corner order for pose solver: TL, BL, BR, TR
                                let c = apr_det.corners;
                                let corners_raw = [
                                    (c[3][0], c[3][1]),
                                    (c[0][0], c[0][1]),
                                    (c[1][0], c[1][1]),
                                    (c[2][0], c[2][1]),
                                ];

                                let corners = if needs_undistort {
                                    crate::undistort::undistort_points(&corners_raw, &effective_config)
                                } else {
                                    corners_raw
                                };

                                let tag_size = effective_config.tag_size_m;
                                let (x, y, z, floor_z_error) = if let Some(pose) = pose::estimate_pose(
                                    &corners,
                                    tag_size,
                                    effective_config.fx, effective_config.fy, effective_config.cx, effective_config.cy
                                ) {
                                    if let Some(tag_field) = tag_map_by_id.get(&apr_det.id) {
                                        let (p_field_robot, z_err) =
                                            estimate_robot_field_from_tag(&pose, tag_field, &effective_config);
                                        (p_field_robot.x, p_field_robot.y, 0.0, z_err)
                                    } else {
                                        // fallback: map missing this tag -> keep robot-frame estimate
                                        let p_robot = transform_camera_to_robot(
                                            pose.translation,
                                            &effective_config
                                        );
                                        (p_robot.x, p_robot.y, p_robot.z, 0.0)
                                    }
                                } else {
                                    // fallback to simple estimation
                                    let side_len_px = (
                                        ((corners[0].0 - corners[1].0).powi(2) + (corners[0].1 - corners[1].1).powi(2)).sqrt() +
                                        ((corners[1].0 - corners[2].0).powi(2) + (corners[1].1 - corners[2].1).powi(2)).sqrt() +
                                        ((corners[2].0 - corners[3].0).powi(2) + (corners[2].1 - corners[3].1).powi(2)).sqrt() +
                                        ((corners[3].0 - corners[0].0).powi(2) + (corners[3].1 - corners[0].1).powi(2)).sqrt()
                                    ) / 4.0;

                                    let z = (effective_config.fx * tag_size) / side_len_px;
                                    let center_x = (corners[0].0 + corners[2].0) / 2.0;
                                    let center_y = (corners[0].1 + corners[2].1) / 2.0;
                                    let x = (center_x - effective_config.cx) * z / effective_config.fx;
                                    let y = (center_y - effective_config.cy) * z / effective_config.fy;
                                    let p_robot = transform_camera_to_robot(Vector3::new(x, y, z), &effective_config);
                                    (p_robot.x, p_robot.y, p_robot.z, 0.0)
                                };

                                processed_detections.push(ProcessedDetection::AprilTag(AprilTagPose {
                                    id: apr_det.id,
                                    x,
                                    y,
                                    z,
                                    floor_z_error,
                                }));
                            }
                            Detection::Yolo(yolo_det) => {
                                let conf_threshold = object_detection_config.confidence_threshold.clamp(0.0, 1.0);
                                if yolo_det.confidence < conf_threshold {
                                    continue;
                                }
                                let bbox = yolo_det.bbox;
                                let u_raw = bbox[0] + bbox[2] / 2.0;
                                let v_raw = bbox[1] + bbox[3] / 2.0;
                                let (u, v) = if needs_undistort {
                                    crate::undistort::undistort_point((u_raw, v_raw), &effective_config)
                                } else {
                                    (u_raw, v_raw)
                                };

                                // approximate object depth from known nominal object size and detected bbox size, tune with YOLO_OBJ_WIDTH_M / YOLO_OBJ_HEIGHT_M
                                let obj_w_m = object_detection_config.yolo_obj_width_m;
                                let obj_h_m = object_detection_config.yolo_obj_height_m;

                                let mut z_candidates = Vec::new();
                                if bbox[2] > 1.0 {
                                    z_candidates.push((effective_config.fx * obj_w_m) / bbox[2]);
                                }
                                if bbox[3] > 1.0 {
                                    z_candidates.push((effective_config.fy * obj_h_m) / bbox[3]);
                                }
                                if z_candidates.is_empty() {
                                    continue;
                                }
                                let z_cam = z_candidates.iter().sum::<f64>() / z_candidates.len() as f64;
                                let x_cam = (u - effective_config.cx) * z_cam / effective_config.fx;
                                let y_cam = (v - effective_config.cy) * z_cam / effective_config.fy;

                                let p_robot = transform_camera_to_robot(
                                    Vector3::new(x_cam, y_cam, z_cam),
                                    &effective_config
                                );

                                processed_detections.push(ProcessedDetection::Yolo(YoloBBox {
                                    class_name: yolo_det.class_name,
                                    confidence: yolo_det.confidence,
                                    bbox,
                                    x: p_robot.x,
                                    y: p_robot.y,
                                    z: p_robot.z,
                                }));
                            }
                        }
                    }
                }

                let stat = PipelineStats {
                    camera_index,
                    detections: ProcessedDetections(processed_detections),
                    timestamp: Instant::now(),
                };

                if tx_stats.send(stat).is_err() {
                    eprintln!("Stats channel disconnected for cam {}", camera_index);
                    break; // exit loop, thread will terminate
                }
            }
        }
    });
}

fn transform_camera_to_robot(p_cam: Vector3<f64>, config: &CameraConfig) -> Vector3<f64> {
    let r_yaw = Rotation3::from_axis_angle(&Vector3::y_axis(), config.yaw_deg.to_radians());
    let r_pitch = Rotation3::from_axis_angle(&Vector3::x_axis(), config.pitch_deg.to_radians());
    let r_roll = Rotation3::from_axis_angle(&Vector3::z_axis(), config.roll_deg.to_radians());
    let rotation = r_yaw * r_pitch * r_roll;
    rotation * p_cam + Vector3::new(config.x_offset, config.y_offset, config.z_offset)
}

fn camera_to_robot_rotation(config: &CameraConfig) -> Matrix3<f64> {
    let r_yaw = Rotation3::from_axis_angle(&Vector3::y_axis(), config.yaw_deg.to_radians());
    let r_pitch = Rotation3::from_axis_angle(&Vector3::x_axis(), config.pitch_deg.to_radians());
    let r_roll = Rotation3::from_axis_angle(&Vector3::z_axis(), config.roll_deg.to_radians());
    (r_yaw * r_pitch * r_roll).into_inner()
}

fn estimate_robot_field_from_tag(
    pose: &pose::Pose,
    tag_field: &FieldTagPose,
    camera_cfg: &CameraConfig,
) -> (Vector3<f64>, f64) {
    // pose is tag->camera.
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

fn load_tag_map(path: &Path) -> HashMap<usize, FieldTagPose> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };
    let parsed: TagMap = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let mut out = HashMap::new();
    for tag in parsed.tags {
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            tag.pose.rotation.quaternion.w,
            tag.pose.rotation.quaternion.x,
            tag.pose.rotation.quaternion.y,
            tag.pose.rotation.quaternion.z,
        ));
        out.insert(
            tag.id,
            FieldTagPose {
                pos: Vector3::new(
                    tag.pose.translation.x,
                    tag.pose.translation.y,
                    tag.pose.translation.z,
                ),
                // Match bridge behavior for current tag-map convention.
                rot_field_from_tag: q.to_rotation_matrix().into_inner().transpose(),
            },
        );
    }
    out
}

fn resolve_tag_map_path() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = env::var("VORTEX_TAG_MAP") {
        if !p.trim().is_empty() {
            candidates.push(PathBuf::from(p));
        }
    }
    candidates.push(PathBuf::from("config/apriltag_map.json"));
    candidates.push(PathBuf::from("../config/apriltag_map.json"));
    candidates.push(PathBuf::from("/home/vortex/deployments/Vortex/config/apriltag_map.json"));
    candidates.push(PathBuf::from("/home/jetson/deployments/Vortex/config/apriltag_map.json"));

    candidates
        .into_iter()
        .find(|p| std::fs::metadata(p).map(|m| m.is_file()).unwrap_or(false))
}

fn robust_fuse_field_pose(candidates: &[(f64, f64, f64, f64)]) -> Option<(f64, f64, usize, f64)> {
    // (x, y, floor_z_error, cam_depth)
    const MAX_FLOOR_ERR_M: f64 = 3.0;
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

fn apply_processing(gray: &mut [u8], cfg: &ProcessingConfig) {
    let gain = cfg.sensor_gain.max(0.01);
    let black_offset = cfg.black_level_offset;
    // approximate color balance as a luminance gain term for grayscale
    let wb_gain = ((cfg.red_balance.max(0.0) + cfg.blue_balance.max(0.0)) * 0.5 / 1600.0)
        .clamp(0.25, 4.0);
    let total_gain = gain * wb_gain;

    for p in gray.iter_mut() {
        let mut v = (*p as f64) / 255.0;
        v *= total_gain;
        v += black_offset / 255.0;
        v = v.clamp(0.0, 1.0);
        *p = (v * 255.0).clamp(0.0, 255.0) as u8;
    }
}

fn resize_gray_nearest(
    input: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<u8> {
    if src_width == dst_width && src_height == dst_height {
        return input.to_vec();
    }

    let mut out = vec![0u8; dst_width * dst_height];
    for y in 0..dst_height {
        let src_y = ((y as f64 + 0.5) * src_height as f64 / dst_height as f64)
            .floor()
            .clamp(0.0, (src_height - 1) as f64) as usize;
        for x in 0..dst_width {
            let src_x = ((x as f64 + 0.5) * src_width as f64 / dst_width as f64)
                .floor()
                .clamp(0.0, (src_width - 1) as f64) as usize;
            out[y * dst_width + x] = input[src_y * src_width + src_x];
        }
    }
    out
}

fn rescale_apriltag_detections(detections: &mut [Detection], scale_back: f64) {
    for det in detections {
        if let Detection::AprilTag(apr) = det {
            apr.center[0] *= scale_back;
            apr.center[1] *= scale_back;
            for corner in &mut apr.corners {
                corner[0] *= scale_back;
                corner[1] *= scale_back;
            }
        }
    }
}
