mod detector;
mod gpu_detector;
mod vpi;
mod config;
mod undistort;
mod preprocess;
mod pose;
mod yolo_detector;

use nalgebra as _;
use nalgebra::{Vector3, Rotation3};
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

use crate::config::{CameraConfig, ProcessingConfig, RuntimeConfig};
use crate::detector::Detection;




#[derive(Debug, Clone)]
pub struct AprilTagPose {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
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
            runtime_config.camera,
            runtime_config.processing,
        );
    }

    // monitor loop
    let mut cam_stats: HashMap<usize, (u64, Instant)> = HashMap::new(); // (frame_count, last_report)
    let mut cam_fps: HashMap<usize, f64> = HashMap::new();
    let mut cam_detections: HashMap<usize, Vec<ProcessedDetection>> = HashMap::new();
    
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
                        for det in detections {
                            match det {
                                ProcessedDetection::AprilTag(a) => {
                                    println!("  - Tag ID: {} | Dist: {:.2}m | X: {:.2}m | Y: {:.2}m", 
                                        a.id, a.z, a.x, a.y);
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
            if let Ok((pixels, width, height)) = rx_decode.recv() {
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
                        let scale_factor = processing_config.resolution_scale_factor;
                        if let Ok(d) = gpu_detector::GpuDetector::new(width, height, &camera_config, scale_factor) {
                            println!("Initialized GPU Detector for Camera {} (Scale: {})", camera_index, scale_factor);
                            detectors.push(DetectorWrapper::Gpu(d));
                            tag_initialized = true;
                        } else {
                            eprintln!("Error building GPU detector for cam {}: Falling back to CPU.", camera_index);
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
                        if let Ok(d) = yolo_detector::YoloDetector::new() {
                            println!("Initialized YoloDetector for Camera {}", camera_index);
                            detectors.push(DetectorWrapper::Yolo(d));
                        } else {
                            eprintln!("Error building YoloDetector for cam {}: Skipping YOLO detection.", camera_index);
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
                                let corners_raw = [
                                    (apr_det.corners[0][0], apr_det.corners[0][1]),
                                    (apr_det.corners[1][0], apr_det.corners[1][1]),
                                    (apr_det.corners[2][0], apr_det.corners[2][1]),
                                    (apr_det.corners[3][0], apr_det.corners[3][1]),
                                ];

                                let corners = if needs_undistort {
                                    crate::undistort::undistort_points(&corners_raw, &effective_config)
                                } else {
                                    corners_raw
                                };

                                let tag_size = effective_config.tag_size_m;

                                let (x, y, z) = if let Some(pose) = pose::estimate_pose(
                                    &corners,
                                    tag_size,
                                    effective_config.fx, effective_config.fy, effective_config.cx, effective_config.cy
                                ) {
                                    (pose.translation.x, pose.translation.y, pose.translation.z)
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
                                    (x, y, z)
                                };

                                let p_robot = transform_camera_to_robot(Vector3::new(x, y, z), &effective_config);

                                let (x, y, z) = (p_robot.x, p_robot.y, p_robot.z);

                                processed_detections.push(ProcessedDetection::AprilTag(AprilTagPose {
                                    id: apr_det.id,
                                    x,
                                    y,
                                    z,
                                }));
                            }
                            Detection::Yolo(yolo_det) => {
                                let bbox = yolo_det.bbox;
                                let u_raw = bbox[0] + bbox[2] / 2.0;
                                let v_raw = bbox[1] + bbox[3] / 2.0;
                                let (u, v) = if needs_undistort {
                                    crate::undistort::undistort_point((u_raw, v_raw), &effective_config)
                                } else {
                                    (u_raw, v_raw)
                                };

                                // approximate object depth from known nominal object size and detected bbox size, tune with YOLO_OBJ_WIDTH_M / YOLO_OBJ_HEIGHT_M
                                let obj_w_m = processing_config.yolo_obj_width_m;
                                let obj_h_m = processing_config.yolo_obj_height_m;

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
