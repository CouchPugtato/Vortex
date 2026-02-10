mod detector;
mod gpu_detector;
mod vpi;
mod config;
mod undistort;
mod preprocess;
mod pose;

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

use crate::config::CameraConfig;

#[derive(Debug, Clone)]
struct DetectionInfo {
    id: usize,
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug)]
struct PipelineStats {
    camera_index: usize,
    detections: Vec<DetectionInfo>,
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

    if camera_indices.is_empty() {
        camera_indices.push(0);
    }

    println!("Starting Multi-Camera AprilTag Detector");
    println!("Cameras: {:?}", camera_indices);
    println!("Output Dir: {:?}", output_dir_base);

    fs::create_dir_all(&output_dir_base)?;

    let config_path = Path::new("config/distortion.json");
    let cam_config = match CameraConfig::load(config_path) {
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
        spawn_camera_pipeline(idx, threads_per_cam, tx_stats.clone(), cam_config.clone());
    }

    // monitor loop
    let mut cam_stats: HashMap<usize, (u64, Instant)> = HashMap::new(); // (frame_count, last_report)
    let mut cam_fps: HashMap<usize, f64> = HashMap::new();
    let mut cam_detections: HashMap<usize, Vec<DetectionInfo>> = HashMap::new();
    
    // simple exponential smoothing for pose
    // store last known pose for each tag id per camera
    let mut pose_filters: HashMap<(usize, usize), (f64, f64, f64)> = HashMap::new(); // (cam_idx, tag_id) -> (x, y, z)
    let alpha = 0.1; // Reduced from 0.3 to 0.1 for stronger smoothing

    // init stats
    let start_time = Instant::now();
    for &idx in &camera_indices {
        cam_stats.insert(idx, (0, start_time));
        cam_fps.insert(idx, 0.0);
        cam_detections.insert(idx, Vec::new());
    }

    loop {
        if let Ok(stat) = rx_stats.recv() {
            // apply smoothing to detections
            let mut smoothed_detections = Vec::new();
            for det in stat.detections {
                let key = (stat.camera_index, det.id);
                let (s_x, s_y, s_z) = if let Some(&(last_x, last_y, last_z)) = pose_filters.get(&key) {
                    (
                        last_x + alpha * (det.x - last_x),
                        last_y + alpha * (det.y - last_y),
                        last_z + alpha * (det.z - last_z),
                    )
                } else {
                    (det.x, det.y, det.z)
                };
                
                pose_filters.insert(key, (s_x, s_y, s_z));
                
                smoothed_detections.push(DetectionInfo {
                    id: det.id,
                    x: s_x,
                    y: s_y,
                    z: s_z,
                });
            }

            // Update last known detections for this camera
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
                        println!("Camera {}: {:.2} FPS | Last Detections: {}", 
                            idx, 
                            cam_fps.get(&idx).unwrap_or(&0.0),
                            detections.len()
                        );
                        for det in detections {
                            println!("  - ID: {} | Dist: {:.2}m | X: {:.2}m | Y: {:.2}m", 
                                det.id, det.z, det.x, det.y);
                        }
                    }
                }
            }
        }
    }
}

fn spawn_camera_pipeline(
    camera_index: usize, 
    _detector_threads: i32, 
    tx_stats: mpsc::Sender<PipelineStats>,
    config: CameraConfig
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
            
            fmt.width = 1920;
            fmt.height = 1080;
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
        }

        impl DetectorWrapper {
            fn detect(&mut self, data: &[u8], width: usize, height: usize) -> anyhow::Result<Vec<crate::detector::RawDetection>> {
                match self {
                    DetectorWrapper::Cpu(d) => d.detect(data, width, height),
                    #[cfg(feature = "gpu")]
                    DetectorWrapper::Gpu(d) => d.detect(data, width, height),
                }
            }

            fn get_effective_config(&self, original_config: &CameraConfig) -> CameraConfig {
                match self {
                    DetectorWrapper::Cpu(_) => original_config.clone(),
                    #[cfg(feature = "gpu")]
                    DetectorWrapper::Gpu(d) => d.scaled_config.clone(),
                }
            }
            
            fn requires_undistort(&self) -> bool {
                match self {
                    DetectorWrapper::Cpu(_) => true,
                    #[cfg(feature = "gpu")]
                    DetectorWrapper::Gpu(_) => false,
                }
            }
        }

        let mut detector: Option<DetectorWrapper> = None;
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

                if detector.is_none() {
                    #[cfg(feature = "gpu")]
                    {
                        let scale_factor = 1.0; // 1.0 as full resolution
                        match gpu_detector::GpuDetector::new(width, height, &config, scale_factor) {
                            Ok(d) => {
                                println!("Initialized GPU Detector for Camera {} (Scale: {})", camera_index, scale_factor);
                                detector = Some(DetectorWrapper::Gpu(d));
                            },
                            Err(e) => {
                                eprintln!("Error building GPU detector for cam {}: {}. Falling back to CPU.", camera_index, e);
                                match detector::CpuDetector::new(_detector_threads) {
                                    Ok(d) => {
                                        println!("Initialized CPU Detector (Fallback) for Camera {}", camera_index);
                                        detector = Some(DetectorWrapper::Cpu(d));
                                    },
                                    Err(e) => {
                                        eprintln!("Error building CPU detector for cam {}: {}", camera_index, e);
                                        return;
                                    }
                                }
                            }
                        }
                    }

                    #[cfg(not(feature = "gpu"))]
                    {
                        match detector::CpuDetector::new(_detector_threads) {
                            Ok(d) => detector = Some(DetectorWrapper::Cpu(d)),
                            Err(e) => {
                                eprintln!("Error building CPU detector for cam {}: {}", camera_index, e);
                                return;
                            }
                        }
                    }
                }

                let detections = match detector.as_mut().unwrap().detect(&pixels, width, height) {
                    Ok(c) => c,
                    Err(_) => vec![],
                };
                
                let mut detection_infos = Vec::new();
                let effective_config = detector.as_ref().unwrap().get_effective_config(&config);
                let needs_undistort = detector.as_ref().unwrap().requires_undistort();

                for det in detections {
                    let corners_raw = [
                        (det.corners[0][0], det.corners[0][1]),
                        (det.corners[1][0], det.corners[1][1]),
                        (det.corners[2][0], det.corners[2][1]),
                        (det.corners[3][0], det.corners[3][1]),
                    ];
                    
                    let corners = if needs_undistort {
                         crate::undistort::undistort_points(&corners_raw, &effective_config)
                    } else {
                         corners_raw
                    };
                    
                    let tag_size = 0.1651; // m
                    
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

                    // apply camera offset 
                    let p_cam = Vector3::new(x, y, z);
                    
                    let r_yaw = Rotation3::from_axis_angle(&Vector3::y_axis(), effective_config.yaw_deg.to_radians());
                    let r_pitch = Rotation3::from_axis_angle(&Vector3::x_axis(), effective_config.pitch_deg.to_radians());
                    let r_roll = Rotation3::from_axis_angle(&Vector3::z_axis(), effective_config.roll_deg.to_radians());
                    
                    let rotation = r_yaw * r_pitch * r_roll;
                    let p_robot = rotation * p_cam + Vector3::new(effective_config.x_offset, effective_config.y_offset, effective_config.z_offset);
                    
                    let (x, y, z) = (p_robot.x, p_robot.y, p_robot.z);
                    
                    detection_infos.push(DetectionInfo {
                        id: det.id,
                        x,
                        y,
                        z,
                    });
                }
                
                let stat = PipelineStats {
                    camera_index,
                    detections: detection_infos,
                    timestamp: Instant::now(),
                };

                if tx_stats.send(stat).is_err() { break; }
            }
        }
    });
}
