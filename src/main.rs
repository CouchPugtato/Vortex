mod detector;
mod config;
mod undistort;
mod preprocess;

use nalgebra as _;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::mpsc;

use v4l::{Device, FourCC};
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::video::Capture;
use v4l::io::traits::CaptureStream;
use turbojpeg::{Decompressor, Image, PixelFormat};
use image::GrayImage as _;

use crate::config::CameraConfig;

#[derive(Debug)]
struct PipelineStats {
    camera_index: usize,
    detections: usize,
    timestamp: Instant,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    
    // parse camera indices from args
    // examples: "0", "0,1", "0 2", "0, 2"
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
    // jetson orin nano has 6 cores. reserve 2 for system/capture/decode, 4 for detection?
    // or just oversubscribe slightly.
    // let's try to target 6 total detector threads.
    let total_detector_threads = 6;
    let threads_per_cam = std::cmp::max(1, total_detector_threads / camera_indices.len() as i32);
    
    println!("Allocating {} detector threads per camera", threads_per_cam);

    let (tx_stats, rx_stats) = mpsc::channel();

    for &idx in &camera_indices {
        spawn_camera_pipeline(idx, threads_per_cam, tx_stats.clone());
    }

    // monitor loop
    let mut cam_stats: HashMap<usize, (u64, Instant)> = HashMap::new(); // (frame_count, last_report)
    let mut cam_fps: HashMap<usize, f64> = HashMap::new();
    
    // init stats
    let start_time = Instant::now();
    for &idx in &camera_indices {
        cam_stats.insert(idx, (0, start_time));
        cam_fps.insert(idx, 0.0);
    }

    loop {
        if let Ok(stat) = rx_stats.recv() {
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
                        println!("Camera {}: {:.2} FPS | Last Detections: {}", 
                            idx, 
                            cam_fps.get(&idx).unwrap_or(&0.0),
                            stat.detections // this is just the most recent one, acceptable for now
                        );
                    }
                }
            }
        }
    }
}

fn spawn_camera_pipeline(
    camera_index: usize, 
    detector_threads: i32, 
    tx_stats: mpsc::Sender<PipelineStats>
) {
    println!("Spawning pipeline for Camera {}...", camera_index);

    let (tx_capture, rx_capture) = mpsc::sync_channel(1); 
    let (tx_decode, rx_decode) = mpsc::sync_channel(1);

    // 1. capture thread
    std::thread::spawn(move || {
        let dev = match Device::new(camera_index) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error opening camera {}: {}", camera_index, e);
                return;
            }
        };

        let mut fmt = match dev.format() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error getting format for camera {}: {}", camera_index, e);
                return;
            }
        };
        
        fmt.width = 1920;
        fmt.height = 1080;
        fmt.fourcc = FourCC::new(b"MJPG");
        
        if let Err(e) = dev.set_format(&fmt) {
            eprintln!("Error setting format for camera {}: {}", camera_index, e);
            return;
        }
        
        let mut stream = match MmapStream::with_buffers(&dev, Type::VideoCapture, 4) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error creating stream for camera {}: {}", camera_index, e);
                return;
            }
        };
        
        loop {
            match stream.next() {
                Ok((buf, _meta)) => {
                    let buf_vec = buf.to_vec();
                    // Use try_send to drop frames if the pipeline is backed up (latency optimization)
                    match tx_capture.try_send(buf_vec) {
                        Ok(_) => {},
                        Err(mpsc::TrySendError::Full(_)) => {
                            // buffer full, drop frame
                        },
                        Err(mpsc::TrySendError::Disconnected(_)) => break,
                    }
                }
                Err(e) => {
                    // eprintln!("Stream error cam {}: {}", camera_index, e);
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
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
        let mut detector = match detector::build_detector(detector_threads) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error building detector for cam {}: {}", camera_index, e);
                return;
            }
        };

        loop {
            if let Ok((pixels, width, height)) = rx_decode.recv() {
                let corners = match detector::detect_corners(&mut detector, &pixels, width, height) {
                    Ok(c) => c,
                    Err(_) => vec![],
                };
                
                let stat = PipelineStats {
                    camera_index,
                    detections: corners.len(),
                    timestamp: Instant::now(),
                };

                if tx_stats.send(stat).is_err() { break; }
            }
        }
    });
}
