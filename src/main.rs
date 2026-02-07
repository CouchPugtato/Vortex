mod detector;
mod config;
mod undistort;
mod preprocess;

use nalgebra as _;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use v4l::{Device, FourCC};
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::video::Capture;
use v4l::io::traits::CaptureStream;
use turbojpeg::{Decompressor, Image, PixelFormat};
use image::GrayImage as _;

use crate::config::CameraConfig;

fn main() -> anyhow::Result<()> {
    let mut args = env::args();
    let _prog = args.next();
    let first = args.next();
    let second = args.next();
    
    let (camera_index, output_dir) = match (first.as_deref(), second.as_deref()) {
        (Some(s), Some(out)) if s.chars().all(|c| c.is_ascii_digit()) => {
            (s.parse::<usize>().unwrap_or(0), PathBuf::from(out))
        }
        (Some(s), None) if s.chars().all(|c| c.is_ascii_digit()) => {
            (s.parse::<usize>().unwrap_or(0), PathBuf::from("output"))
        }
        (Some(out), _) => (0, PathBuf::from(out)),
        _ => (0, PathBuf::from("output")),
    };

    println!("Starting application (V4L2 + Multithreaded Detector)...");
    fs::create_dir_all(&output_dir)?;

    let config_path = Path::new("config/distortion.json");
    let cam_config = match CameraConfig::load(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load config from {:?}: {}", config_path, e);
            return Err(anyhow::anyhow!("Config load failed"));
        }
    };
    println!("Loaded camera config: {:?}", cam_config);

    run_capture_loop(camera_index, output_dir, cam_config)?;

    Ok(())
}

fn run_capture_loop(camera_index: usize, _output_dir: PathBuf, cam_config: CameraConfig) -> anyhow::Result<()> {
    println!("Opening camera {}...", camera_index);
    
    // original detector initialization is kept for the main thread usage
    // let mut detector = detector::build_detector()?;
    
    let mut frame_count = 0;
    let mut last_report = Instant::now();

    // pipelining:
    // thread 1: captures frames -> channel a
    // thread 2: decodes frames -> channel b
    // thread 3: detects tags -> channel c
    // thread 4: network sender (sim)
    
    let (tx_capture, rx_capture) = std::sync::mpsc::sync_channel(1); 
    let (tx_decode, rx_decode) = std::sync::mpsc::sync_channel(1);
    let (tx_network, rx_network) = std::sync::mpsc::sync_channel(10); // buffer detections

    // 1. capture thread
    std::thread::spawn(move || {
        let dev = Device::new(camera_index).unwrap();
        let mut fmt = dev.format().unwrap();
        fmt.width = 1920;
        fmt.height = 1080;
        fmt.fourcc = FourCC::new(b"MJPG");
        dev.set_format(&fmt).unwrap();
        
        let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4).unwrap();
        
        loop {
            if let Ok((buf, _meta)) = stream.next() {
                let buf_vec = buf.to_vec();
                if tx_capture.send(buf_vec).is_err() { break; }
            }
        }
    });

    // 2. decode thread
    std::thread::spawn(move || {
        let mut decompressor = Decompressor::new().unwrap();
        loop {
            if let Ok(buf) = rx_capture.recv() {
                let header = decompressor.read_header(&buf).unwrap();
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
                    if tx_decode.send((pixels, width, height)).is_err() { break; }
                }
            }
        }
    });

    // 4. network sender thread (simulated)
    std::thread::spawn(move || {
        println!("Network Sender started...");
        loop {
            if let Ok(detections) = rx_network.recv() {
                // simulate networktables / udp overhead (<1ms in theory)
                let _len: usize = detections; 
                // std::thread::sleep(std::time::Duration::from_micros(100)); 
            }
        }
    });

    // 3. detect loop (main thread)
    println!("Starting Pipelined Detection...");
    
    // for 1 camera, use 6 threads. for 4 cameras, you'd set this to 1 or 2.
    let mut detector = detector::build_detector(6)?; 
    
    loop {
        if let Ok((pixels, width, height)) = rx_decode.recv() {
            let corners = detector::detect_corners(&mut detector, &pixels, width, height)?;
            
            // send to network thread
            let _ = tx_network.send(corners.len());

            frame_count += 1;
            let now = Instant::now();
            if now.duration_since(last_report).as_secs() >= 1 {
                let fps = frame_count as f64 / now.duration_since(last_report).as_secs_f64();
                println!("FPS: {:.2} | Detections: {}", fps, corners.len());
                if !corners.is_empty() {
                    println!("  Tag 0 corners: {:?}", corners[0]);
                }
                frame_count = 0;
                last_report = now;
            }
        }
    }
}
