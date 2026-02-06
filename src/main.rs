mod detector;
mod config;
mod undistort;

use nalgebra as na;
use apriltag::Detector as AprilDetector;

use std::env;
use std::fs;
use std::io::{Write as IoWrite, Cursor};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::thread;

use image::RgbImage;
use v4l::{Device, FourCC};
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::video::Capture;
use v4l::io::traits::CaptureStream;
use jpeg_decoder as jpeg;

use crate::config::CameraConfig;

fn main() -> std::io::Result<()> {
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

    println!("Starting application...");
    fs::create_dir_all(&output_dir)?;

    let config_path = Path::new("config/distortion.json");
    let cam_config = match CameraConfig::load(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load config from {:?}: {}", config_path, e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    };
    println!("Loaded camera config: {:?}", cam_config);

    let output_dir_clone = output_dir.clone();
    let cam_config_clone = cam_config.clone();

    let mut cam_idx = camera_index;
    
    if Device::new(cam_idx).is_err() {
        println!("Camera {} not found, searching for available cameras...", cam_idx);
        if let Some(idx) = find_available_camera() {
            println!("Found camera at index {}", idx);
            cam_idx = idx;
        } else {
            eprintln!("No cameras found. Falling back to INPUT_DIR processing if available.");
            let mut detector = match detector::build_detector() {
                Ok(d) => d,
                Err(e) => { eprintln!("Failed to build detector: {}", e); return Ok(()); }
            };
            if let Err(e) = process_input_dir(&output_dir_clone, &mut detector, cam_config_clone) {
                eprintln!("Input dir processing failed: {}", e);
            }
            return Ok(());
        }
    }

    if let Err(e) = run_capture_loop(cam_idx, output_dir_clone, cam_config_clone) {
        eprintln!("Capture loop error: {:?}", e);
    }

    Ok(())
}

fn find_available_camera() -> Option<usize> {
    for i in 0..10 {
        if Device::new(i).is_ok() {
            return Some(i);
        }
    }
    None
}

fn run_capture_loop(mut camera_index: usize, output_dir: PathBuf, cam_config: CameraConfig) -> anyhow::Result<()> {
    println!("using initial camera_index={} output={}", camera_index, output_dir.display());

    let mut detector = detector::build_detector()?;
    let mut frame_idx: u64 = 0;

    loop {
        let dev_result = Device::new(camera_index);
        let mut dev = match dev_result {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to open camera {}: {}. Retrying in 2s...", camera_index, e);
                thread::sleep(Duration::from_secs(2));
                if let Some(new_idx) = find_available_camera() {
                     if new_idx != camera_index {
                         println!("Found available camera at index {}. Switching...", new_idx);
                         camera_index = new_idx;
                     }
                }
                continue;
            }
        };

        let mut fmt = match dev.format() {
             Ok(f) => f,
             Err(e) => {
                 eprintln!("Failed to query format: {}. Retrying...", e);
                 thread::sleep(Duration::from_secs(1));
                 continue;
             }
        };
        fmt.width = 1920;
        fmt.height = 1200;
        fmt.fourcc = FourCC::new(b"MJPG");
        if let Err(e) = dev.set_format(&fmt) {
            eprintln!("failed to set MJPG, falling back: {}", e);
            let mut f2 = dev.format().unwrap_or(fmt);
            f2.fourcc = FourCC::new(b"YUYV");
            if let Err(e) = dev.set_format(&f2) {
                 eprintln!("set YUYV failed: {}. Retrying...", e);
                 thread::sleep(Duration::from_secs(1));
                 continue;
            }
            fmt = f2;
        }

        let mut stream = match MmapStream::new(&dev, Type::VideoCapture) {
            Ok(s) => s,
            Err(e) => {
                 eprintln!("mmap stream failed: {}. Retrying...", e);
                 thread::sleep(Duration::from_secs(1));
                 continue;
            }
        };
        
        println!("Capture started/resumed on camera {}.", camera_index);

        loop {
            frame_idx += 1;
            let data = match stream.next() { 
                Ok((buf, _meta)) => buf.to_vec(), 
                Err(e) => { 
                    eprintln!("stream error: {}", e); 
                    let err_str = e.to_string();
                    if err_str.contains("No such device") || err_str.contains("os error 19") {
                        eprintln!("Device disconnected. Reconnecting...");
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                    continue; 
                } 
            };

            let rgb_opt: Option<RgbImage> = if fmt.fourcc == FourCC::new(b"MJPG") {
                let mut dec = jpeg::Decoder::new(Cursor::new(&data));
                match dec.decode() {
                    Ok(pixels) => {
                        if let Some(info) = dec.info() {
                            RgbImage::from_raw(info.width as u32, info.height as u32, pixels)
                        } else { None }
                    }
                    Err(err) => { eprintln!("jpeg decode error: {}", err); None }
                }
            } else {
                let (w, h) = (fmt.width as usize, fmt.height as usize);
                Some(yuyv_to_rgb(w, h, &data))
            };

            let rgb = match rgb_opt { Some(img) => img, None => { continue; } };
            let gray = image::DynamicImage::ImageRgb8(rgb).to_luma8();

            let corners_list = match detector::detect_corners(&mut detector, &gray) {
                Ok(c) => c,
                Err(err) => { eprintln!("detection error on frame {}: {}", frame_idx, err); continue; }
            };

            let mut undistorted_corners_list = Vec::new();
            for c in &corners_list {
                undistorted_corners_list.push(undistort::undistort_points(c, &cam_config));
            }
            
            let (w, h) = (gray.width() as f32, gray.height() as f32);
            let center = (w * 0.5, h * 0.5);
            let (distance_px, class_label, offset_px, norm_offset, translation_m) = if let Some(c) = undistorted_corners_list.first() {
                let cx = ((c[0].0 + c[1].0 + c[2].0 + c[3].0) as f32) / 4.0;
                let cy = ((c[0].1 + c[1].1 + c[2].1 + c[3].1) as f32) / 4.0;
                let dx = cx - center.0;
                let dy = cy - center.1;
                let dist = (dx * dx + dy * dy).sqrt();
                let diag = (w * w + h * h).sqrt();
                let norm = dist / diag;
                let cls = if norm < 0.05 { "centered" } else if norm < 0.20 { "near" } else { "far" };
                let offs = [dx, dy];
                let n_offs = [dx / w, dy / h];

                let trans3d = estimate_translation_3d([
                    (c[0].0, c[0].1),
                    (c[1].0, c[1].1),
                    (c[2].0, c[2].1),
                    (c[3].0, c[3].1),
                ], cam_config, w as u32, h as u32);

                (dist, cls, offs, n_offs, trans3d)
            } else {
                (0.0, "no-tag", [0.0, 0.0], [0.0, 0.0], None)
            };

            if let Some(t) = translation_m {
                println!("Frame {}: Translation: [x={:.4}, y={:.4}, z={:.4}]", frame_idx, t[0], t[1], t[2]);
            } else if frame_idx % 30 == 0 {
                println!("Frame {}: No tag detected", frame_idx);
            }
        }
    }
}

fn yuyv_to_rgb(width: usize, height: usize, data: &[u8]) -> RgbImage {
    let mut out = vec![0u8; width * height * 3];
    let mut i = 0usize;
    let mut o = 0usize;
    while i + 3 < data.len() && o + 5 < out.len() {
        let y0 = data[i] as f32;
        let u  = data[i + 1] as f32;
        let y1 = data[i + 2] as f32;
        let v  = data[i + 3] as f32;
        i += 4;
        let c0 = y0 - 16.0; let c1 = y1 - 16.0; let d = u - 128.0; let e = v - 128.0;
        let r0 = (1.164 * c0 + 1.596 * e).clamp(0.0, 255.0) as u8;
        let g0 = (1.164 * c0 - 0.392 * d - 0.813 * e).clamp(0.0, 255.0) as u8;
        let b0 = (1.164 * c0 + 2.017 * d).clamp(0.0, 255.0) as u8;
        let r1 = (1.164 * c1 + 1.596 * e).clamp(0.0, 255.0) as u8;
        let g1 = (1.164 * c1 - 0.392 * d - 0.813 * e).clamp(0.0, 255.0) as u8;
        let b1 = (1.164 * c1 + 2.017 * d).clamp(0.0, 255.0) as u8;
        out[o] = r0; out[o + 1] = g0; out[o + 2] = b0;
        out[o + 3] = r1; out[o + 4] = g1; out[o + 5] = b1;
        o += 6;
    }
    RgbImage::from_raw(width as u32, height as u32, out).unwrap()
}

fn process_input_dir(output_dir: &Path, detector: &mut AprilDetector, cam_config: CameraConfig) -> anyhow::Result<()> {
    let input_dir = std::env::var("INPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("input"));

    if !input_dir.exists() {
        return Err(anyhow::anyhow!(
            "input dir {:?} not found. Mount a folder to input or set INPUT_DIR",
            input_dir
        ));
    }

    println!("Processing images from {:?}", input_dir);

    for entry in std::fs::read_dir(&input_dir)? {
        let entry = match entry { Ok(e) => e, Err(_) => continue };
        let path = entry.path();
        if !path.is_file() { continue; }
        let ext_ok = path.extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg" | "png"))
            .unwrap_or(false);
        if !ext_ok { continue; }

        let img = match image::open(&path) { Ok(i) => i, Err(err) => { eprintln!("open {} failed: {}", path.display(), err); continue; } };
        let gray = img.to_luma8();

        let corners_list = match detector::detect_corners(detector, &gray) {
            Ok(c) => c,
            Err(err) => { eprintln!("detection error on {}: {}", path.display(), err); continue; }
        };

        let mut undistorted_corners_list = Vec::new();
        for c in &corners_list {
            undistorted_corners_list.push(undistort::undistort_points(c, &cam_config));
        }

        let (w, h) = (gray.width() as f32, gray.height() as f32);
        
        if let Some(c) = undistorted_corners_list.first() {
            let trans3d = estimate_translation_3d([
                (c[0].0, c[0].1),
                (c[1].0, c[1].1),
                (c[2].0, c[2].1),
                (c[3].0, c[3].1),
            ], cam_config, w as u32, h as u32);

            if let Some(t) = trans3d {
                 println!("{}: Translation: [x={:.4}, y={:.4}, z={:.4}]", path.file_name().unwrap().to_string_lossy(), t[0], t[1], t[2]);
            } else {
                 println!("{}: Tag detected but pose estimation failed", path.file_name().unwrap().to_string_lossy());
            }
        } else {
            println!("{}: No tag detected", path.file_name().unwrap().to_string_lossy());
        }
    }

    Ok(())
}

fn estimate_translation_3d(corners: [(f64, f64); 4], cam: CameraConfig, _w: u32, _h: u32) -> Option<[f32; 3]> {
    let s = 0.16;
    let world = [
        [-s * 0.5, -s * 0.5],
        [ s * 0.5, -s * 0.5],
        [ s * 0.5,  s * 0.5],
        [-s * 0.5,  s * 0.5],
    ];
    let img = [
        [corners[0].0, corners[0].1],
        [corners[1].0, corners[1].1],
        [corners[2].0, corners[2].1],
        [corners[3].0, corners[3].1],
    ];

    let mut a = na::DMatrix::<f64>::zeros(8, 9);
    for i in 0..4 {
        let x = world[i][0];
        let y = world[i][1];
        let u = img[i][0];
        let v = img[i][1];
        a[(2*i, 0)] = -x; a[(2*i, 1)] = -y; a[(2*i, 2)] = -1.0;
        a[(2*i, 6)] = u * x; a[(2*i, 7)] = u * y; a[(2*i, 8)] = u;
        a[(2*i+1, 3)] = -x; a[(2*i+1, 4)] = -y; a[(2*i+1, 5)] = -1.0;
        a[(2*i+1, 6)] = v * x; a[(2*i+1, 7)] = v * y; a[(2*i+1, 8)] = v;
    }

    let svd = na::linalg::SVD::new(a.clone(), true, true);
    let vt = svd.v_t?;
    let h_vec = vt.row(vt.nrows()-1).clone_owned();
    let h = na::Matrix3::<f64>::from_row_slice(h_vec.as_slice());

    let k = na::Matrix3::new(cam.fx, 0.0, cam.cx, 0.0, cam.fy, cam.cy, 0.0, 0.0, 1.0);
    let k_inv = k.try_inverse()?;

    let hn = k_inv * h;
    let h1 = hn.column(0).into_owned();
    let h2 = hn.column(1).into_owned();
    let h3 = hn.column(2).into_owned();
    let norm1 = h1.norm();
    let norm2 = h2.norm();
    let denom = norm1 + norm2;
    if denom == 0.0 { return None; }
    let lambda = 2.0 / denom;

    let r1 = (h1 * lambda).into_owned();
    let r2 = (h2 * lambda).into_owned();
    let r3 = r1.cross(&r2);
    let r = na::Matrix3::<f64>::from_columns(&[r1, r2, r3]);
    let svd_r = na::linalg::SVD::new(r.clone(), true, true);
    if let (Some(u_r), Some(v_r_t)) = (svd_r.u, svd_r.v_t) {
        let _r = u_r * v_r_t;
    }

    let mut t = (h3 * lambda).into_owned();
    if t[2] < 0.0 {
        t = -t;
    }

    Some([t[0] as f32, t[1] as f32, t[2] as f32])
}
