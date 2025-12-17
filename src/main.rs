mod detector;
mod overlay;
mod preprocess;
mod config;

use nalgebra as na;
use serde::Serialize;
use apriltag::Detector as AprilDetector;

use std::env;
use std::fs;
use std::io::{Write as IoWrite, Cursor};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::thread;
use std::sync::Arc;
use parking_lot::RwLock;

use image::RgbImage;
use v4l::{Device, FourCC};
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::video::Capture;
use v4l::io::traits::CaptureStream;
use jpeg_decoder as jpeg;

use actix_web::{web, App, HttpServer, HttpResponse, Responder};
// use futures::StreamExt;

use crate::config::CameraParams;

#[derive(Serialize)]
struct ImageResult {
    image: String,
    width: u32,
    height: u32,
    num_detections: usize,
    class: String,
    offset_px: [f32; 2],
    norm_offset: [f32; 2],
    distance_px: f32,
    translation_m: Option<[f32; 3]>,
}

struct AppState {
    // Stores the latest processed frame as a JPEG
    last_frame: Arc<RwLock<Option<Vec<u8>>>>,
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // <CAM_INDEX?> <OUTPUT_DIR?>
    let mut args = env::args();
    let _prog = args.next();
    let first = args.next();
    let second = args.next();
    
    // Determine camera index and output dir
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

    // Auto-detect camera if default 0 fails (or if user didn't specify strict index, but here we assume user arg is strict if provided)
    // Actually, let's try to open the specified camera, if it fails and user didn't explicitly provide it (arg count check?), we search.
    // For simplicity: If open fails later, we'll try to find one.
    
    println!("Starting application...");
    fs::create_dir_all(&output_dir)?;

    let last_frame = Arc::new(RwLock::new(None));
    let capture_state = last_frame.clone();
    let output_dir_clone = output_dir.clone();

    // Spawn capture thread
    thread::spawn(move || {
        let mut cam_idx = camera_index;
        
        // If initial attempt fails, try to find a working camera
        if Device::new(cam_idx).is_err() {
            println!("Camera {} not found, searching for available cameras...", cam_idx);
            if let Some(idx) = find_available_camera() {
                println!("Found camera at index {}", idx);
                cam_idx = idx;
            } else {
                eprintln!("No cameras found. Falling back to INPUT_DIR processing if available (non-streaming).");
                // Fallback to input dir processing (one-off), but we want to stream.
                // We'll run process_input_dir in a loop or just once?
                // The original code ran process_input_dir once.
                // If we want to verify the stream, we might want to feed dummy frames or just exit.
                // For now, let's just try process_input_dir and then exit the thread.
                let mut detector = match detector::build_detector() {
                    Ok(d) => d,
                    Err(e) => { eprintln!("Failed to build detector: {}", e); return; }
                };
                let cam_params = config::camera_params();
                if let Err(e) = process_input_dir(&output_dir_clone, &mut detector, cam_params) {
                    eprintln!("Input dir processing failed: {}", e);
                }
                return;
            }
        }

        if let Err(e) = run_capture_loop(cam_idx, output_dir_clone, capture_state) {
            eprintln!("Capture loop error: {:?}", e);
        }
    });

    println!("Starting web server at http://0.0.0.0:8080");
    println!("Stream available at http://0.0.0.0:8080/stream");
    
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(AppState {
                last_frame: last_frame.clone(),
            }))
            .route("/", web::get().to(index))
            .route("/stream", web::get().to(stream))
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}

fn find_available_camera() -> Option<usize> {
    for i in 0..10 {
        if Device::new(i).is_ok() {
            return Some(i);
        }
    }
    None
}

async fn index() -> impl Responder {
    HttpResponse::Ok().content_type("text/html").body(
        r#"
        <!DOCTYPE html>
        <html>
        <head><title>Camera Stream</title></head>
        <body style="background: #111; color: #eee; text-align: center;">
            <h1>Camera Stream</h1>
            <img src="/stream" style="max-width: 100%; border: 2px solid #333;" />
        </body>
        </html>
        "#
    )
}

async fn stream(data: web::Data<AppState>) -> HttpResponse {
    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(Duration::from_millis(50)); // 20 FPS cap for stream
        loop {
            interval.tick().await;
            let frame = {
                let lock = data.last_frame.read();
                lock.clone()
            };
            if let Some(f) = frame {
                let msg = format!("--myboundary\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", f.len());
                yield Ok::<_, actix_web::Error>(web::Bytes::from(msg));
                yield Ok::<_, actix_web::Error>(web::Bytes::from(f));
                yield Ok::<_, actix_web::Error>(web::Bytes::from("\r\n"));
            } else {
                // If no frame is available yet, just wait for the next tick.
            }
        }
    };

    HttpResponse::Ok()
        .insert_header(("Content-Type", "multipart/x-mixed-replace; boundary=myboundary"))
        .insert_header(("Cache-Control", "no-cache, no-store, must-revalidate"))
        .insert_header(("Pragma", "no-cache"))
        .insert_header(("Expires", "0"))
        .streaming(stream)
}

fn run_capture_loop(mut camera_index: usize, output_dir: PathBuf, state: Arc<RwLock<Option<Vec<u8>>>>) -> anyhow::Result<()> {
    println!("using initial camera_index={} output={}", camera_index, output_dir.display());

    let mut detector = detector::build_detector()?;
    let cam_params = config::camera_params();
    let mut frame_idx: u64 = 0;

    loop {
        // Attempt to open camera
        let dev_result = Device::new(camera_index);
        let mut dev = match dev_result {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to open camera {}: {}. Retrying in 2s...", camera_index, e);
                thread::sleep(Duration::from_secs(2));
                
                // Try to find any available camera if the current one is failing
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
                        eprintln!("Device disconnected (os error 19). Reconnecting...");
                        break; // Break inner loop to trigger reconnection
                    }
                    // Adding a small delay to prevent busy loop on error
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
                // YUYV -> RGB
                let (w, h) = (fmt.width as usize, fmt.height as usize);
                Some(yuyv_to_rgb(w, h, &data))
            };

            let rgb = match rgb_opt { Some(img) => img, None => { continue; } };
            let gray = image::DynamicImage::ImageRgb8(rgb).to_luma8();

            // detect corners after preprocessing
            let corners_list = match detector::detect_corners(&mut detector, &gray, 1.5) {
                Ok(c) => c,
                Err(err) => { eprintln!("detection error on frame {}: {}", frame_idx, err); continue; }
            };

            // render overlay on a darker background
            let gamma = config::DARKEN_GAMMA;
            let gray_dark = preprocess::gamma_correct(&gray, gamma);
            let mut overlay_img = image::DynamicImage::ImageLuma8(gray_dark).to_rgba8();
            overlay::draw_detections(&mut overlay_img, &corners_list);
            
            if frame_idx % 30 == 0 {
                 println!("frame {}: detections: {}", frame_idx, corners_list.len());
            }

            // estimate translational offset
            let (w, h) = (gray.width() as f32, gray.height() as f32);
            let center = (w * 0.5, h * 0.5);
            let (distance_px, class_label, offset_px, norm_offset, translation_m) = if let Some(c) = corners_list.first() {
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

                // estimate 3D translation from tag
                let trans3d = estimate_translation_3d([
                    (c[0].0, c[0].1),
                    (c[1].0, c[1].1),
                    (c[2].0, c[2].1),
                    (c[3].0, c[3].1),
                ], cam_params, w as u32, h as u32);

                (dist, cls, offs, n_offs, trans3d)
            } else {
                (0.0, "no-tag", [0.0, 0.0], [0.0, 0.0], None)
            };

            // Update shared state for streaming
            // We encode the overlay image to JPEG
            // Resize for streaming to reduce bandwidth (e.g. 640 width)
            let mut jpeg_buf = Vec::new();
            let mut cursor = Cursor::new(&mut jpeg_buf);
            let dynamic_overlay = image::DynamicImage::ImageRgba8(overlay_img.clone());
            let resized = dynamic_overlay.resize(640, 400, image::imageops::FilterType::Nearest);
            
            if let Err(e) = resized.write_to(&mut cursor, image::ImageOutputFormat::Jpeg(70)) {
                 eprintln!("Failed to encode jpeg for stream: {}", e);
            } else {
                 *state.write() = Some(jpeg_buf);
            }

            // Save overlay with timestamp
            let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
            let out_name = if class_label == "no-tag" {
                format!("no-tag-frame-{}-{}.png", frame_idx, ts)
            } else {
                format!("{:.0}px-frame-{}-{}.png", distance_px, frame_idx, ts)
            };
            let out_path = Path::new(&output_dir).join(out_name);
            
            if let Err(err) = overlay_img.save(&out_path) {
                eprintln!("failed to save overlay {}: {}", out_path.display(), err);
            }

            // Append a JSON line per frame
            let frame_result = ImageResult {
                image: out_path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string(),
                width: w as u32,
                height: h as u32,
                num_detections: corners_list.len(),
                class: class_label.to_string(),
                offset_px,
                norm_offset,
                distance_px,
                translation_m,
            };
            let json_line = format!("{}\n", serde_json::to_string(&frame_result).unwrap_or_default());
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(Path::new(&output_dir).join("results_stream.jsonl")) 
            {
                 let _ = f.write_all(json_line.as_bytes());
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

fn process_input_dir(output_dir: &Path, detector: &mut AprilDetector, cam_params: CameraParams) -> anyhow::Result<()> {
    // Default input dir is /data/input; can be overridden via INPUT_DIR env.
    let input_dir = std::env::var("INPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("input"));

    if !input_dir.exists() {
        return Err(anyhow::anyhow!(
            "input dir {:?} not found. Mount a folder to input or set INPUT_DIR",
            input_dir
        ));
    }

    let mut results_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(Path::new(&output_dir).join("results_stream.jsonl"))?;

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

        let corners_list = match detector::detect_corners(detector, &gray, 1.5) {
            Ok(c) => c,
            Err(err) => { eprintln!("detection error on {}: {}", path.display(), err); continue; }
        };

        let gamma = config::DARKEN_GAMMA;
        let gray_dark = preprocess::gamma_correct(&gray, gamma);
        let mut overlay_img = image::DynamicImage::ImageLuma8(gray_dark).to_rgba8();
        overlay::draw_detections(&mut overlay_img, &corners_list);

        let (w, h) = (gray.width() as f32, gray.height() as f32);
        let center = (w * 0.5, h * 0.5);
        let (distance_px, class_label, offset_px, norm_offset, translation_m) = if let Some(c) = corners_list.first() {
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
            ], cam_params, w as u32, h as u32);

            (dist, cls, offs, n_offs, trans3d)
        } else {
            (0.0, "no-tag", [0.0, 0.0], [0.0, 0.0], None)
        };

        let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let base = path.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
        let out_name = if class_label == "no-tag" {
            format!("no-tag-{}-{}.png", base, ts)
        } else {
            format!("{:.0}px-{}-{}.png", distance_px, base, ts)
        };
        let out_path = Path::new(&output_dir).join(out_name);
        if let Err(err) = overlay_img.save(&out_path) {
            eprintln!("failed to save overlay {}: {}", out_path.display(), err);
            continue;
        }

        let frame_result = ImageResult {
            image: out_path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string(),
            width: w as u32,
            height: h as u32,
            num_detections: corners_list.len(),
            class: class_label.to_string(),
            offset_px,
            norm_offset,
            distance_px,
            translation_m,
        };
        let json_line = format!("{}\n", serde_json::to_string(&frame_result)?);
        results_file.write_all(json_line.as_bytes())?;
        println!("processed {} -> {} (class: {})", path.display(), out_path.display(), class_label);
    }

    Ok(())
}

fn estimate_translation_3d(corners: [(f64, f64); 4], cam: CameraParams, _w: u32, _h: u32) -> Option<[f32; 3]> {
    // world/tag plane points (meters), matching corner order
    let s = cam.tag_size_m;
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

    // build DLT matrix A (8x9)
    let mut a = na::DMatrix::<f64>::zeros(8, 9);
    for i in 0..4 {
        let x = world[i][0];
        let y = world[i][1];
        let u = img[i][0];
        let v = img[i][1];
        // Row 2*i
        a[(2*i, 0)] = -x; a[(2*i, 1)] = -y; a[(2*i, 2)] = -1.0;
        a[(2*i, 6)] = u * x; a[(2*i, 7)] = u * y; a[(2*i, 8)] = u;
        // Row 2*i+1
        a[(2*i+1, 3)] = -x; a[(2*i+1, 4)] = -y; a[(2*i+1, 5)] = -1.0;
        a[(2*i+1, 6)] = v * x; a[(2*i+1, 7)] = v * y; a[(2*i+1, 8)] = v;
    }

    let svd = na::linalg::SVD::new(a.clone(), true, true);
    let vt = svd.v_t?; // 9x9
    let h_vec = vt.row(vt.nrows()-1).clone_owned(); // 1x9
    let h = na::Matrix3::<f64>::from_row_slice(h_vec.as_slice());

    // camera matrix K and inverse
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
    let lambda = 2.0 / denom; // average scaling from r1 and r2

    let r1 = (h1 * lambda).into_owned();
    let r2 = (h2 * lambda).into_owned();
    let r3 = r1.cross(&r2);
    let mut r = na::Matrix3::<f64>::from_columns(&[r1, r2, r3]);
    // orthogonalize R via SVD
    let svd_r = na::linalg::SVD::new(r.clone(), true, true);
    if let (Some(u_r), Some(v_r_t)) = (svd_r.u, svd_r.v_t) {
        let _r = u_r * v_r_t; // closest rotation matrix
    }

    let mut t = (h3 * lambda).into_owned();
    // ensure positive Z
    if t[2] < 0.0 {
        t = -t;
    }

    Some([t[0] as f32, t[1] as f32, t[2] as f32])
}