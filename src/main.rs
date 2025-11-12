mod detector;
mod overlay;
mod preprocess;
mod config;

use nalgebra as na;
use serde::Serialize;

use image::io::Reader as ImageReader;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

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

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:?}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    // <INPUT_DIR> <OUTPUT_DIR> or defaults to /data/input and /data/output
    let mut args = env::args_os();
    let _prog = args.next();
    let input_dir: PathBuf = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/data/input"));
    let output_dir: PathBuf = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/data/output"));
    eprintln!("using input={} output={}", input_dir.display(), output_dir.display());

    let mut detector = detector::build_detector()?;

    let cam_params = config::camera_params();

    let mut results: Vec<ImageResult> = Vec::new();

    fs::create_dir_all(&output_dir)?;

    for entry in fs::read_dir(&input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() { continue; }
        if let Some(ext) = path.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()) {
            match ext.as_str() {
                "png" | "jpg" | "jpeg" | "bmp" | "tiff" => {}
                _ => { continue; }
            }
        }

        println!("=== {} ===", path.display());
        // load image, convert to grayscale
        let dyn_img = match ImageReader::open(&path)?.decode() {
            Ok(img) => img,
            Err(err) => { eprintln!("failed to decode {}: {}", path.display(), err); continue; }
        };
        let gray = dyn_img.to_luma8();

        // detect corners after preprocessing
        let corners_list = match detector::detect_corners(&mut detector, &gray, 1.5) {
            Ok(c) => c,
            Err(err) => { eprintln!("detection error on {}: {}", path.display(), err); continue; }
        };

        // render overlay on a darker background
        let gamma = config::DARKEN_GAMMA;
        let gray_dark = preprocess::gamma_correct(&gray, gamma);
        let mut overlay_img = image::DynamicImage::ImageLuma8(gray_dark).to_rgba8();
        overlay::draw_detections(&mut overlay_img, &corners_list);
        println!("detections: {}", corners_list.len());

        // estimate translational offset (distance from image center to first tag centroid)
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

            // estimate 3D translation from tag using camera intrinsics and tag size
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

        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        // name output primarily by estimated distance (pixels), include stem for uniqueness
        let out_name = if class_label == "no-tag" {
            format!("no-tag-{}.png", stem)
        } else {
            format!("{:.0}px-{}.png", distance_px, stem)
        };
        let out_path = Path::new(&output_dir).join(out_name);
        if let Err(err) = overlay_img.save(&out_path) {
            eprintln!("failed to save overlay {}: {}", out_path.display(), err);
            continue;
        }
        println!("wrote overlay: {} (class: {})", out_path.display(), class_label);

        results.push(ImageResult {
            image: path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string(),
            width: w as u32,
            height: h as u32,
            num_detections: corners_list.len(),
            class: class_label.to_string(),
            offset_px,
            norm_offset,
            distance_px,
            translation_m,
        });
    }

    let results_path = Path::new(&output_dir).join("results.json");
    if let Err(err) = std::fs::write(&results_path, serde_json::to_string_pretty(&results)? ) {
        eprintln!("failed to write results.json: {}", err);
    } else {
        println!("wrote results: {}", results_path.display());
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
    let (u_r, v_r_t) = (svd_r.u?, svd_r.v_t?);
    r = u_r * v_r_t; // closest rotation matrix

    let mut t = (h3 * lambda).into_owned();
    // ensure positive Z
    if t[2] < 0.0 {
        t = -t;
    }

    Some([t[0] as f32, t[1] as f32, t[2] as f32])
}