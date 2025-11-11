mod detector;
mod overlay;
mod preprocess;

use image::io::Reader as ImageReader;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> anyhow::Result<()> {
    // Minimal positional arg parsing: <INPUT_DIR> <OUTPUT_DIR>
    let mut args = env::args_os();
    let _prog = args.next();
    let input_arg = args.next().ok_or_else(|| anyhow::anyhow!("Usage: <prog> <INPUT_DIR> <OUTPUT_DIR>"))?;
    let output_arg = args.next().ok_or_else(|| anyhow::anyhow!("Usage: <prog> <INPUT_DIR> <OUTPUT_DIR>"))?;
    let input_dir: PathBuf = input_arg.into();
    let output_dir: PathBuf = output_arg.into();

    let mut detector = detector::build_detector()?;

    fs::create_dir_all(&output_dir)?;

    for entry in fs::read_dir(&input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() { continue; }
        let ext = path.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase());

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

        // render overlay
        let mut overlay_img = image::DynamicImage::ImageLuma8(gray.clone()).to_rgba8();
        overlay::draw_detections(&mut overlay_img, &corners_list);
        println!("detections: {}", corners_list.len());

        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        let out_name = format!("{}-tags.png", stem);
        let out_path = Path::new(&output_dir).join(out_name);
        if let Err(err) = overlay_img.save(&out_path) {
            eprintln!("failed to save overlay {}: {}", out_path.display(), err);
            continue;
        }
        println!("wrote overlay: {}", out_path.display());
    }

    Ok(())
}