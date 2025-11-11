use anyhow::Result;
use apriltag::{DetectorBuilder, families::Family, Image as AprilImage, Detector};
use image::GrayImage;
use std::collections::HashSet;
use std::io::Write;
use tempfile::NamedTempFile;

pub fn build_detector() -> Result<Detector> {
    let family = Family::tag_36h11();
    let bits: usize = 3;
    Ok(
        DetectorBuilder::new()
            .add_family_bits(family, bits)
            .build()?
    )
}

pub fn detect_corners(detector: &mut Detector, gray: &GrayImage, scale_factor: f32) -> Result<Vec<[(i32, i32); 4]>> {
    let at_img: AprilImage = gray_to_april_image(gray)?;

    let gray_eq = crate::preprocess::contrast_stretch(gray);
    let at_img_eq: AprilImage = gray_to_april_image(&gray_eq)?;

    let sf = if scale_factor < 1.0 { 1.0 } else { scale_factor };
    let new_w = (gray_eq.width() as f32 * sf).round() as u32;
    let new_h = (gray_eq.height() as f32 * sf).round() as u32;
    let gray_scaled_eq = image::imageops::resize(&gray_eq, new_w, new_h, image::imageops::FilterType::Lanczos3);
    let at_img_scaled_eq: AprilImage = gray_to_april_image(&gray_scaled_eq)?;

    let mut detections = detector.detect(&at_img);
    let mut det_eq = detector.detect(&at_img_eq);
    let det_scaled_eq = detector.detect(&at_img_scaled_eq);
    detections.append(&mut det_eq);
    let mut ids: HashSet<usize> = detections.iter().map(|d| d.id()).collect();

    let mut corners_list: Vec<[(i32, i32); 4]> = Vec::new();
    for det in detections.iter() {
        let corners = det.corners();
        let to_i = |p: [f64; 2]| (p[0].round() as i32, p[1].round() as i32);
        corners_list.push([to_i(corners[0]), to_i(corners[1]), to_i(corners[2]), to_i(corners[3])]);
    }

    for det in det_scaled_eq.iter() {
        if ids.insert(det.id()) {
            let corners = det.corners();
            let to_i_scaled = |p: [f64; 2]| ((p[0] / sf as f64).round() as i32, (p[1] / sf as f64).round() as i32);
            corners_list.push([
                to_i_scaled(corners[0]),
                to_i_scaled(corners[1]),
                to_i_scaled(corners[2]),
                to_i_scaled(corners[3]),
            ]);
        }
    }

    Ok(corners_list)
}

fn gray_to_april_image(gray: &GrayImage) -> Result<AprilImage> {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let mut f = NamedTempFile::new()?;
    write!(f, "P5\n{} {}\n255\n", w, h)?;
    f.write_all(gray.as_raw())?;
    let path = f.into_temp_path();
    let img = AprilImage::from_pnm_file(path.to_string_lossy().as_ref())?;
    path.close()?;
    Ok(img)
}