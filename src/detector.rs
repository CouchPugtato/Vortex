use anyhow::Result;
use apriltag::{
    families::Family, pose::TagParams, Detection as AprilRawDetection, Detector, DetectorBuilder,
    Image as AprilImage,
};
use nalgebra::{Matrix3, Vector3};

use crate::config::CameraConfig;
use crate::pose::{self, Pose, PoseEstimate};

#[repr(C)]
struct RawDetector {
    nthreads: i32,
    quad_decimate: f32,
    quad_sigma: f32,
    refine_edges: i32,
    decode_sharpening: f64,
    debug: i32,
}

#[derive(Debug)]
pub struct AprilTagDetection {
    pub id: usize,
    pub center: [f64; 2],
    pub corners: [[f64; 2]; 4],
    pub(crate) cpu_detection: Option<AprilRawDetection>,
    pub(crate) cpu_secondary_detection: Option<AprilRawDetection>,
    pub(crate) image_patch: Option<crate::tag_refine::TagPatch>,
}

impl AprilTagDetection {
    pub fn cpu_pose_candidates(
        &self,
        camera: &CameraConfig,
        apply_distortion: bool,
        n_iters: usize,
    ) -> Vec<PoseEstimate> {
        let Some(det) = self.cpu_detection.as_ref() else {
            return Vec::new();
        };
        let params = TagParams {
            tagsize: camera.tag_size_m,
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
        };
        let s = camera.tag_size_m / 2.0;
        let object_points = [
            Vector3::new(-s, s, 0.0),
            Vector3::new(s, s, 0.0),
            Vector3::new(s, -s, 0.0),
            Vector3::new(-s, -s, 0.0),
        ];
        let corners = solver_corners(self.corners);
        let raw: Vec<_> = std::iter::once(det)
            .chain(self.cpu_secondary_detection.as_ref())
            .flat_map(|source| source.estimate_tag_pose_orthogonal_iteration(&params, n_iters))
            .filter_map(|candidate| {
                let rot = candidate.pose.rotation();
                let t = candidate.pose.translation();
                let r_data = rot.data();
                let t_data = t.data();
                if r_data.len() != 9 || t_data.len() < 3 {
                    return None;
                }
                let pose = Pose {
                    rotation: Matrix3::from_row_slice(r_data),
                    translation: Vector3::new(t_data[0], t_data[1], t_data[2]),
                };
                let reprojection_rmse_px =
                    pose::reprojection_rmse_px_for_pose(&pose, &corners, camera, apply_distortion)?;
                Some(PoseEstimate {
                    pose,
                    reprojection_rmse_px,
                })
            })
            .collect();
        // Keep both original orientation hypotheses. Corner-only refinement can
        // pull two starts into the same minimum before image alignment sees them.
        let mut estimates = raw.clone();
        for candidate in &raw {
            if let Some(estimate) = pose::estimate_pose_from_correspondences(
                &object_points,
                &corners,
                camera,
                apply_distortion,
                &candidate.pose,
            ) {
                estimates.push(estimate);
            }
        }
        if let Some(estimate) = pose::estimate_pose(&corners, camera, apply_distortion) {
            estimates.push(estimate);
        }
        if let Some(patch) = &self.image_patch {
            if let Some(estimate) = patch.refine(&estimates, camera, &corners, apply_distortion) {
                return vec![estimate];
            }
        }
        estimates
    }

    pub fn drop_cpu_pose_source(&mut self) {
        self.cpu_detection = None;
        self.cpu_secondary_detection = None;
        self.image_patch = None;
    }
}

#[derive(Debug, Clone)]
pub struct YoloDetection {
    pub class_name: String,
    pub confidence: f64,
    // [x, y, w, h]
    pub bbox: [f64; 4],
}

#[derive(Debug)]
pub enum Detection {
    AprilTag(AprilTagDetection),
    Yolo(YoloDetection),
}

pub struct CpuDetector {
    inner: Detector,
    secondary: Detector,
}

impl CpuDetector {
    pub fn new(nthreads: i32) -> Result<Self> {
        let detector = build_inner_detector(nthreads, 1.0)?;
        let secondary = build_inner_detector(nthreads, 2.0)?;
        Ok(Self {
            inner: detector,
            secondary,
        })
    }

    pub fn detect(
        &mut self,
        gray_data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<Detection>> {
        detect_corners(
            &mut self.inner,
            &mut self.secondary,
            gray_data,
            width,
            height,
        )
    }
}

fn build_inner_detector(nthreads: i32, decimate: f32) -> Result<Detector> {
    let family = Family::tag_36h11();
    // Three-bit correction can turn low-contrast noise into a valid tag code.
    let bits: usize = 2;
    let detector = DetectorBuilder::new()
        .add_family_bits(family, bits)
        .build()?;

    unsafe {
        let ptr_ptr = &detector as *const Detector as *const *mut RawDetector;
        let raw_ptr = *ptr_ptr;

        if !raw_ptr.is_null() {
            (*raw_ptr).nthreads = nthreads;
            // Small, oblique field tags can lose both their border and code at
            // half resolution. Keep quad extraction at the input resolution.
            (*raw_ptr).quad_decimate = decimate;
            (*raw_ptr).quad_sigma = 0.0;
            (*raw_ptr).refine_edges = 1;
        }
    }

    Ok(detector)
}

fn detect_corners(
    detector: &mut Detector,
    secondary: &mut Detector,
    gray_data: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<Detection>> {
    anyhow::ensure!(
        width > 1 && height > 1 && width.checked_mul(height) == Some(gray_data.len()),
        "Invalid grayscale frame dimensions or buffer length"
    );
    let mut img = unsafe { AprilImage::new_uinit(width, height)? };
    let stride = img.stride();
    let dst: &mut _ = img.as_mut();
    dst.fill(0);
    for row in 0..height {
        dst[row * stride..row * stride + width]
            .copy_from_slice(&gray_data[row * width..(row + 1) * width]);
    }
    let detections: Vec<_> = detector
        .detect(&img)
        .into_iter()
        .filter(|det| det.decision_margin().is_finite() && det.decision_margin() >= 20.0)
        .collect();
    // Multiple tags constrain the pose geometrically; reserve image alignment
    // for isolated detections instead of repeating it for every field tag.
    let isolated = detections.len() == 1;
    // Different quad sampling can seed different planar pose minima. Retain a
    // matching half-resolution hypothesis only for small isolated tags; the
    // full-resolution detection still determines IDs, corners, and image fitting.
    let small_isolated = isolated
        && (detections[0].decision_margin() < 160.0 || {
            let c = detections[0].corners();
            (0..4)
                .map(|i| (c[i][0] - c[(i + 1) % 4][0]).hypot(c[i][1] - c[(i + 1) % 4][1]))
                .sum::<f64>()
                / 4.0
                <= 40.0
        });
    let mut secondary_detections = if small_isolated {
        secondary.detect(&img)
    } else {
        Vec::new()
    };

    let mut results: Vec<Detection> = Vec::new();
    for mut det in detections {
        let full_center = det.center();
        let mut cpu_secondary_detection = secondary_detections
            .iter()
            .position(|candidate| {
                let center = candidate.center();
                candidate.id() == det.id()
                    && candidate.decision_margin() >= 20.0
                    && (center[0] - full_center[0]).hypot(center[1] - full_center[1]) < 5.0
            })
            .map(|i| secondary_detections.swap_remove(i));
        // Weak edge contrast is more sensitive to pixel noise. If the coarser
        // detector independently agrees on this tag, retain its denoised quad
        // and original pose seeds rather than fitting the noisy fine-grid quad.
        if det.decision_margin() < 160.0 {
            if let Some(coarse) = cpu_secondary_detection.take() {
                det = coarse;
            }
        }
        let corners = det.corners();
        let initial_corners = [
            [corners[0][0], corners[0][1]],
            [corners[1][0], corners[1][1]],
            [corners[2][0], corners[2][1]],
            [corners[3][0], corners[3][1]],
        ];
        let c_arr = refine_border_corners(gray_data, width, height, initial_corners)
            .unwrap_or(initial_corners);

        let center = det.center();
        let center_arr = [center[0], center[1]];

        let image_patch = if isolated {
            crate::tag_refine::TagPatch::new(gray_data, width, height, c_arr)
        } else {
            None
        };
        results.push(Detection::AprilTag(AprilTagDetection {
            id: det.id(),
            center: center_arr,
            corners: c_arr,
            cpu_detection: Some(det),
            cpu_secondary_detection,
            image_patch,
        }));
    }

    Ok(results)
}

/// Fit the black/white border at subpixel precision using bilinear samples and
/// the signed gradient's first moment. Integer gradient sampling biases small
/// quads by a fraction of a pixel, which can flip planar pose solutions.
fn refine_border_corners(
    data: &[u8],
    width: usize,
    height: usize,
    corners: [[f64; 2]; 4],
) -> Option<[[f64; 2]; 4]> {
    let mut lines = [[0.0; 3]; 4];
    for edge in 0..4 {
        let a = corners[edge];
        let b = corners[(edge + 1) % 4];
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let length = dx.hypot(dy);
        if length < 8.0 {
            return None;
        }
        // Native BL/BR/TR/TL winding: this normal points toward the white margin.
        let normal = [-dy / length, dx / length];
        let count = (length as usize).clamp(16, 64);
        let mut points = Vec::new();
        for i in 0..count {
            let t = 0.15 + 0.7 * (i as f64 + 0.5) / count as f64;
            let base = [a[0] + t * dx, a[1] + t * dy];
            let mut previous = sample_gray(
                data,
                width,
                height,
                base[0] - 2.0 * normal[0],
                base[1] - 2.0 * normal[1],
            )?;
            let mut weight = 0.0;
            let mut moment = 0.0;
            for step in 1..=16 {
                let offset = -2.0 + step as f64 * 0.25;
                let value = sample_gray(
                    data,
                    width,
                    height,
                    base[0] + offset * normal[0],
                    base[1] + offset * normal[1],
                )?;
                let gradient = value - previous;
                weight += gradient;
                moment += gradient * (offset - 0.125);
                previous = value;
            }
            if weight < 60.0 {
                continue;
            }
            let offset = moment / weight;
            if !offset.is_finite() || offset.abs() > 1.75 {
                continue;
            }
            points.push([base[0] + offset * normal[0], base[1] + offset * normal[1]]);
        }
        if points.len() < 8 {
            return None;
        }
        let mean = [
            points.iter().map(|p| p[0]).sum::<f64>() / points.len() as f64,
            points.iter().map(|p| p[1]).sum::<f64>() / points.len() as f64,
        ];
        let mut xx = 0.0;
        let mut xy = 0.0;
        let mut yy = 0.0;
        for p in points {
            let x = p[0] - mean[0];
            let y = p[1] - mean[1];
            xx += x * x;
            xy += x * y;
            yy += y * y;
        }
        let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
        let n = [-angle.sin(), angle.cos()];
        lines[edge] = [n[0], n[1], n[0] * mean[0] + n[1] * mean[1]];
    }
    let mut refined = corners;
    for i in 0..4 {
        let a = lines[(i + 3) % 4];
        let b = lines[i];
        let determinant = a[0] * b[1] - a[1] * b[0];
        if determinant.abs() < 0.1 {
            return None;
        }
        let point = [
            (a[2] * b[1] - a[1] * b[2]) / determinant,
            (a[0] * b[2] - a[2] * b[0]) / determinant,
        ];
        if (point[0] - corners[i][0]).hypot(point[1] - corners[i][1]) > 2.5 {
            return None;
        }
        refined[i] = point;
    }
    Some(refined)
}

fn sample_gray(data: &[u8], width: usize, height: usize, x: f64, y: f64) -> Option<f64> {
    if x < 0.0 || y < 0.0 || x >= (width - 1) as f64 || y >= (height - 1) as f64 {
        return None;
    }
    let ix = x.floor() as usize;
    let iy = y.floor() as usize;
    let fx = x - ix as f64;
    let fy = y - iy as f64;
    let top = data[iy * width + ix] as f64 * (1.0 - fx) + data[iy * width + ix + 1] as f64 * fx;
    let bottom = data[(iy + 1) * width + ix] as f64 * (1.0 - fx)
        + data[(iy + 1) * width + ix + 1] as f64 * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

fn solver_corners(corners: [[f64; 2]; 4]) -> [(f64, f64); 4] {
    [
        (corners[0][0], corners[0][1]),
        (corners[1][0], corners[1][1]),
        (corners[2][0], corners[2][1]),
        (corners[3][0], corners[3][1]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_tags_in_frames_with_padded_native_rows() {
        let tag = image::load_from_memory(include_bytes!(
            "../simulation_tests/reference/tag36h11_1.png"
        ))
        .unwrap()
        .to_luma8();
        let tag = image::imageops::resize(&tag, 400, 400, image::imageops::FilterType::Nearest);
        let mut frame = image::GrayImage::from_pixel(641, 481, image::Luma([255]));
        image::imageops::overlay(&mut frame, &tag, 120, 40);
        let mut detector = CpuDetector::new(1).unwrap();
        let results = detector.detect(frame.as_raw(), 641, 481).unwrap();
        assert!(results
            .iter()
            .any(|d| matches!(d,Detection::AprilTag(tag) if tag.id==1)));
        assert!(detector.detect(&[0; 4], 641, 481).is_err());
    }

    #[test]
    fn rejects_low_contrast_even_when_the_tag_code_is_valid() {
        let tag = image::load_from_memory(include_bytes!(
            "../simulation_tests/reference/tag36h11_1.png"
        ))
        .unwrap()
        .to_luma8();
        let mut tag = image::imageops::resize(&tag, 400, 400, image::imageops::FilterType::Nearest);
        for pixel in tag.pixels_mut() {
            pixel[0] = if pixel[0] == 0 { 120 } else { 130 };
        }
        let mut detector = CpuDetector::new(1).unwrap();
        assert!(detector.detect(tag.as_raw(), 400, 400).unwrap().is_empty());
    }

    #[test]
    fn dim_noisy_tag_retains_stable_pose_after_full_resolution_detection() {
        let frame = image::load_from_memory(include_bytes!(
            "../simulation_tests/reference/dim_noisy_tag.png"
        ))
        .unwrap()
        .to_luma8();
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../config/config.json")).unwrap();
        let mut camera: CameraConfig = serde_json::from_value(config["camera"].clone()).unwrap();
        camera.fx = 590.2919154759219;
        camera.fy = 591.8160873557966;
        camera.cx = 960.2348193847832;
        camera.cy = 472.7595158081418;
        camera.tag_size_m = 0.16;
        camera.k1 = 0.;
        camera.k2 = 0.;
        camera.k3 = 0.;
        camera.p1 = 0.;
        camera.p2 = 0.;
        let mut detector = CpuDetector::new(1).unwrap();
        let detections = detector
            .detect(
                frame.as_raw(),
                frame.width() as usize,
                frame.height() as usize,
            )
            .unwrap();
        let tag = detections
            .iter()
            .find_map(|d| match d {
                Detection::AprilTag(tag) if tag.id == 1 => Some(tag),
                _ => None,
            })
            .expect("dim tag should still decode");
        let best = tag
            .cpu_pose_candidates(&camera, true, 50)
            .into_iter()
            .min_by(|a, b| a.reprojection_rmse_px.total_cmp(&b.reprojection_rmse_px))
            .unwrap();
        let camera_in_tag = -best.pose.rotation.transpose() * best.pose.translation;
        let error = (camera_in_tag - Vector3::new(0., 0., -2.)).norm();
        assert!(error < 0.20, "dim noisy tag camera error: {error}");
    }

    #[test]
    fn border_refinement_removes_fractional_pixel_bias() {
        let (width, height) = (120, 100);
        let mut data = vec![255u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let cover_x = ((x as f64 + 0.5).min(80.75) - (x as f64 - 0.5).max(30.25)).max(0.0);
                let cover_y = ((y as f64 + 0.5).min(70.80) - (y as f64 - 0.5).max(25.40)).max(0.0);
                data[y * width + x] = (255.0 * (1.0 - cover_x * cover_y)).round() as u8;
            }
        }
        let initial = [[30.8, 71.3], [81.3, 71.3], [81.3, 25.9], [30.8, 25.9]];
        let expected = [
            [30.25, 70.80],
            [80.75, 70.80],
            [80.75, 25.40],
            [30.25, 25.40],
        ];
        let corners = refine_border_corners(&data, width, height, initial).unwrap();
        for (point, truth) in corners.iter().zip(expected) {
            assert!((point[0] - truth[0]).hypot(point[1] - truth[1]) < 0.03);
        }
    }
}
