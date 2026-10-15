//! Fixed saved-image detection/pose check, independent of device camera calibration.
use crate::{config::CameraConfig, detector::{CpuDetector, Detection}};
use nalgebra::Vector3;
use std::path::Path;

pub fn run(path: &Path) -> anyhow::Result<serde_json::Value> {
    let image = image::open(path)?.to_luma8();
    anyhow::ensure!(image.width() == 1920 && image.height() == 1080, "Unexpected smoke image dimensions");
    let camera = CameraConfig { fx: 590.2919154759219, fy: 591.8160873557966,
        cx: 960.2348193847832, cy: 472.7595158081418, k1: 0., k2: 0., k3: 0., p1: 0., p2: 0.,
        tag_size_m: 0.16, x_offset: 0., y_offset: 0., z_offset: 0., roll_deg: 0., pitch_deg: 0., yaw_deg: 0. };
    let mut detector = CpuDetector::new(1)?;
    let detections = detector.detect(image.as_raw(), image.width() as usize, image.height() as usize)?;
    let tag = detections.iter().find_map(|d| match d { Detection::AprilTag(t) if t.id == 1 => Some(t), _ => None })
        .ok_or_else(|| anyhow::anyhow!("Smoke test missed tag 1"))?;
    let best = tag.cpu_pose_candidates(&camera, true, 50).into_iter()
        .min_by(|a, b| a.reprojection_rmse_px.total_cmp(&b.reprojection_rmse_px))
        .ok_or_else(|| anyhow::anyhow!("Smoke test returned no pose"))?;
    let position = -best.pose.rotation.transpose() * best.pose.translation;
    let error = (position - Vector3::new(0., 0., -2.)).norm();
    anyhow::ensure!(error.is_finite() && error < 0.20 && best.reprojection_rmse_px <= 2., "Smoke pose failed: {error} m");
    Ok(serde_json::json!({"schema_version": 1, "passed": true, "tag_id": 1,
        "camera_error_m": error, "reprojection_rmse_px": best.reprojection_rmse_px}))
}

#[cfg(test)]
mod tests {
    #[test]
    fn saved_image_exercises_production_cpu_detection_and_pose() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let image = [root.join("../reference/dim_noisy_tag.png"),
                     root.join("simulation_tests/reference/dim_noisy_tag.png")]
            .into_iter().find(|path| path.exists()).expect("Missing smoke fixture");
        assert_eq!(super::run(&image).unwrap()["passed"], true);
    }
}
