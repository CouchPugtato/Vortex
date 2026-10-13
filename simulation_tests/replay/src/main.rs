#![allow(dead_code)]
// Compile production detector and pose modules directly, without Linux camera/GPU dependencies.
#[path = "../../../src/config.rs"]
mod config;
#[path = "../../../src/detector.rs"]
mod detector;
#[path = "../../../src/pose.rs"]
mod pose;

use anyhow::{bail, Context, Result};
use nalgebra::{Matrix3, Vector3};
use serde::Deserialize;
use serde_json::json;
use std::{collections::HashMap, fs, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Manifest {
    schema_version: u8,
    width: u32,
    height: u32,
    camera: config::CameraConfig,
    tags: Vec<Tag>,
    frames: Vec<Frame>,
}
#[derive(Deserialize)]
struct Tag {
    id: usize,
    position: [f64; 3],
    rotation: [[f64; 3]; 3],
}
#[derive(Deserialize)]
struct Frame {
    name: String,
    image: String,
    camera_position: [f64; 3],
    expected_ids: Vec<usize>,
}
fn matrix(rows: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_row_slice(&rows.into_iter().flatten().collect::<Vec<_>>())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(
        args.next()
            .context("Usage: vortex-synthetic-replay DATASET [MAX_CAMERA_ERROR_M=0.20]")?,
    );
    let max_error: f64 = args.next().unwrap_or_else(|| "0.20".into()).parse()?;
    if !max_error.is_finite() || max_error <= 0.0 {
        bail!("Threshold must be positive and finite");
    }
    let manifest: Manifest = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    if manifest.schema_version != 1 || manifest.frames.is_empty() {
        bail!("Unsupported or empty manifest");
    }
    let tags: HashMap<_, _> = manifest.tags.iter().map(|t| (t.id, t)).collect();
    let map_from_solver = Matrix3::new(0., 0., -1., 1., 0., 0., 0., -1., 0.);
    let mut detector = detector::CpuDetector::new(1)?;
    let mut reports = Vec::new();
    let mut failed = 0;
    for frame in &manifest.frames {
        let img = image::open(root.join(&frame.image))
            .with_context(|| frame.image.clone())?
            .to_luma8();
        if img.width() != manifest.width || img.height() != manifest.height {
            bail!("Incorrect dimensions: {}", frame.image);
        }
        let start = Instant::now();
        let detections =
            detector.detect(img.as_raw(), img.width() as usize, img.height() as usize)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut results = Vec::new();
        let mut found = Vec::new();
        let mut valid = Vec::new();
        let mut errors = Vec::new();
        for detection in detections {
            if let detector::Detection::AprilTag(tag_det) = detection {
                found.push(tag_det.id);
                if let Some(tag) = tags.get(&tag_det.id) {
                    let r_ft = matrix(tag.rotation) * map_from_solver;
                    // Invert tag->camera and compose with tag->field, exactly as production does.
                    let best = tag_det
                        .cpu_pose_candidates(&manifest.camera, true, 50)
                        .into_iter()
                        .filter(|p| p.reprojection_rmse_px <= 2.)
                        .filter_map(|p| {
                            let camera_field = Vector3::from(tag.position)
                                - r_ft * p.pose.rotation.transpose() * p.pose.translation;
                            let normal = matrix(tag.rotation).column(0).into_owned();
                            if (camera_field - Vector3::from(tag.position)).dot(&normal) <= 0. {
                                return None;
                            }
                            Some((p.reprojection_rmse_px, camera_field))
                        })
                        .min_by(|a, b| a.0.total_cmp(&b.0));
                    if let Some((rmse, xyz)) = best {
                        let error = (xyz - Vector3::from(frame.camera_position)).norm();
                        if frame.expected_ids.contains(&tag_det.id) {
                            errors.push(error);
                        }
                        valid.push(tag_det.id);
                        results.push(
                            json!({"id": tag_det.id, "camera_position": [xyz.x, xyz.y, xyz.z],
                            "camera_error_m": error, "reprojection_rmse_px": rmse}),
                        );
                    }
                }
            }
        }
        let missing: Vec<_> = frame
            .expected_ids
            .iter()
            .filter(|id| !valid.contains(id))
            .collect();
        let passed = if frame.expected_ids.is_empty() {
            found.is_empty()
        } else {
            missing.is_empty()
                && errors.iter().all(|e| e.is_finite() && *e <= max_error)
                && found.iter().all(|id| tags.contains_key(id))
        };
        if !passed {
            failed += 1;
        }
        println!(
            "{} {}: detected {:?}, missing {:?}",
            if passed { "PASS" } else { "FAIL" },
            frame.name,
            found,
            missing
        );
        reports.push(
            json!({"name": frame.name, "passed": passed, "detected_ids": found,
            "missing_or_invalid_ids": missing, "detection_ms": elapsed_ms, "poses": results}),
        );
    }
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&json!({
            "scope": "production CPU detection and single-tag camera field localization",
            "frames": reports, "failed": failed, "max_camera_error_m": max_error
        }))?,
    )?;
    if failed > 0 {
        bail!("{failed} replay cases failed; see report.json");
    }
    Ok(())
}
