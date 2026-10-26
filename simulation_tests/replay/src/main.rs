#![allow(dead_code)]
// Compile production detector and pose modules directly, without Linux camera/GPU dependencies.
#[path = "../../../src/config.rs"]
mod config;
#[path = "../../../src/deployment_smoke.rs"]
mod deployment_smoke;
#[path = "../../../src/detector.rs"]
mod detector;
#[path = "../../../src/pose.rs"]
mod pose;
#[path = "../../../src/runtime_diagnostics.rs"]
mod runtime_diagnostics;
#[path = "../../../src/runtime_health.rs"]
mod runtime_health;
#[path = "../../../src/tag_refine.rs"]
mod tag_refine;

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
    #[serde(default)]
    robot_mount: Option<pose::RobotMount>,
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
    #[serde(default)]
    allowed_ids: Option<Vec<usize>>,
}
fn matrix(rows: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_row_slice(&rows.into_iter().flatten().collect::<Vec<_>>())
}
fn rotation_rows(r: &Matrix3<f64>) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| r[(i, j)]))
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--deployment-smoke") {
        let path = std::env::args()
            .nth(2)
            .ok_or_else(|| anyhow::anyhow!("Missing smoke image"))?;
        println!(
            "VORTEX_SMOKE {}",
            deployment_smoke::run(std::path::Path::new(&path))?
        );
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("--stream") {
        return stream();
    }
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
        let report = evaluate_frame(&manifest, frame, &img, &mut detector, max_error)?;
        if report["passed"] != true {
            failed += 1;
        }
        println!(
            "{} {}: detected {}",
            if report["passed"] == true {
                "PASS"
            } else {
                "FAIL"
            },
            frame.name,
            report["detected_ids"]
        );
        reports.push(report);
    }
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&json!({
            "scope": "production CPU detection and multi-tag camera field localization with single-tag fallback",
            "frames": reports, "failed": failed, "max_camera_error_m": max_error
        }))?,
    )?;
    if failed > 0 {
        bail!("{failed} replay cases failed; see report.json");
    }
    Ok(())
}

fn evaluate_frame(
    manifest: &Manifest,
    frame: &Frame,
    img: &image::GrayImage,
    detector: &mut detector::CpuDetector,
    max_error: f64,
) -> Result<serde_json::Value> {
    let tags: HashMap<_, _> = manifest.tags.iter().map(|t| (t.id, t)).collect();
    let map_from_solver = Matrix3::new(0., 0., -1., 1., 0., 0., 0., -1., 0.);
    let start = Instant::now();
    let detections = detector.detect(img.as_raw(), img.width() as usize, img.height() as usize)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
    let mut results = Vec::new();
    let mut found = Vec::new();
    let mut valid = Vec::new();
    let mut errors = Vec::new();
    let mut field_points = Vec::new();
    let mut image_points = Vec::new();
    let mut field_seeds = Vec::new();
    let mut detection_quality = Vec::new();
    let mut pose_candidate_diagnostics = Vec::new();
    let mut field_patches = Vec::new();
    let mut image_patch_ids = Vec::new();
    for detection in detections {
        if let detector::Detection::AprilTag(mut tag_det) = detection {
            found.push(tag_det.id);
            detection_quality.push(json!({"id": tag_det.id,
                    "hamming": tag_det.cpu_detection.as_ref().map(|d| d.hamming()),
                    "decision_margin": tag_det.cpu_detection.as_ref().map(|d| d.decision_margin()),
                    "corners": tag_det.corners}));
            if let Some(tag) = tags.get(&tag_det.id) {
                let r_ft = matrix(tag.rotation) * map_from_solver;
                // Invert tag->camera and compose with tag->field, exactly as production does.
                let estimates = if manifest.robot_mount.is_some() {
                    tag_det.cpu_corner_pose_candidates(&manifest.camera, true, 50)
                } else {
                    tag_det.cpu_pose_candidates(&manifest.camera, true, 50)
                };
                let best = estimates
                        .into_iter()
                        .filter(|p| p.reprojection_rmse_px <= 2.)
                        .filter_map(|p| {
                            let camera_field = Vector3::from(tag.position)
                                - r_ft * p.pose.rotation.transpose() * p.pose.translation;
                            pose_candidate_diagnostics.push(json!({"id": tag_det.id,
                                "reprojection_rmse_px": p.reprojection_rmse_px,
                                "camera_position": [camera_field.x, camera_field.y, camera_field.z],
                                "camera_rotation": rotation_rows(&(r_ft * p.pose.rotation.transpose())),
                                "camera_error_m": (camera_field - Vector3::from(frame.camera_position)).norm()}));
                            let normal = matrix(tag.rotation).column(0).into_owned();
                            if (camera_field - Vector3::from(tag.position)).dot(&normal) <= 0. {
                                return None;
                            }
                            let r_cf = p.pose.rotation * r_ft.transpose();
                            field_seeds.push(pose::Pose {
                                rotation: r_cf,
                                translation: p.pose.translation
                                    - r_cf * Vector3::from(tag.position),
                            });
                            Some((p.reprojection_rmse_px, camera_field, r_cf.transpose()))
                        })
                        .min_by(|a, b| a.0.total_cmp(&b.0));
                if let Some((rmse, xyz, r_fc)) = best {
                    let error = (xyz - Vector3::from(frame.camera_position)).norm();
                    if frame.expected_ids.contains(&tag_det.id) {
                        errors.push(error);
                    }
                    valid.push(tag_det.id);
                    let s = manifest.camera.tag_size_m / 2.;
                    for (local, pixel) in [
                        Vector3::new(-s, s, 0.),
                        Vector3::new(s, s, 0.),
                        Vector3::new(s, -s, 0.),
                        Vector3::new(-s, -s, 0.),
                    ]
                    .iter()
                    .zip(tag_det.corners.iter())
                    {
                        field_points.push(Vector3::from(tag.position) + r_ft * local);
                        image_points.push((pixel[0], pixel[1]));
                    }
                    results.push(
                        json!({"id": tag_det.id, "camera_position": [xyz.x, xyz.y, xyz.z],
                            "camera_rotation": rotation_rows(&r_fc),
                            "camera_error_m": error, "reprojection_rmse_px": rmse}),
                    );
                }
                if let Some(patch) = tag_det.image_patch.take() {
                    image_patch_ids.push(tag_det.id);
                    field_patches.push((
                        patch,
                        pose::Pose {
                            rotation: r_ft,
                            translation: Vector3::from(tag.position),
                        },
                    ));
                }
            }
        }
    }
    let missing: Vec<_> = frame
        .expected_ids
        .iter()
        .filter(|id| !valid.contains(id))
        .collect();
    let raw_joint = if let Some(mount) = &manifest.robot_mount {
        let candidates = pose::planar_pose_candidates(
            &field_points,
            &image_points,
            &manifest.camera,
            true,
            mount,
            &field_seeds,
        );
        let patches: Vec<_> = field_patches.iter().map(|(p, t)| (p, t.clone())).collect();
        tag_refine::refine_planar_field_pose(
            &patches,
            &candidates,
            &manifest.camera,
            mount,
            &field_points,
            &image_points,
        )
        .or_else(|| candidates.into_iter().next())
    } else if field_points.len() >= 8 {
        let patches: Vec<_> = field_patches.iter().map(|(p, t)| (p, t.clone())).collect();
        tag_refine::estimate_field_pose(
            &patches,
            &manifest.camera,
            &field_points,
            &image_points,
            &field_seeds,
        )
    } else {
        None
    };
    let joint = raw_joint
        .as_ref()
        .filter(|p| p.reprojection_rmse_px <= 2.)
        .map(|p| {
            let xyz = -p.pose.rotation.transpose() * p.pose.translation;
            let error = (xyz - Vector3::from(frame.camera_position)).norm();
            (
                error,
                json!({"camera_position": [xyz.x, xyz.y, xyz.z],
                        "camera_rotation": rotation_rows(&p.pose.rotation.transpose()),
                        "camera_error_m": error, "reprojection_rmse_px": p.reprojection_rmse_px,
                        "tags_used": field_points.len() / 4}),
            )
        });
    let planar_rejected = manifest.robot_mount.is_some() && joint.is_none();
    if planar_rejected {
        results.clear();
    }
    let position_ok = if let Some((error, _)) = &joint {
        error.is_finite() && *error <= max_error
    } else {
        errors.iter().all(|e| e.is_finite() && *e <= max_error)
    };
    let passed = if frame.expected_ids.is_empty() {
        found.is_empty()
    } else {
        missing.is_empty()
            && position_ok
            && found.iter().all(|id| {
                frame
                    .allowed_ids
                    .as_ref()
                    .map_or_else(|| tags.contains_key(id), |ids| ids.contains(id))
            })
    };
    Ok(
        json!({"name": frame.name, "passed": passed, "detected_ids": found,
            "image_patch_ids": image_patch_ids,
            "planar_rejected": planar_rejected,
            "detection_quality": detection_quality,
            "pose_candidates": pose_candidate_diagnostics,
            "missing_or_invalid_ids": missing, "detection_ms": elapsed_ms, "poses": results,
            "localization_ms": start.elapsed().as_secs_f64()*1000.0-elapsed_ms,
            "unexpected_ids": found.iter().filter(|id| frame.allowed_ids.as_ref()
                .map_or_else(|| !tags.contains_key(id), |ids| !ids.contains(id))).collect::<Vec<_>>(),
            "position_passed": position_ok,
            "localization_mode": if joint.is_some() && manifest.robot_mount.is_some() { "planar" }
                else if joint.is_some() && field_points.len()>=8 { "multi_tag" } else { "single_tag" },
            "joint_reprojection_rmse_px": raw_joint.map(|p| p.reprojection_rmse_px),
            "single_tag_diagnostic_passed": missing.is_empty()
                && errors.iter().all(|e| e.is_finite() && *e <= max_error),
            "joint_pose": joint.map(|(_, value)| value)}),
    )
}

// One JSON initialization line, then fixed-size grayscale frames. No image files or
// detector restarts between frames; diagnostic native stdout is ignored by the client.
fn stream() -> Result<()> {
    use std::io::{BufRead, Read, Write};
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut header = String::new();
    input.read_line(&mut header)?;
    let manifest: Manifest = serde_json::from_str(&header)?;
    if manifest.schema_version != 1 || manifest.width < 2 || manifest.height < 2 {
        bail!("Invalid stream configuration");
    }
    let size = (manifest.width as usize)
        .checked_mul(manifest.height as usize)
        .filter(|s| *s <= 32 * 1024 * 1024)
        .context("Frame size too large")?;
    let mut detector = detector::CpuDetector::new(1)?;
    let frame = Frame {
        name: "live".into(),
        image: String::new(),
        camera_position: [0.; 3],
        expected_ids: vec![],
        allowed_ids: None,
    };
    loop {
        let mut pixels = vec![0u8; size];
        // EOF before the first byte is a clean shutdown; a partial frame is an error.
        if input.read(&mut pixels[..1])? == 0 {
            break;
        }
        input.read_exact(&mut pixels[1..])?;
        let img = image::GrayImage::from_raw(manifest.width, manifest.height, pixels)
            .context("Invalid grayscale frame")?;
        let mut report = evaluate_frame(&manifest, &frame, &img, &mut detector, 0.20)?;
        // Live streaming has no ground truth. Only the dataset mode scores poses.
        if let Some(object) = report.as_object_mut() {
            for key in ["passed", "position_passed", "single_tag_diagnostic_passed"] {
                object.remove(key);
            }
            for key in ["poses", "pose_candidates"] {
                if let Some(values) = object.get_mut(key).and_then(|v| v.as_array_mut()) {
                    for value in values {
                        if let Some(pose) = value.as_object_mut() {
                            pose.remove("camera_error_m");
                        }
                    }
                }
            }
            if let Some(pose) = object.get_mut("joint_pose").and_then(|v| v.as_object_mut()) {
                pose.remove("camera_error_m");
            }
        }
        println!("VORTEX_FRAME {}", serde_json::to_string(&report)?);
        std::io::stdout().flush()?;
    }
    Ok(())
}
