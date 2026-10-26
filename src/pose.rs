use nalgebra::{DMatrix, DVector, Matrix3, Point2, Rotation3, SymmetricEigen, Vector2, Vector3};

use crate::config::CameraConfig;

#[derive(Debug, Clone)]
pub struct Pose {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

#[derive(Debug, Clone)]
pub struct PoseEstimate {
    pub pose: Pose,
    pub reprojection_rmse_px: f64,
}

/// Calibrated optical-camera mount for explicitly planar chassis localization.
/// This is configuration, never a measured/ground-truth chassis position.
#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub struct RobotMount {
    pub x_offset: f64,
    pub y_offset: f64,
    pub z_offset: f64,
    pub roll_deg: f64,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
}

impl From<&CameraConfig> for RobotMount {
    fn from(camera: &CameraConfig) -> Self {
        Self {
            x_offset: camera.x_offset,
            y_offset: camera.y_offset,
            z_offset: camera.z_offset,
            roll_deg: camera.roll_deg,
            pitch_deg: camera.pitch_deg,
            yaw_deg: camera.yaw_deg,
        }
    }
}

impl RobotMount {
    pub(crate) fn rotation(self) -> Matrix3<f64> {
        (Rotation3::from_axis_angle(&Vector3::y_axis(), self.yaw_deg.to_radians())
            * Rotation3::from_axis_angle(&Vector3::x_axis(), self.pitch_deg.to_radians())
            * Rotation3::from_axis_angle(&Vector3::z_axis(), self.roll_deg.to_radians()))
        .into_inner()
    }

    pub(crate) fn offset(self) -> Vector3<f64> {
        Vector3::new(self.x_offset, self.y_offset, self.z_offset)
    }

    pub(crate) fn field_pose(self, params: &DVector<f64>) -> Pose {
        let yaw = Rotation3::from_axis_angle(&Vector3::z_axis(), params[2]).into_inner();
        let rotation = (yaw * self.rotation()).transpose();
        let position = Vector3::new(params[0], params[1], 0.) + yaw * self.offset();
        Pose {
            rotation,
            translation: -rotation * position,
        }
    }
}

/// Fit only chassis X/Y/yaw on a level floor, keeping the calibrated camera
/// height and attitude relative to the chassis fixed. No field truth is used.
/// Callers must opt in: this model is unsuitable for ramps or an unknown mount.
pub fn estimate_planar_pose(
    object_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
    distortion: bool,
    mount: &RobotMount,
    initial_poses: &[Pose],
) -> Option<PoseEstimate> {
    planar_pose_candidates(
        object_points,
        image_points,
        camera,
        distortion,
        mount,
        initial_poses,
    )
    .into_iter()
    .next()
}

pub fn planar_pose_candidates(
    object_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
    distortion: bool,
    mount: &RobotMount,
    initial_poses: &[Pose],
) -> Vec<PoseEstimate> {
    if object_points.len() != image_points.len()
        || object_points.len() < 4
        || object_points.len() % 4 != 0
        || ![
            mount.x_offset,
            mount.y_offset,
            mount.z_offset,
            mount.roll_deg,
            mount.pitch_deg,
            mount.yaw_deg,
            camera.fx,
            camera.fy,
        ]
        .iter()
        .all(|v| v.is_finite())
        || camera.fx <= 0.
        || camera.fy <= 0.
    {
        return Vec::new();
    }
    let mount_rotation = mount.rotation();
    let mut seeds = Vec::new();
    // Intersect observed rays with each mapped corner's known height, then
    // align those robot-relative points to the map with a 2D rigid transform.
    // This provides a seed even when free 6-DoF planar PnP picks its mirror.
    let mut pairs = Vec::new();
    for (point, pixel) in object_points.iter().zip(image_points) {
        let (u, v) = if distortion {
            undistort_normalized_point(*pixel, camera)
        } else {
            (
                (pixel.0 - camera.cx) / camera.fx,
                (pixel.1 - camera.cy) / camera.fy,
            )
        };
        let ray = mount_rotation * Vector3::new(u, v, 1.);
        if ray.z.abs() < 1e-5 {
            continue;
        }
        let depth = (point.z - mount.z_offset) / ray.z;
        if depth <= 0. || !depth.is_finite() {
            continue;
        }
        pairs.push((mount.offset() + ray * depth, *point));
    }
    if pairs.len() >= 2 {
        let a = pairs.iter().map(|p| p.0).sum::<Vector3<f64>>() / pairs.len() as f64;
        let b = pairs.iter().map(|p| p.1).sum::<Vector3<f64>>() / pairs.len() as f64;
        let (mut dot, mut cross) = (0., 0.);
        for (p, q) in &pairs {
            let (p, q) = (p - a, q - b);
            dot += p.x * q.x + p.y * q.y;
            cross += p.x * q.y - p.y * q.x;
        }
        if dot.hypot(cross) > 1e-10 {
            let yaw = cross.atan2(dot);
            let position = b - Rotation3::from_axis_angle(&Vector3::z_axis(), yaw) * a;
            seeds.push(DVector::from_vec(vec![position.x, position.y, yaw]));
        }
    }
    for initial in initial_poses.iter().take(8) {
        let r = initial.rotation.transpose() * mount_rotation.transpose();
        let yaw = r[(1, 0)].atan2(r[(0, 0)]);
        let p = -initial.rotation.transpose() * initial.translation
            - Rotation3::from_axis_angle(&Vector3::z_axis(), yaw) * mount.offset();
        seeds.push(DVector::from_vec(vec![p.x, p.y, yaw]));
    }
    // A small tag can send every unconstrained seed to the same mirrored
    // minimum. Cover heading globally; translation for each heading follows
    // from the observed bearing equations, not a presumed robot location.
    for index in 0..36 {
        let yaw = index as f64 * std::f64::consts::TAU / 36.;
        let rz = Rotation3::from_axis_angle(&Vector3::z_axis(), yaw).into_inner();
        let r = (rz * mount_rotation).transpose();
        let mut a = DMatrix::zeros(object_points.len() * 2, 2);
        let mut b = DVector::zeros(object_points.len() * 2);
        for (i, (point, pixel)) in object_points.iter().zip(image_points).enumerate() {
            let (u, v) = if distortion {
                undistort_normalized_point(*pixel, camera)
            } else {
                (
                    (pixel.0 - camera.cx) / camera.fx,
                    (pixel.1 - camera.cy) / camera.fy,
                )
            };
            for (j, value) in [u, v].into_iter().enumerate() {
                let normal = r.row(j) - r.row(2) * value;
                a[(i * 2 + j, 0)] = normal[0];
                a[(i * 2 + j, 1)] = normal[1];
                b[i * 2 + j] = normal.transpose().dot(point) - normal[2] * mount.z_offset;
            }
        }
        let at = a.transpose();
        if let Some(position) = (&at * &a).lu().solve(&(&at * b)) {
            let offset = rz * mount.offset();
            seeds.push(DVector::from_vec(vec![
                position[0] - offset.x,
                position[1] - offset.y,
                yaw,
            ]));
        }
    }
    let residual = |params: &DVector<f64>| -> Option<DVector<f64>> {
        if !params.iter().all(|v| v.is_finite()) {
            return None;
        }
        let pose = mount.field_pose(params);
        let mut values = Vec::with_capacity(object_points.len() * 2);
        for (point, observed) in object_points.iter().zip(image_points) {
            let projected = project_point(point, &pose, camera, distortion)?;
            values.extend([projected.0 - observed.0, projected.1 - observed.1]);
        }
        Some(DVector::from_vec(values))
    };
    let mut estimates = Vec::<PoseEstimate>::new();
    seeds.sort_by(|a, b| {
        residual(a)
            .map(|v| v.norm_squared())
            .unwrap_or(f64::INFINITY)
            .total_cmp(
                &residual(b)
                    .map(|v| v.norm_squared())
                    .unwrap_or(f64::INFINITY),
            )
    });
    seeds.truncate(18);
    for mut params in seeds {
        let Some(mut values) = residual(&params) else {
            continue;
        };
        let mut cost = values.norm_squared();
        let mut lambda = 1e-3;
        for _ in 0..60 {
            let mut jacobian = DMatrix::zeros(values.len(), 3);
            for j in 0..3 {
                let mut next = params.clone();
                next[j] += 1e-6;
                if let Some(shifted) = residual(&next) {
                    jacobian.set_column(j, &((shifted - &values) / 1e-6));
                }
            }
            let jt = jacobian.transpose();
            let mut normal = &jt * &jacobian;
            for j in 0..3 {
                normal[(j, j)] += lambda;
            }
            let Some(delta) = normal.lu().solve(&(-&jt * &values)) else {
                break;
            };
            if delta.norm() < 1e-9 {
                break;
            }
            let next = &params + delta;
            if let Some(shifted) = residual(&next) {
                let next_cost = shifted.norm_squared();
                if next_cost < cost {
                    params = next;
                    values = shifted;
                    cost = next_cost;
                    lambda = (lambda * 0.3).max(1e-9);
                    continue;
                }
            }
            lambda = (lambda * 10.).min(1e12);
        }
        let rmse = (cost / (object_points.len() * 2) as f64).sqrt();
        let pose = mount.field_pose(&params);
        let origin = -pose.rotation.transpose() * pose.translation;
        let front_facing = object_points.chunks_exact(4).all(|tag| {
            let normal = (tag[1] - tag[0]).cross(&(tag[2] - tag[0]));
            normal.dot(&(origin - tag[0])) > 0.
        });
        if front_facing
            && rmse.is_finite()
            && !estimates.iter().any(|old| {
                (old.pose.translation - pose.translation).norm() < 0.01
                    && (old.pose.rotation - pose.rotation).norm() < 0.01
            })
        {
            estimates.push(PoseEstimate {
                pose,
                reprojection_rmse_px: rmse,
            });
        }
    }
    estimates.sort_by(|a, b| a.reprojection_rmse_px.total_cmp(&b.reprojection_rmse_px));
    estimates.truncate(6);
    estimates
}

#[cfg(test)]
mod planar_tests {
    use super::*;

    fn camera() -> CameraConfig {
        serde_json::from_str(r#"{"fx":600,"fy":610,"cx":320,"cy":240,"k1":0,"k2":0,"k3":0,"p1":0,"p2":0,"tag_size_m":0.1651}"#).unwrap()
    }

    fn wall_tags() -> Vec<Vector3<f64>> {
        [3., 3.7]
            .into_iter()
            .flat_map(|y| {
                [
                    Vector3::new(5., y + 0.08255, 0.91745),
                    Vector3::new(5., y - 0.08255, 0.91745),
                    Vector3::new(5., y - 0.08255, 1.08255),
                    Vector3::new(5., y + 0.08255, 1.08255),
                ]
            })
            .collect()
    }

    #[test]
    fn planar_fit_recovers_position_from_pixels_without_pose_seed() {
        for (height, tilt, distortion) in
            [(0.65, 90., false), (0.85, 100., false), (0.65, 90., true)]
        {
            let mut camera = camera();
            if distortion {
                camera.k1 = -0.08;
                camera.k2 = 0.015;
                camera.p1 = 0.0003;
            }
            let mount = RobotMount {
                x_offset: 0.2,
                y_offset: -0.1,
                z_offset: height,
                roll_deg: -90.,
                pitch_deg: 0.,
                yaw_deg: tilt,
            };
            let truth = mount.field_pose(&DVector::from_vec(vec![2., 3., 0.12]));
            let points = wall_tags();
            let pixels: Vec<_> = points
                .iter()
                .map(|p| project_point(p, &truth, &camera, distortion).unwrap())
                .collect();
            let result =
                estimate_planar_pose(&points, &pixels, &camera, distortion, &mount, &[]).unwrap();
            assert!((result.pose.rotation - truth.rotation).norm() < 1e-5);
            assert!((result.pose.translation - truth.translation).norm() < 1e-5);
            assert!(result.reprojection_rmse_px < 1e-5);
        }
    }

    #[test]
    fn single_tag_planar_fit_escapes_bad_free_pose_seed() {
        let camera = camera();
        let mount = RobotMount {
            x_offset: 0.2,
            y_offset: -0.1,
            z_offset: 0.65,
            roll_deg: -90.,
            pitch_deg: 0.,
            yaw_deg: 90.,
        };
        let truth = mount.field_pose(&DVector::from_vec(vec![2., 3., 0.1]));
        let points = wall_tags()[..4].to_vec();
        let pixels: Vec<_> = points
            .iter()
            .map(|p| project_point(p, &truth, &camera, false).unwrap())
            .collect();
        let bad = mount.field_pose(&DVector::from_vec(vec![1., 2., 0.6]));
        let result =
            estimate_planar_pose(&points, &pixels, &camera, false, &mount, &[bad]).unwrap();
        assert!((result.pose.translation - truth.translation).norm() < 1e-5);
    }

    #[test]
    fn planar_fit_rejects_invalid_mount_and_incomplete_tag() {
        let camera = camera();
        let mut mount = RobotMount::from(&camera);
        mount.z_offset = f64::NAN;
        assert!(
            estimate_planar_pose(&wall_tags(), &[(0., 0.); 8], &camera, false, &mount, &[])
                .is_none()
        );
        mount.z_offset = 0.65;
        assert!(estimate_planar_pose(
            &wall_tags()[..5],
            &[(0., 0.); 5],
            &camera,
            false,
            &mount,
            &[]
        )
        .is_none());
    }
}

/// estimates 3d pose from 4 image corners
///
/// # Arguments
/// * `corners` - Native AprilTag corners (u, v), ordered BL, BR, TR, TL.
/// * `tag_size` - The physical size of the tag (e.g., in meters).
/// * `fx, fy, cx, cy` - Camera intrinsics.
///
/// # Returns
/// a pose struct containing the rotation matrix and translation vector (x, y, z).
pub fn estimate_pose(
    corners: &[(f64, f64); 4],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<PoseEstimate> {
    // 1. normalize image coordinates
    let image_points: Vec<Point2<f64>> = corners
        .iter()
        .map(|p| {
            let (x, y) = if apply_distortion {
                undistort_normalized_point(*p, camera)
            } else {
                ((p.0 - camera.cx) / camera.fx, (p.1 - camera.cy) / camera.fy)
            };
            Point2::new(x, y)
        })
        .collect();

    // 2. define model points tag relative
    let s = camera.tag_size_m / 2.0;
    let model_points_3d = [
        Vector3::new(-s, s, 0.0),  // 0: Bottom-Left
        Vector3::new(s, s, 0.0),   // 1: Bottom-Right
        Vector3::new(s, -s, 0.0),  // 2: Top-Right
        Vector3::new(-s, -s, 0.0), // 3: Top-Left
    ];
    let model_points = [
        Point2::new(-s, s),  // 0: Bottom-Left
        Point2::new(s, s),   // 1: Bottom-Right
        Point2::new(s, -s),  // 2: Top-Right
        Point2::new(-s, -s), // 3: Top-Left
    ];

    // 3. solve homography such that p ~ H * P
    let mut a_data = Vec::with_capacity(8 * 9);
    for i in 0..4 {
        let X = model_points[i].x;
        let Y = model_points[i].y;
        let u = image_points[i].x;
        let v = image_points[i].y;

        // Row 1
        a_data.extend_from_slice(&[-X, -Y, -1.0, 0.0, 0.0, 0.0, u * X, u * Y, u]);
        // Row 2
        a_data.extend_from_slice(&[0.0, 0.0, 0.0, -X, -Y, -1.0, v * X, v * Y, v]);
    }

    let a = DMatrix::from_row_slice(8, 9, &a_data);

    // solve Ah=0 via svd/eigen
    let ata = a.transpose() * &a; // 9x9 matrix

    let eigen = SymmetricEigen::new(ata);

    // Find index of smallest eigenvalue
    let mut min_val = f64::MAX;
    let mut min_idx = 0;
    for (i, val) in eigen.eigenvalues.iter().enumerate() {
        if *val < min_val {
            min_val = *val;
            min_idx = i;
        }
    }

    let h_vec = eigen.eigenvectors.column(min_idx);

    let mut h = Matrix3::new(
        h_vec[0], h_vec[1], h_vec[2], h_vec[3], h_vec[4], h_vec[5], h_vec[6], h_vec[7], h_vec[8],
    );

    // 4. decompose homography
    // H = [h1 h2 h3]
    // R = [r1 r2 r3], t
    // h1 ~ r1, h2 ~ r2, h3 ~ t
    // Constraint: ||r1|| = 1, ||r2|| = 1.

    // scale estimation: avg(norm(h1), norm(h2))

    let norm_h1 = h.column(0).norm();
    let norm_h2 = h.column(1).norm();
    let scale = (norm_h1 + norm_h2) / 2.0;

    if scale.abs() < 1e-6 {
        return None;
    }

    let mut t = h.column(2) / scale;

    let r1 = h.column(0) / scale;
    let r2 = h.column(1) / scale;
    let r3 = r1.cross(&r2);

    // enforce orthogonality via svd
    let r_raw = Matrix3::from_columns(&[r1, r2, r3]);
    let r_svd = r_raw.svd(true, true);
    if let (Some(u), Some(v_t)) = (r_svd.u, r_svd.v_t) {
        let mut R = u * v_t;

        // ensure det(R) = 1 (proper rotation)
        if R.determinant() < 0.0 {
            // det=-1 implies reflection
            R = -R; // This flips det for 3x3
        }

        // fix ambiguity: t.z > 0
        // t.z < 0 implies H sign wrong
        if t.z < 0.0 {
            t = -t;

            let c0 = -R.column(0);
            let c1 = -R.column(1);
            let c2 = R.column(2).into_owned();
            R = Matrix3::from_columns(&[c0, c1, c2]);
        }

        let initial_pose = Pose {
            rotation: R,
            translation: t,
        };

        let fronto_parallel_pose = fronto_parallel_initial_pose(corners, camera);

        return select_best_pose_estimate(
            &model_points_3d,
            corners,
            camera,
            apply_distortion,
            [Some(initial_pose), fronto_parallel_pose],
        );
    }

    None
}

pub fn estimate_pose_from_correspondences(
    object_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
    apply_distortion: bool,
    initial_pose: &Pose,
) -> Option<PoseEstimate> {
    if object_points.len() != image_points.len() || object_points.len() < 4 {
        return None;
    }
    select_best_pose_estimate(
        object_points,
        image_points,
        camera,
        apply_distortion,
        [Some(initial_pose.clone()), None],
    )
}

/// Refine the best field-pose seeds against every observed corner. Seed ranking
/// uses image reprojection only; no known camera/robot position is required.
pub fn estimate_pose_from_correspondence_candidates(
    object_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
    apply_distortion: bool,
    initial_poses: &[Pose],
) -> Option<PoseEstimate> {
    if object_points.len() != image_points.len() || object_points.len() < 4 {
        return None;
    }
    let mut ranked: Vec<_> = initial_poses
        .iter()
        .filter_map(|initial| {
            let cost = reprojection_cost(
                &pose_to_params(initial),
                object_points,
                image_points,
                camera,
                apply_distortion,
            )?;
            cost.is_finite().then_some((cost, initial))
        })
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    ranked
        .into_iter()
        .take(3)
        .filter_map(|(_, initial)| {
            estimate_pose_from_correspondences(
                object_points,
                image_points,
                camera,
                apply_distortion,
                initial,
            )
        })
        .min_by(|a, b| a.reprojection_rmse_px.total_cmp(&b.reprojection_rmse_px))
}

pub fn reprojection_rmse_px_for_pose(
    pose: &Pose,
    corners: &[(f64, f64)],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<f64> {
    let cost = reprojection_cost(
        &pose_to_params(pose),
        &square_model_points(camera.tag_size_m),
        corners,
        camera,
        apply_distortion,
    )?;
    Some((cost / (corners.len() as f64 * 2.0)).sqrt())
}

pub fn correspondence_reprojection_rmse(
    pose: &Pose,
    object_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
) -> Option<f64> {
    if object_points.len() != image_points.len() || object_points.is_empty() {
        return None;
    }
    let cost = reprojection_cost(
        &pose_to_params(pose),
        object_points,
        image_points,
        camera,
        true,
    )?;
    Some((cost / (object_points.len() * 2) as f64).sqrt())
}

fn refine_pose_pnp(
    initial_pose: &Pose,
    model_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<Pose> {
    let mut params = pose_to_params(initial_pose);

    let mut lambda = 1e-3;
    let mut best_cost = reprojection_cost(
        &params,
        model_points,
        image_points,
        camera,
        apply_distortion,
    )?;

    for _ in 0..20 {
        let residuals = reprojection_residuals(
            &params,
            model_points,
            image_points,
            camera,
            apply_distortion,
        )?;
        let jacobian = numerical_jacobian(
            &params,
            model_points,
            image_points,
            camera,
            apply_distortion,
        )?;
        let jt = jacobian.transpose();
        let mut normal = &jt * &jacobian;
        for i in 0..6 {
            normal[(i, i)] += lambda;
        }
        let rhs = -(&jt * residuals);
        let delta = normal.lu().solve(&rhs)?;

        if delta.norm() < 1e-9 {
            break;
        }

        let candidate = &params + delta;
        let Some(candidate_cost) = reprojection_cost(
            &candidate,
            model_points,
            image_points,
            camera,
            apply_distortion,
        ) else {
            lambda *= 10.0;
            continue;
        };

        if candidate_cost < best_cost {
            params = candidate;
            best_cost = candidate_cost;
            lambda = (lambda * 0.3).max(1e-6);
        } else {
            lambda = (lambda * 10.0).min(1e6);
        }
    }

    params_to_pose(&params)
}

fn reprojection_cost(
    params: &DVector<f64>,
    model_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<f64> {
    let residuals =
        reprojection_residuals(params, model_points, image_points, camera, apply_distortion)?;
    Some(residuals.dot(&residuals))
}

fn reprojection_residuals(
    params: &DVector<f64>,
    model_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<DVector<f64>> {
    if model_points.len() != image_points.len() || model_points.is_empty() {
        return None;
    }
    let pose = params_to_pose(params)?;
    let mut residuals = Vec::with_capacity(8);
    for (point, observed) in model_points.iter().zip(image_points.iter()) {
        let projected = project_point(point, &pose, camera, apply_distortion)?;
        residuals.push(projected.0 - observed.0);
        residuals.push(projected.1 - observed.1);
    }
    Some(DVector::from_vec(residuals))
}

fn numerical_jacobian(
    params: &DVector<f64>,
    model_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<DMatrix<f64>> {
    let base =
        reprojection_residuals(params, model_points, image_points, camera, apply_distortion)?;
    let mut jacobian = DMatrix::zeros(base.len(), params.len());
    for col in 0..params.len() {
        let mut perturbed = params.clone();
        let eps = if col < 3 { 1e-6 } else { 1e-5 };
        perturbed[col] += eps;
        let shifted = reprojection_residuals(
            &perturbed,
            model_points,
            image_points,
            camera,
            apply_distortion,
        )?;
        let diff = (shifted - &base) / eps;
        jacobian.set_column(col, &diff);
    }
    Some(jacobian)
}

fn pose_to_params(pose: &Pose) -> DVector<f64> {
    let rotation = Rotation3::from_matrix_unchecked(pose.rotation);
    let axis = rotation.scaled_axis();
    DVector::from_vec(vec![
        axis.x,
        axis.y,
        axis.z,
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
    ])
}

fn params_to_pose(params: &DVector<f64>) -> Option<Pose> {
    if params.len() != 6 || !params.iter().all(|v| v.is_finite()) {
        return None;
    }

    let rotation = Rotation3::new(Vector3::new(params[0], params[1], params[2]));
    let translation = Vector3::new(params[3], params[4], params[5]);
    // A field-origin translation may be behind the camera. Visibility is checked
    // per transformed object point in project_point, not at the coordinate origin.
    Some(Pose {
        rotation: rotation.into_inner(),
        translation,
    })
}

fn project_point(
    point: &Vector3<f64>,
    pose: &Pose,
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<(f64, f64)> {
    let camera_point = pose.rotation * point + pose.translation;
    if camera_point.z <= 1e-6 {
        return None;
    }

    let mut x = camera_point.x / camera_point.z;
    let mut y = camera_point.y / camera_point.z;
    if apply_distortion {
        (x, y) = distort_normalized_point((x, y), camera);
    }
    Some((camera.fx * x + camera.cx, camera.fy * y + camera.cy))
}

pub(crate) fn distort_normalized_point(p: (f64, f64), camera: &CameraConfig) -> (f64, f64) {
    let (x, y) = p;
    let r2 = x * x + y * y;
    let radial = 1.0 + camera.k1 * r2 + camera.k2 * r2 * r2 + camera.k3 * r2 * r2 * r2;
    let x_tan = 2.0 * camera.p1 * x * y + camera.p2 * (r2 + 2.0 * x * x);
    let y_tan = camera.p1 * (r2 + 2.0 * y * y) + 2.0 * camera.p2 * x * y;
    (x * radial + x_tan, y * radial + y_tan)
}

pub(crate) fn undistort_normalized_point(pixel: (f64, f64), camera: &CameraConfig) -> (f64, f64) {
    let x_dist = (pixel.0 - camera.cx) / camera.fx;
    let y_dist = (pixel.1 - camera.cy) / camera.fy;
    let mut x = x_dist;
    let mut y = y_dist;

    // Damped Newton inversion remains accurate at the edge of wide-angle
    // cameras, where a short fixed-point iteration can oscillate or stall.
    for _ in 0..30 {
        let r2 = x * x + y * y;
        let radial = 1.0 + camera.k1 * r2 + camera.k2 * r2 * r2 + camera.k3 * r2 * r2 * r2;
        let (u, v) = distort_normalized_point((x, y), camera);
        let error = Vector2::new(u - x_dist, v - y_dist);
        if error.norm_squared() < 1e-24 {
            break;
        }
        let slope = camera.k1 + 2. * camera.k2 * r2 + 3. * camera.k3 * r2 * r2;
        let a = radial + 2. * x * x * slope + 2. * camera.p1 * y + 6. * camera.p2 * x;
        let b = 2. * x * y * slope + 2. * camera.p1 * x + 2. * camera.p2 * y;
        let d = radial + 2. * y * y * slope + 6. * camera.p1 * y + 2. * camera.p2 * x;
        let determinant = a * d - b * b;
        if !determinant.is_finite() || determinant.abs() < 1e-12 {
            break;
        }
        let step = Vector2::new(
            (d * error.x - b * error.y) / determinant,
            (a * error.y - b * error.x) / determinant,
        );
        let mut scale = 1.;
        let mut improved = false;
        for _ in 0..12 {
            let next = (x - scale * step.x, y - scale * step.y);
            let projected = distort_normalized_point(next, camera);
            let cost = (projected.0 - x_dist).powi(2) + (projected.1 - y_dist).powi(2);
            if cost.is_finite() && cost < error.norm_squared() {
                (x, y) = next;
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if !improved {
            break;
        }
    }

    (x, y)
}

fn fronto_parallel_initial_pose(corners: &[(f64, f64); 4], camera: &CameraConfig) -> Option<Pose> {
    let side_len_px = (((corners[0].0 - corners[1].0).powi(2)
        + (corners[0].1 - corners[1].1).powi(2))
    .sqrt()
        + ((corners[1].0 - corners[2].0).powi(2) + (corners[1].1 - corners[2].1).powi(2)).sqrt()
        + ((corners[2].0 - corners[3].0).powi(2) + (corners[2].1 - corners[3].1).powi(2)).sqrt()
        + ((corners[3].0 - corners[0].0).powi(2) + (corners[3].1 - corners[0].1).powi(2)).sqrt())
        / 4.0;
    if side_len_px <= 1e-6 {
        return None;
    }

    let z = (camera.fx * camera.tag_size_m) / side_len_px;
    let center_x = (corners[0].0 + corners[2].0) * 0.5;
    let center_y = (corners[0].1 + corners[2].1) * 0.5;
    let x = (center_x - camera.cx) * z / camera.fx;
    let y = (center_y - camera.cy) * z / camera.fy;
    Some(Pose {
        rotation: Matrix3::identity(),
        translation: Vector3::new(x, y, z),
    })
}

fn square_model_points(tag_size_m: f64) -> [Vector3<f64>; 4] {
    let s = tag_size_m / 2.0;
    [
        Vector3::new(-s, s, 0.0),
        Vector3::new(s, s, 0.0),
        Vector3::new(s, -s, 0.0),
        Vector3::new(-s, -s, 0.0),
    ]
}

fn select_best_pose_estimate<const N: usize>(
    model_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
    camera: &CameraConfig,
    apply_distortion: bool,
    candidates: [Option<Pose>; N],
) -> Option<PoseEstimate> {
    let mut best_pose = None;
    let mut best_cost = f64::INFINITY;
    for candidate in candidates.into_iter().flatten() {
        let refined = refine_pose_pnp(
            &candidate,
            model_points,
            image_points,
            camera,
            apply_distortion,
        )
        .or(Some(candidate));
        if let Some(pose) = refined {
            if let Some(cost) = reprojection_cost(
                &pose_to_params(&pose),
                model_points,
                image_points,
                camera,
                apply_distortion,
            ) {
                if cost < best_cost {
                    best_cost = cost;
                    best_pose = Some(PoseEstimate {
                        pose,
                        reprojection_rmse_px: (cost / (image_points.len() as f64 * 2.0)).sqrt(),
                    });
                }
            }
        }
    }
    best_pose
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strong_distortion_roundtrips_wide_angle_edge_rays() {
        for (k1, k2, k3) in [
            (-0.3, 0.08, -0.005),
            (0.35, 0.04, 0.01),
            (0.00055, -0.0255, 0.0044),
        ] {
            let mut camera = camera();
            camera.k1 = k1;
            camera.k2 = k2;
            camera.k3 = k3;
            camera.p1 = 0.003;
            camera.p2 = -0.002;
            for ix in -7..=7 {
                for iy in -4..=4 {
                    let point = (ix as f64 * 0.18, iy as f64 * 0.18);
                    let distorted = distort_normalized_point(point, &camera);
                    let recovered = undistort_normalized_point(
                        (
                            distorted.0 * camera.fx + camera.cx,
                            distorted.1 * camera.fy + camera.cy,
                        ),
                        &camera,
                    );
                    assert!(
                        (recovered.0 - point.0).hypot(recovered.1 - point.1) < 1e-8,
                        "{point:?} -> {recovered:?}, k1={k1}"
                    );
                }
            }
        }
    }

    fn camera() -> CameraConfig {
        CameraConfig {
            fx: 600.0,
            fy: 590.0,
            cx: 960.0,
            cy: 540.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
            tag_size_m: 0.16,
            x_offset: 0.0,
            y_offset: 0.0,
            z_offset: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
            roll_deg: 0.0,
        }
    }

    fn field_points() -> Vec<Vector3<f64>> {
        [-0.5, 0.5]
            .into_iter()
            .flat_map(|x| square_model_points(0.16).map(|p| p + Vector3::new(x, 0.0, 10.0)))
            .collect()
    }

    #[test]
    fn refines_visible_field_points_with_negative_origin_depth() {
        let c = camera();
        let points = field_points();
        let truth = Pose {
            rotation: Matrix3::identity(),
            translation: Vector3::new(0.0, 0.0, -8.0),
        };
        let pixels: Vec<_> = points
            .iter()
            .map(|p| project_point(p, &truth, &c, false).unwrap())
            .collect();
        let seed = Pose {
            rotation: Matrix3::identity(),
            translation: Vector3::new(0.03, -0.02, -7.9),
        };
        let result =
            estimate_pose_from_correspondences(&points, &pixels, &c, false, &seed).unwrap();
        assert!((result.pose.translation - truth.translation).norm() < 1e-5);
        assert!(result.reprojection_rmse_px < 1e-4);
    }

    #[test]
    fn rejects_points_behind_camera_even_when_origin_is_in_front() {
        let c = camera();
        let points = vec![Vector3::new(0.0, 0.0, -10.0); 4];
        let seed = Pose {
            rotation: Matrix3::identity(),
            translation: Vector3::new(0.0, 0.0, 1.0),
        };
        assert!(estimate_pose_from_correspondences(
            &points,
            &[(960.0, 540.0); 4],
            &c,
            false,
            &seed
        )
        .is_none());
    }

    #[test]
    fn field_origin_shift_preserves_projection() {
        let c = camera();
        let points = field_points();
        let truth = Pose {
            rotation: Matrix3::identity(),
            translation: Vector3::new(0.0, 0.0, -8.0),
        };
        let pixels: Vec<_> = points
            .iter()
            .map(|p| project_point(p, &truth, &c, false).unwrap())
            .collect();
        let offset = Vector3::new(12.0, -3.0, 20.0);
        let shifted: Vec<_> = points.iter().map(|p| p + offset).collect();
        let shifted_pose = Pose {
            rotation: truth.rotation,
            translation: truth.translation - offset,
        };
        let estimate = estimate_pose_from_correspondence_candidates(
            &shifted,
            &pixels,
            &c,
            false,
            &[shifted_pose],
        )
        .unwrap();
        assert!(estimate.reprojection_rmse_px < 1e-8);
        assert!((estimate.pose.translation + offset - truth.translation).norm() < 1e-8);
    }
}
