use nalgebra::{DMatrix, DVector, Matrix3, Point2, Rotation3, SymmetricEigen, Vector3};

use crate::config::CameraConfig;

#[derive(Debug, Clone)]
pub struct Pose {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

/// estimates 3d pose from 4 image corners
/// 
/// # Arguments
/// * `corners` - The 4 corners of the tag in the image (u, v). Order: TL, BL, BR, TR (or CCW).
/// * `tag_size` - The physical size of the tag (e.g., in meters).
/// * `fx, fy, cx, cy` - Camera intrinsics.
/// 
/// # Returns
/// a pose struct containing the rotation matrix and translation vector (x, y, z).
pub fn estimate_pose(
    corners: &[(f64, f64); 4],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<Pose> {
    // 1. normalize image coordinates
    let image_points: Vec<Point2<f64>> = corners.iter().map(|p| {
        let (x, y) = if apply_distortion {
            undistort_normalized_point(*p, camera)
        } else {
            (
                (p.0 - camera.cx) / camera.fx,
                (p.1 - camera.cy) / camera.fy,
            )
        };
        Point2::new(x, y)
    }).collect();

    // 2. define model points tag relative
    let s = camera.tag_size_m / 2.0;
    let model_points_3d = [
        Vector3::new(-s, -s, 0.0), // 0: Top-Left
        Vector3::new(-s,  s, 0.0), // 1: Bottom-Left
        Vector3::new( s,  s, 0.0), // 2: Bottom-Right
        Vector3::new( s, -s, 0.0), // 3: Top-Right
    ];
    let model_points = [
        Point2::new(-s, -s), // 0: Top-Left
        Point2::new(-s,  s), // 1: Bottom-Left
        Point2::new( s,  s), // 2: Bottom-Right
        Point2::new( s, -s), // 3: Top-Right
    ];

    // 3. solve homography such that p ~ H * P
    let mut a_data = Vec::with_capacity(8 * 9);
    for i in 0..4 {
        let X = model_points[i].x;
        let Y = model_points[i].y;
        let u = image_points[i].x;
        let v = image_points[i].y;

        // Row 1
        a_data.extend_from_slice(&[
            -X, -Y, -1.0, 
             0.0, 0.0, 0.0, 
             u*X, u*Y, u
        ]);
        // Row 2
        a_data.extend_from_slice(&[
             0.0, 0.0, 0.0, 
            -X, -Y, -1.0, 
             v*X, v*Y, v
        ]);
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
        h_vec[0], h_vec[1], h_vec[2],
        h_vec[3], h_vec[4], h_vec[5],
        h_vec[6], h_vec[7], h_vec[8],
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
    
    if scale.abs() < 1e-6 { return None; }

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

        let mut best_pose = None;
        let mut best_cost = f64::INFINITY;
        for candidate in [Some(initial_pose), fronto_parallel_pose].into_iter().flatten() {
            let refined = refine_pose_pnp(
                &candidate,
                &model_points_3d,
                corners,
                camera,
                apply_distortion,
            )
            .or(Some(candidate));
            if let Some(pose) = refined {
                if let Some(cost) = reprojection_cost(
                    &pose_to_params(&pose),
                    &model_points_3d,
                    corners,
                    camera,
                    apply_distortion,
                ) {
                    if cost < best_cost {
                        best_cost = cost;
                        best_pose = Some(pose);
                    }
                }
            }
        }
        return best_pose;
    }

    None
}

fn refine_pose_pnp(
    initial_pose: &Pose,
    model_points: &[Vector3<f64>; 4],
    image_points: &[(f64, f64); 4],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<Pose> {
    let mut params = pose_to_params(initial_pose);

    if params[5] <= 1e-6 {
        return None;
    }

    let mut lambda = 1e-3;
    let mut best_cost =
        reprojection_cost(&params, model_points, image_points, camera, apply_distortion)?;

    for _ in 0..20 {
        let residuals =
            reprojection_residuals(&params, model_points, image_points, camera, apply_distortion)?;
        let jacobian =
            numerical_jacobian(&params, model_points, image_points, camera, apply_distortion)?;
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
        let Some(candidate_cost) =
            reprojection_cost(&candidate, model_points, image_points, camera, apply_distortion)
        else {
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
    model_points: &[Vector3<f64>; 4],
    image_points: &[(f64, f64); 4],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<f64> {
    let residuals =
        reprojection_residuals(params, model_points, image_points, camera, apply_distortion)?;
    Some(residuals.dot(&residuals))
}

fn reprojection_residuals(
    params: &DVector<f64>,
    model_points: &[Vector3<f64>; 4],
    image_points: &[(f64, f64); 4],
    camera: &CameraConfig,
    apply_distortion: bool,
) -> Option<DVector<f64>> {
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
    model_points: &[Vector3<f64>; 4],
    image_points: &[(f64, f64); 4],
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
    if translation.z <= 1e-6 {
        return None;
    }

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

fn distort_normalized_point(p: (f64, f64), camera: &CameraConfig) -> (f64, f64) {
    let (x, y) = p;
    let r2 = x * x + y * y;
    let radial = 1.0 + camera.k1 * r2 + camera.k2 * r2 * r2 + camera.k3 * r2 * r2 * r2;
    let x_tan = 2.0 * camera.p1 * x * y + camera.p2 * (r2 + 2.0 * x * x);
    let y_tan = camera.p1 * (r2 + 2.0 * y * y) + 2.0 * camera.p2 * x * y;
    (x * radial + x_tan, y * radial + y_tan)
}

fn undistort_normalized_point(pixel: (f64, f64), camera: &CameraConfig) -> (f64, f64) {
    let x_dist = (pixel.0 - camera.cx) / camera.fx;
    let y_dist = (pixel.1 - camera.cy) / camera.fy;
    let mut x = x_dist;
    let mut y = y_dist;

    for _ in 0..8 {
        let r2 = x * x + y * y;
        let radial = 1.0 + camera.k1 * r2 + camera.k2 * r2 * r2 + camera.k3 * r2 * r2 * r2;
        if radial.abs() < 1e-9 {
            break;
        }
        let x_tan = 2.0 * camera.p1 * x * y + camera.p2 * (r2 + 2.0 * x * x);
        let y_tan = camera.p1 * (r2 + 2.0 * y * y) + 2.0 * camera.p2 * x * y;
        x = (x_dist - x_tan) / radial;
        y = (y_dist - y_tan) / radial;
    }

    (x, y)
}

fn fronto_parallel_initial_pose(
    corners: &[(f64, f64); 4],
    camera: &CameraConfig,
) -> Option<Pose> {
    let side_len_px = (
        ((corners[0].0 - corners[1].0).powi(2) + (corners[0].1 - corners[1].1).powi(2)).sqrt() +
        ((corners[1].0 - corners[2].0).powi(2) + (corners[1].1 - corners[2].1).powi(2)).sqrt() +
        ((corners[2].0 - corners[3].0).powi(2) + (corners[2].1 - corners[3].1).powi(2)).sqrt() +
        ((corners[3].0 - corners[0].0).powi(2) + (corners[3].1 - corners[0].1).powi(2)).sqrt()
    ) / 4.0;
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
