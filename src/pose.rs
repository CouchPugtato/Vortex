use nalgebra::{Matrix3, Vector3, Point2, SymmetricEigen};
use apriltag::Detection;
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
    tag_size: f64,
    fx: f64, fy: f64, cx: f64, cy: f64
) -> Option<Pose> {
    // 1. normalize image coordinates
    let image_points: Vec<Point2<f64>> = corners.iter().map(|p| {
        Point2::new((p.0 - cx) / fx, (p.1 - cy) / fy)
    }).collect();

    // 2. define model points tag relative
    let s = tag_size / 2.0;
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

    let a = nalgebra::DMatrix::from_row_slice(8, 9, &a_data);
    
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

        return Some(Pose {
            rotation: R,
            translation: t,
        });
    }

    None
}
