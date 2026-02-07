use crate::config::CameraConfig;

// returns pixel coordinates as they would appear in an ideal pinhole camera with the same fx, fy, cx, cy but zero distortion.
pub fn undistort_points(corners: &[(f64, f64); 4], cam: &CameraConfig) -> [(f64, f64); 4] {
    let mut out = [(0.0, 0.0); 4];
    for (i, p) in corners.iter().enumerate() {
        out[i] = undistort_point(*p, cam);
    }
    out
}

fn undistort_point(p: (f64, f64), cam: &CameraConfig) -> (f64, f64) {
    let u0 = p.0;
    let v0 = p.1;

    let mut x = (u0 - cam.cx) / cam.fx;
    let mut y = (v0 - cam.cy) / cam.fy;

    // iterative solver  for (x, y) such that distort(x, y) ~ (x_obs, y_obs)
    let x0 = x;
    let y0 = y;
    
    for _ in 0..5 {
        let r2 = x*x + y*y;
        let r4 = r2*r2;
        let r6 = r2*r4;
        
        let k = 1.0 + cam.k1 * r2 + cam.k2 * r4 + cam.k3 * r6;
        let p_term_x = 2.0 * cam.p1 * x * y + cam.p2 * (r2 + 2.0 * x * x);
        let p_term_y = cam.p1 * (r2 + 2.0 * y * y) + 2.0 * cam.p2 * x * y;
        
        x = (x0 - p_term_x) / k;
        y = (y0 - p_term_y) / k;
    }
    
    // project back to pixel coordinates using linear model
    (x * cam.fx + cam.cx, y * cam.fy + cam.cy)
}
