// Camera intrinsics and tag size
// Defaults aligned for a 1920x1200 AR0234-based USB camera with ~126° HFOV.
// fx is approximated from HFOV: fx ≈ W / (2 * tan(HFOV/2)).
// These are placeholders; for accurate 3D, calibrate and update.
pub const CAM_FX: f64 = 490.0; // pixels (approx for 126° at 1920 width)
pub const CAM_FY: f64 = 490.0; // assume square pixels; update after calibration
pub const CAM_CX: f64 = 960.0; // principal point at image center (1920/2)
pub const CAM_CY: f64 = 600.0; // principal point at image center (1200/2)
pub const TAG_SIZE_M: f64 = 0.16; // meters

// Image darkening gamma
pub const DARKEN_GAMMA: f32 = 0.7;

#[derive(Clone, Copy)]
pub struct CameraParams {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub tag_size_m: f64,
}

pub fn camera_params() -> CameraParams {
    CameraParams { fx: CAM_FX, fy: CAM_FY, cx: CAM_CX, cy: CAM_CY, tag_size_m: TAG_SIZE_M }
}