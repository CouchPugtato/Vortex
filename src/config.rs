// Camera intrinsics and tag size
pub const CAM_FX: f64 = 1000.0;
pub const CAM_FY: f64 = 1000.0;
pub const CAM_CX: f64 = 640.0;
pub const CAM_CY: f64 = 360.0;
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