use anyhow::Result;
use apriltag::{
    pose::TagParams,
    Detection as AprilRawDetection,
    Detector,
    DetectorBuilder,
    Image as AprilImage,
    families::Family,
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
        det.estimate_tag_pose_orthogonal_iteration(&params, n_iters)
            .into_iter()
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
                    pose::reprojection_rmse_px_for_pose(&pose, &solver_corners(self.corners), camera, apply_distortion)?;
                Some(PoseEstimate {
                    pose,
                    reprojection_rmse_px,
                })
            })
            .collect()
    }

    pub fn drop_cpu_pose_source(&mut self) {
        self.cpu_detection = None;
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
}

impl CpuDetector {
    pub fn new(nthreads: i32) -> Result<Self> {
        let detector = build_inner_detector(nthreads)?;
        Ok(Self { inner: detector })
    }

    pub fn detect(&mut self, gray_data: &[u8], width: usize, height: usize) -> Result<Vec<Detection>> {
        detect_corners(&mut self.inner, gray_data, width, height)
    }
}

fn build_inner_detector(nthreads: i32) -> Result<Detector> {
    let family = Family::tag_36h11();
    let bits: usize = 3;
    let detector = DetectorBuilder::new()
            .add_family_bits(family, bits)
            .build()?;

    // access the underlying C struct to set parameters not exposed by the wrapper
    unsafe {
        let ptr_ptr = &detector as *const Detector as *const *mut RawDetector;
        let raw_ptr = *ptr_ptr;
        
        if !raw_ptr.is_null() {
            (*raw_ptr).nthreads = nthreads;
            (*raw_ptr).quad_decimate = 3.0;
            (*raw_ptr).quad_sigma = 0.0;
            (*raw_ptr).refine_edges = 1;
        }
    }

    Ok(detector)
}

fn detect_corners(detector: &mut Detector, gray_data: &[u8], width: usize, height: usize) -> Result<Vec<Detection>> {
    let mut img = unsafe { AprilImage::new_uinit(width, height)? };
    
    let dst = img.as_mut();
    
    if dst.len() == gray_data.len() {
        dst.copy_from_slice(gray_data);
    } else {
        let copy_len = std::cmp::min(dst.len(), gray_data.len());
        dst[..copy_len].copy_from_slice(&gray_data[..copy_len]);
    }

    let detections = detector.detect(&img);
    
    let mut results: Vec<Detection> = Vec::new();
    for det in detections {
        let corners = det.corners();
        let c_arr = [
            [corners[0][0], corners[0][1]],
            [corners[1][0], corners[1][1]],
            [corners[2][0], corners[2][1]],
            [corners[3][0], corners[3][1]],
        ];
        
        let center = det.center();
        let center_arr = [center[0], center[1]];

        results.push(Detection::AprilTag(AprilTagDetection {
            id: det.id(),
            center: center_arr,
            corners: c_arr,
            cpu_detection: Some(det),
        }));
    }

    Ok(results)
}

fn solver_corners(corners: [[f64; 2]; 4]) -> [(f64, f64); 4] {
    [
        (corners[3][0], corners[3][1]),
        (corners[0][0], corners[0][1]),
        (corners[1][0], corners[1][1]),
        (corners[2][0], corners[2][1]),
    ]
}
