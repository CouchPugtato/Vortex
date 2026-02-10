use anyhow::Result;
use apriltag::{DetectorBuilder, families::Family, Detector, Image as AprilImage};

#[repr(C)]
struct RawDetector {
    nthreads: i32,
    quad_decimate: f32,
    quad_sigma: f32,
    refine_edges: i32,
    decode_sharpening: f64,
    debug: i32,
}

#[derive(Debug, Clone)]
pub struct RawDetection {
    pub id: usize,
    pub center: [f64; 2],
    pub corners: [[f64; 2]; 4],
}

pub struct CpuDetector {
    inner: Detector,
}

impl CpuDetector {
    pub fn new(nthreads: i32) -> Result<Self> {
        let detector = build_inner_detector(nthreads)?;
        Ok(Self { inner: detector })
    }

    pub fn detect(&mut self, gray_data: &[u8], width: usize, height: usize) -> Result<Vec<RawDetection>> {
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

fn detect_corners(detector: &mut Detector, gray_data: &[u8], width: usize, height: usize) -> Result<Vec<RawDetection>> {
    let mut img = unsafe { AprilImage::new_uinit(width, height)? };
    
    let dst = img.as_mut();
    
    if dst.len() == gray_data.len() {
        dst.copy_from_slice(gray_data);
    } else {
        let copy_len = std::cmp::min(dst.len(), gray_data.len());
        dst[..copy_len].copy_from_slice(&gray_data[..copy_len]);
    }

    let detections = detector.detect(&img);
    
    let mut results: Vec<RawDetection> = Vec::new();
    for det in detections.iter() {
        let corners = det.corners();
        let c_arr = [
            [corners[0][0], corners[0][1]],
            [corners[1][0], corners[1][1]],
            [corners[2][0], corners[2][1]],
            [corners[3][0], corners[3][1]],
        ];
        
        let center = det.center();
        let center_arr = [center[0], center[1]];

        results.push(RawDetection {
            id: det.id(),
            center: center_arr,
            corners: c_arr,
        });
    }

    Ok(results)
}
