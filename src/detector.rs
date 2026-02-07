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

pub fn build_detector(nthreads: i32) -> Result<Detector> {
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

pub fn detect_corners(detector: &mut Detector, gray_data: &[u8], width: usize, height: usize) -> Result<Vec<[(f64, f64); 4]>> {
    // new_uinit is unsafe because it returns uninitialized memory
    let mut img = unsafe { AprilImage::new_uinit(width, height)? };
    
    let dst = img.as_mut();
    
    if dst.len() == gray_data.len() {
        dst.copy_from_slice(gray_data);
    } else {
        let copy_len = std::cmp::min(dst.len(), gray_data.len());
        dst[..copy_len].copy_from_slice(&gray_data[..copy_len]);
    }

    let detections = detector.detect(&img);
    
    let mut corners_list: Vec<[(f64, f64); 4]> = Vec::new();
    for det in detections.iter() {
        let corners = det.corners();
        let to_f = |p: [f64; 2]| (p[0], p[1]);
        corners_list.push([to_f(corners[0]), to_f(corners[1]), to_f(corners[2]), to_f(corners[3])]);
    }

    Ok(corners_list)
}
