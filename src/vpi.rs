#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use libc::{c_void, c_int};

// VPI Types
pub type VPIStream = *mut c_void;
pub type VPIImage = *mut c_void;
pub type VPIPayload = *mut c_void;
pub type VPIArray = *mut c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum VPIBackend {
    VPI_BACKEND_CPU = 1,
    VPI_BACKEND_CUDA = 2,
    VPI_BACKEND_PVA = 4,
    VPI_BACKEND_VIC = 8,
    // Combine flags as needed
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum VPIImageFormat {
    VPI_IMAGE_FORMAT_U8 = 1, // VPI_IMAGE_FORMAT_Y8_ER
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum VPIAprilTagFamily {
    VPI_APRILTAG_INVALID = 0,
    VPI_APRILTAG_16H5 = 1,
    VPI_APRILTAG_25H9 = 2,
    VPI_APRILTAG_36H10 = 3,
    VPI_APRILTAG_36H11 = 4,
    VPI_APRILTAG_CIRCLE21H7 = 5,
    VPI_APRILTAG_CIRCLE49H12 = 6,
    VPI_APRILTAG_CUSTOM48H12 = 7,
    VPI_APRILTAG_STANDARD41H12 = 8,
    VPI_APRILTAG_STANDARD52H13 = 9,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum VPIArrayType {
    VPI_ARRAY_TYPE_INVALID = 0,
    VPI_ARRAY_TYPE_S8 = 1,
    VPI_ARRAY_TYPE_U8 = 2,
    VPI_ARRAY_TYPE_S16 = 3,
    VPI_ARRAY_TYPE_U16 = 4,
    VPI_ARRAY_TYPE_U32 = 5,
    VPI_ARRAY_TYPE_KEYPOINT_F32 = 6,
    VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D = 7,
    VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX = 8,
    VPI_ARRAY_TYPE_F32 = 9,
    VPI_ARRAY_TYPE_KEYPOINT_U32 = 10,
    VPI_ARRAY_TYPE_KEYPOINT_UQ1616 = 11,
    VPI_ARRAY_TYPE_STATISTICS = 12,
    VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR = 13,
    VPI_ARRAY_TYPE_MATCHES = 14,
    VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX = 15,
    VPI_ARRAY_TYPE_PYRAMIDAL_KEYPOINT_F32 = 16,
    VPI_ARRAY_TYPE_APRILTAG_DETECTION = 17,
    VPI_ARRAY_TYPE_POSE = 18,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct VPIKeypointF32 {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VPIAprilTagDetection {
    pub id: i32,
    pub corners: [VPIKeypointF32; 4],
    pub center: VPIKeypointF32,
}

// warp / lens distortion
pub type VPIWarpMap = *mut c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum VPIInterpolationType {
    VPI_INTERPOLATION_NEAREST = 0,
    VPI_INTERPOLATION_LINEAR = 1,
    VPI_INTERPOLATION_CATMULL_ROM = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum VPIWarpMapType {
    VPI_WARP_MAP_FISHEYE_CORRECTION = 0, // placeholder
    VPI_WARP_MAP_GRID_ABSOLUTE = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VPICameraIntrinsic {
    pub matrix: [[f32; 3]; 2], // 2x3 matrix [[fx, 0, cx], [0, fy, cy]]
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VPIPolynomialLensDistortionModel {
    pub k1: f32,
    pub k2: f32,
    pub p1: f32,
    pub p2: f32,
    pub k3: f32,
    pub k4: f32,
    pub k5: f32,
    pub k6: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VPIImageCreateParams {
    pub width: i32,
    pub height: i32,
    pub format: u64, // vpiimageformat is uint64_t
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VPIAprilTagDecodeParams {
    pub tagIdFilter: *const u16,
    pub tagIdFilterSize: i32,
    pub maxBitsCorrected: i32,
    pub family: VPIAprilTagFamily,
}

impl Default for VPIAprilTagDecodeParams {
    fn default() -> Self {
        Self {
            tagIdFilter: std::ptr::null(),
            tagIdFilterSize: 0,
            maxBitsCorrected: 2, // standard default
            family: VPIAprilTagFamily::VPI_APRILTAG_36H11,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VPIImageData {
    pub bufferType: c_int, // vpiimagebuffertype
    pub buffer: VPIImageBuffer,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union VPIImageBuffer {
    pub pitch: VPIImageBufferPitch,
    // other fields omitted
}

impl std::fmt::Debug for VPIImageBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            write!(f, "VPIImageBuffer {{ pitch: {:?} }}", self.pitch)
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VPIImageBufferPitch {
    pub planes: [*mut c_void; 3],
    pub pitches: [i32; 3],
}

// bindings
#[cfg(feature = "gpu")]
#[link(name = "nvvpi")]
extern "C" {
    pub fn vpiStreamCreate(flags: u32, stream: *mut VPIStream) -> c_int;
    pub fn vpiStreamDestroy(stream: VPIStream);
    pub fn vpiStreamSync(stream: VPIStream) -> c_int;
    
    // vpi 3.x: returns version directly
    pub fn vpiGetVersion() -> i32;

    // status/error
    pub fn vpiGetLastStatusMessage() -> *const libc::c_char;
    pub fn vpiStatusGetName(status: c_int) -> *const libc::c_char;

    pub fn vpiImageCreateWrapper(
        data: *const VPIImageData, 
        params: *const c_void, // VPIImageCreateParams
        flags: u32,
        image: *mut VPIImage
    ) -> c_int;
    
    pub fn vpiImageCreate(
        width: i32,
        height: i32,
        format: u64,
        flags: u32,
        image: *mut VPIImage
    ) -> c_int;

    pub fn vpiImageDestroy(image: VPIImage);

    // warp / remap
    pub fn vpiWarpMapCreate(
        capacity: i32, // capacity
        width: i32,
        height: i32,
        format: i32, // vpiwarpmapformat
        warpMap: *mut VPIWarpMap
    ) -> c_int;

    pub fn vpiWarpMapDestroy(warpMap: VPIWarpMap);

    pub fn vpiWarpMapGenerateFromPolynomialLensDistortionModel(
        warpMap: VPIWarpMap,
        kin: VPICameraIntrinsic,
        kout: VPICameraIntrinsic,
        dist: *const VPIPolynomialLensDistortionModel
    ) -> c_int;

    pub fn vpiSubmitRemap(
        stream: VPIStream,
        backend: u64,
        warpMap: VPIWarpMap,
        input: VPIImage,
        output: VPIImage,
        interp: i32, // vpiinterpolationtype
        boundary: i32, // vpiboundarycond
        bg_color: *const c_void
    ) -> c_int;

    // histogram equalization
    pub fn vpiCreateHistogramEqualizer(
        flags: u32,
        payload: *mut VPIPayload
    ) -> c_int;

    pub fn vpiSubmitHistogramEqualization(
        stream: VPIStream,
        backend: u64,
        payload: VPIPayload,
        input: VPIImage,
        output: VPIImage
    ) -> c_int;

    pub fn vpiCreateAprilTagDetector(
        backends: u64,
        inputWidth: i32,
        inputHeight: i32,
        params: *const VPIAprilTagDecodeParams,
        payload: *mut VPIPayload
    ) -> c_int;
    
    pub fn vpiSubmitAprilTagDetector(
        stream: VPIStream,
        backend: u64,
        payload: VPIPayload,
        maxDetections: u32,
        input: VPIImage,
        output: VPIArray
    ) -> c_int;

    pub fn vpiPayloadDestroy(payload: VPIPayload);
    
    // outputs
    pub fn vpiArrayCreate(
        capacity: i32, 
        type_: i32, // vpiarraytype
        flags: u32, 
        array: *mut VPIArray
    ) -> c_int;
    pub fn vpiArrayDestroy(array: VPIArray);
    pub fn vpiArrayLockData(
        array: VPIArray, 
        mode: i32, // vpilockmode
        stride: *mut i32, 
        data: *mut *mut c_void
    ) -> c_int;
    pub fn vpiArrayUnlock(array: VPIArray) -> c_int;
    pub fn vpiArrayGetSize(array: VPIArray, size: *mut i32) -> c_int;
}

pub fn get_vpi_image_format_u8() -> u64 {
    VPIImageFormat::VPI_IMAGE_FORMAT_U8 as u64
}

pub fn get_vpi_backend_cuda() -> u64 {
    VPIBackend::VPI_BACKEND_CUDA as u64
}

pub fn get_vpi_backend_vic() -> u64 {
    VPIBackend::VPI_BACKEND_VIC as u64
}

pub fn get_vpi_backend_cpu() -> u64 {
    VPIBackend::VPI_BACKEND_CPU as u64
}

pub fn get_vpi_array_type_apriltag_detection() -> i32 {
    VPIArrayType::VPI_ARRAY_TYPE_APRILTAG_DETECTION as i32
}

#[cfg(not(feature = "gpu"))]
extern "C" {
}
