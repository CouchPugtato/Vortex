use anyhow::{Result, anyhow};
#[cfg(feature = "gpu")]
use crate::vpi;
#[cfg(feature = "gpu")]
use crate::vpi::{VPIStream, VPIPayload, VPIImage, VPIArray, VPIWarpMap, VPICameraIntrinsic, VPIPolynomialLensDistortionModel, VPIInterpolationType, VPIWarpMapType};
#[cfg(not(feature = "gpu"))]
type VPIStream = *mut std::ffi::c_void;
#[cfg(not(feature = "gpu"))]
type VPIPayload = *mut std::ffi::c_void;
#[cfg(not(feature = "gpu"))]
type VPIArray = *mut std::ffi::c_void;
#[cfg(not(feature = "gpu"))]
type VPIImage = *mut std::ffi::c_void;
#[cfg(not(feature = "gpu"))]
type VPIWarpMap = *mut std::ffi::c_void;

#[cfg(feature = "gpu")]
use std::ffi::c_void;
use crate::detector::RawDetection;
use crate::config::CameraConfig;

pub struct GpuDetector {
    stream: VPIStream,
    payload: VPIPayload,
    output_array: VPIArray,
    width: i32,
    height: i32,
    vpi_version_major: u32,
    
    // lens distortion & scaling
    warp_map: VPIWarpMap,
    rectified_image: VPIImage,
    
    // CUDA enhancement
    hist_eq_payload: VPIPayload,
    enhanced_image: VPIImage, // intermediate buffer: rectified -> enhanced

    scaled_width: i32,
    scaled_height: i32,
    pub scaled_config: CameraConfig, // exposed for pose estimation in main
}

impl GpuDetector {
    #[cfg(feature = "gpu")]
    pub fn new(width: usize, height: usize, config: &CameraConfig, scale_factor: f32) -> Result<Self> {
        unsafe {
            use std::io::Write;
            println!("DEBUG: Initializing GpuDetector..."); std::io::stdout().flush().ok();

            // vpi 3.x returns version as decimal encoding (3020400 -> 3.2.4)
            let vpi_version_raw = vpi::vpiGetVersion();
            let vpi_version = vpi_version_raw as u32;
            
            // decimal parsing: 3020400 -> 3.02.04
            let v_major = vpi_version / 1000000;
            let v_minor = (vpi_version / 10000) % 100;
            let v_patch = vpi_version % 10000;
            
            println!("DEBUG: VPI Version Detected: {}.{}.{} (Raw: {})", v_major, v_minor, v_patch, vpi_version); 
            std::io::stdout().flush().ok();

            let mut stream: VPIStream = std::ptr::null_mut();
            println!("DEBUG: Calling vpiStreamCreate(0)..."); std::io::stdout().flush().ok();
            if vpi::vpiStreamCreate(0, &mut stream) != 0 {
                return Err(anyhow!("Failed to create VPI Stream"));
            }
            println!("DEBUG: vpiStreamCreate success. Stream: {:?}", stream); std::io::stdout().flush().ok();

            let mut payload: VPIPayload = std::ptr::null_mut();
            
            println!("DEBUG: Calling vpiCreateAprilTagDetector..."); std::io::stdout().flush().ok();

            let backend = vpi::get_vpi_backend_cpu(); 
            // let backend = vpi::get_vpi_backend_cuda();

            // null params for defaults (avoids abi mismatch)
            let status = vpi::vpiCreateAprilTagDetector(
                backend, 
                width as i32, 
                height as i32, 
                std::ptr::null(), // &params, 
                &mut payload
            );

            if status != 0 {
                let msg = vpi::vpiGetLastStatusMessage();
                if !msg.is_null() {
                    let s = CStr::from_ptr(msg);
                    println!("DEBUG: vpiCreateAprilTagDetector failed with status {}: {:?}", status, s);
                } else {
                    println!("DEBUG: vpiCreateAprilTagDetector failed with status {}", status);
                }
                std::io::stdout().flush().ok();
                vpi::vpiStreamDestroy(stream);
                return Err(anyhow!("Failed to create AprilTag detector, status: {}", status));
            }
            println!("DEBUG: vpiCreateAprilTagDetector success. Payload: {:?}", payload); std::io::stdout().flush().ok();

            // optimizing lens distortion + downscaling
            // target resolution: 50% (960x540)
            // runs on vic (video image compositor)
            let scale_factor = 0.5;
            let scaled_width = (width as f32 * scale_factor) as i32;
            let scaled_height = (height as f32 * scale_factor) as i32;
            
            println!("DEBUG: Configuring VIC Pipeline. Input: {}x{} -> Output: {}x{}", width, height, scaled_width, scaled_height);

            // 1. create rectified image
            let mut rectified_image: VPIImage = std::ptr::null_mut();
            let format_u8 = vpi::get_vpi_image_format_u8();
            if vpi::vpiImageCreate(scaled_width, scaled_height, format_u8, 0, &mut rectified_image) != 0 {
                 return Err(anyhow!("Failed to create rectified image buffer"));
            }

            // 2. prepare intrinsics
            let kin = VPICameraIntrinsic {
                matrix: [
                    [config.fx as f32, 0.0, config.cx as f32],
                    [0.0, config.fy as f32, config.cy as f32]
                ]
            };

            // focal length and principal point scale with resolution
            let kout = VPICameraIntrinsic {
                matrix: [
                    [config.fx as f32 * scale_factor, 0.0, config.cx as f32 * scale_factor],
                    [0.0, config.fy as f32 * scale_factor, config.cy as f32 * scale_factor]
                ]
            };
            
            // 4. distortion model
            let dist = VPIPolynomialLensDistortionModel {
                k1: config.k1 as f32,
                k2: config.k2 as f32,
                p1: config.p1 as f32,
                p2: config.p2 as f32,
                k3: config.k3 as f32,
                k4: 0.0,
                k5: 0.0,
                k6: 0.0,
            };

            // 5. generate warp map
            let mut warp_map: VPIWarpMap = std::ptr::null_mut();
            if vpi::vpiWarpMapCreate(0, scaled_width, scaled_height, VPIWarpMapType::VPI_WARP_MAP_GRID_ABSOLUTE as i32, &mut warp_map) != 0 {
                return Err(anyhow!("Failed to create Warp Map"));
            }

            println!("DEBUG: Generating Warp Map on CPU...");
            if vpi::vpiWarpMapGenerateFromPolynomialLensDistortionModel(
                warp_map,
                kin,
                kout,
                &dist
            ) != 0 {
                return Err(anyhow!("Failed to generate Warp Map"));
            }
            println!("DEBUG: Warp Map Generated.");

            // 6. histogram equalization setup
            // create buffer for enhanced image
            let mut enhanced_image: VPIImage = std::ptr::null_mut();
            if vpi::vpiImageCreate(scaled_width, scaled_height, format_u8, 0, &mut enhanced_image) != 0 {
                return Err(anyhow!("Failed to create enhanced image buffer"));
            }

            let mut hist_eq_payload: VPIPayload = std::ptr::null_mut();
            if vpi::vpiCreateHistogramEqualizer(0, &mut hist_eq_payload) != 0 {
                return Err(anyhow!("Failed to create Histogram Equalizer"));
            }
            println!("DEBUG: Histogram Equalizer Created.");

            // 7. create scaled config for pose estimation
            let mut scaled_config = config.clone();
            scaled_config.fx *= scale_factor as f64;
            scaled_config.fy *= scale_factor as f64;
            scaled_config.cx *= scale_factor as f64;
            scaled_config.cy *= scale_factor as f64;
            scaled_config.k1 = 0.0; scaled_config.k2 = 0.0; scaled_config.p1 = 0.0;
            scaled_config.p2 = 0.0; scaled_config.k3 = 0.0; scaled_config.k4 = 0.0;
            scaled_config.k5 = 0.0; scaled_config.k6 = 0.0;

            let mut output_array: VPIArray = std::ptr::null_mut();
            println!("DEBUG: Calling vpiArrayCreate..."); std::io::stdout().flush().ok();
            let array_type = vpi::VPIArrayType::VPI_ARRAY_TYPE_APRILTAG_DETECTION as i32;
            let status = vpi::vpiArrayCreate(100, array_type, 0, &mut output_array);
            if status != 0 {
                let msg = vpi::vpiGetLastStatusMessage();
                if !msg.is_null() {
                    let s = CStr::from_ptr(msg);
                    println!("DEBUG: vpiArrayCreate failed with status {}: {:?}", status, s);
                } else {
                    println!("DEBUG: vpiArrayCreate failed with status {}", status);
                }
                std::io::stdout().flush().ok();
                vpi::vpiPayloadDestroy(payload);
                vpi::vpiStreamDestroy(stream);
                return Err(anyhow!("Failed to create Output Array"));
            }
            println!("DEBUG: vpiArrayCreate success. Array: {:?}", output_array); std::io::stdout().flush().ok();

            Ok(Self {
                stream,
                payload,
                output_array,
                width: width as i32,
                height: height as i32,
                vpi_version_major: v_major,
                warp_map,
                rectified_image,
                scaled_width,
                scaled_height,
                hist_eq_payload,
                enhanced_image,
                scaled_config,
            })
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn new(_width: usize, _height: usize, _config: &CameraConfig, _scale_factor: f32) -> Result<Self> {
        Err(anyhow!("GPU feature not enabled"))
    }

    #[cfg(feature = "gpu")]
    pub fn detect(&mut self, image_data: &[u8], width: usize, height: usize) -> Result<Vec<RawDetection>> {
        if width as i32 != self.width || height as i32 != self.height {
            return Err(anyhow!("Resolution changed, GpuDetector needs recreation"));
        }

        unsafe {
            // wrap cpu buffer into vpiimage assumes contiguous 1-byte gray data
            let mut img: VPIImage = std::ptr::null_mut();
            
            let mut pitches = [0i32; 3];
            pitches[0] = width as i32;
            
            let mut planes = [std::ptr::null_mut(); 3];
            planes[0] = image_data.as_ptr() as *mut c_void;

            let buffer_data = vpi::VPIImageData {
                bufferType: 1, // VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR
                buffer: vpi::VPIImageBuffer {
                    pitch: vpi::VPIImageBufferPitch {
                        planes,
                        pitches,
                    }
                }
            };

            // create image wrapper
            // explicit format u8 (grayscale)
            let format_u8 = vpi::get_vpi_image_format_u8();
            let image_params = vpi::VPIImageCreateParams {
                width: self.width as i32,
                height: self.height as i32,
                format: format_u8,
            };

            let status = vpi::vpiImageCreateWrapper(&buffer_data, &image_params as *const _ as *const std::ffi::c_void, 0, &mut img);
            if status != 0 {
                return Err(anyhow!("Failed to wrap image"));
            }

            // optimization: vic undistort + downscale
            // remap: cpu -> vic (rectified)
            // vic backend
            let vic_backend = vpi::get_vpi_backend_vic();
            
            // let _lock = VIC_LOCK.lock();
            let status = vpi::vpiSubmitRemap(
                self.stream,
                vic_backend,
                self.warp_map,
                img,
                self.rectified_image,
                VPIInterpolationType::VPI_INTERPOLATION_LINEAR as i32,
                0, // VPI_BOUNDARY_COND_ZERO
                std::ptr::null()
            );

            if status != 0 {
                println!("DEBUG: vpiSubmitRemap failed with status {}", status);
                vpi::vpiImageDestroy(img);
                return Err(anyhow!("Failed to submit remap"));
            }

            let cuda_backend = vpi::get_vpi_backend_cuda();
            let status = vpi::vpiSubmitHistogramEqualization(
                self.stream,
                cuda_backend,
                self.hist_eq_payload,
                self.rectified_image,
                self.enhanced_image
            );
            
            if status != 0 {
                // fallback to CPU if CUDA fails
                println!("DEBUG: vpiSubmitHistogramEqualization (CUDA) failed with status {}. Trying CPU.", status);
                let cpu_backend = vpi::get_vpi_backend_cpu();
                let status = vpi::vpiSubmitHistogramEqualization(
                    self.stream,
                    cpu_backend,
                    self.hist_eq_payload,
                    self.rectified_image,
                    self.enhanced_image
                );
                if status != 0 {
                    return Err(anyhow!("Failed to submit histogram equalization"));
                }
            }

            // submit detection
            let backend = vpi::get_vpi_backend_cpu(); 
            // let backend = vpi::get_vpi_backend_cuda();
            
            let status = vpi::vpiSubmitAprilTagDetector(
                self.stream,
                backend,
                self.payload,
                100, // maxDetections
                self.enhanced_image, // Use enhanced image
                self.output_array
            );
            
            if status != 0 {
                println!("DEBUG: vpiSubmitAprilTagDetector failed with status {}", status);
                vpi::vpiImageDestroy(img);
                return Err(anyhow!("Failed to submit detection"));
            }

            let status = vpi::vpiStreamSync(self.stream);
            if status != 0 {
                println!("DEBUG: vpiStreamSync failed with status {}", status);
                vpi::vpiImageDestroy(img);
                return Err(anyhow!("Failed to sync stream"));
            }

            let mut size: i32 = 0;
            vpi::vpiArrayGetSize(self.output_array, &mut size);
            if size > 0 {
                 println!("DEBUG: vpiArrayGetSize returned {}", size);
            }
            
            let mut detections = Vec::new();
            if size > 0 {
                let mut stride: i32 = 0;
                let mut data_ptr: *mut c_void = std::ptr::null_mut();
                // VPI_LOCK_READ = 1
                let lock_status = vpi::vpiArrayLockData(self.output_array, 1, &mut stride, &mut data_ptr);
                
                if lock_status == 0 && !data_ptr.is_null() {
                    println!("DEBUG: Locked array, stride={}, ptr={:?}", stride, data_ptr);
                    for i in 0..size {
                        let element_ptr = (data_ptr as *const u8).add((i * stride) as usize) as *const vpi::VPIAprilTagDetection;
                        let det = &*element_ptr;
                        
                        let center = [det.center.x as f64, det.center.y as f64];
                        let corners = [
                            [det.corners[0].x as f64, det.corners[0].y as f64],
                            [det.corners[1].x as f64, det.corners[1].y as f64],
                            [det.corners[2].x as f64, det.corners[2].y as f64],
                            [det.corners[3].x as f64, det.corners[3].y as f64],
                        ];
                        
                        detections.push(RawDetection {
                            id: det.id as usize,
                            center,
                            corners,
                        });
                    }
                    vpi::vpiArrayUnlock(self.output_array);
                }
            }

            vpi::vpiImageDestroy(img);
            Ok(detections)
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn detect(&mut self, _image_data: &[u8], _width: usize, _height: usize) -> Result<Vec<RawDetection>> {
        Err(anyhow!("GPU feature not enabled"))
    }
}

#[cfg(feature = "gpu")]
impl Drop for GpuDetector {
    fn drop(&mut self) {
        unsafe {
            // clean up vpi resources
            if !self.output_array.is_null() { vpi::vpiArrayDestroy(self.output_array); }
            if !self.payload.is_null() { vpi::vpiPayloadDestroy(self.payload); }
            if !self.hist_eq_payload.is_null() { vpi::vpiPayloadDestroy(self.hist_eq_payload); }
            
            if !self.warp_map.is_null() { vpi::vpiWarpMapDestroy(self.warp_map); }
            if !self.rectified_image.is_null() { vpi::vpiImageDestroy(self.rectified_image); }
            if !self.enhanced_image.is_null() { vpi::vpiImageDestroy(self.enhanced_image); }
            
            if !self.stream.is_null() { vpi::vpiStreamDestroy(self.stream); }
        }
    }
}
