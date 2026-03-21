use anyhow::{anyhow, Result};
#[cfg(feature = "gpu")]
use crate::vpi;
#[cfg(feature = "gpu")]
use crate::vpi::{VPIArray, VPIImage, VPIPayload, VPIStream};
#[cfg(not(feature = "gpu"))]
type VPIArray = *mut std::ffi::c_void;
#[cfg(not(feature = "gpu"))]
type VPIImage = *mut std::ffi::c_void;
#[cfg(not(feature = "gpu"))]
type VPIPayload = *mut std::ffi::c_void;
#[cfg(not(feature = "gpu"))]
type VPIStream = *mut std::ffi::c_void;

#[cfg(feature = "gpu")]
use std::ffi::{c_void, CStr};
use crate::config::CameraConfig;
use crate::detector::{AprilTagDetection, Detection};

pub struct GpuDetector {
    stream: VPIStream,
    payload: VPIPayload,
    output_array: VPIArray,
    width: i32,
    height: i32,
    pub scaled_config: CameraConfig,
}

impl GpuDetector {
    #[cfg(feature = "gpu")]
    pub fn new(width: usize, height: usize, config: &CameraConfig, scale_factor: f32) -> Result<Self> {
        unsafe {
            let mut stream: VPIStream = std::ptr::null_mut();
            if vpi::vpiStreamCreate(0, &mut stream) != 0 {
                return Err(anyhow!("Failed to create VPI stream"));
            }

            let mut payload: VPIPayload = std::ptr::null_mut();
            let status = vpi::vpiCreateAprilTagDetector(
                vpi::get_vpi_backend_cpu(),
                width as i32,
                height as i32,
                std::ptr::null(),
                &mut payload,
            );
            if status != 0 {
                let msg = vpi::vpiGetLastStatusMessage();
                if !msg.is_null() {
                    let s = CStr::from_ptr(msg);
                    eprintln!("vpiCreateAprilTagDetector failed: {:?}", s);
                }
                vpi::vpiStreamDestroy(stream);
                return Err(anyhow!("Failed to create AprilTag detector, status {}", status));
            }

            let mut output_array: VPIArray = std::ptr::null_mut();
            let array_type = vpi::VPIArrayType::VPI_ARRAY_TYPE_APRILTAG_DETECTION as i32;
            if vpi::vpiArrayCreate(100, array_type, 0, &mut output_array) != 0 {
                vpi::vpiPayloadDestroy(payload);
                vpi::vpiStreamDestroy(stream);
                return Err(anyhow!("Failed to create VPI output array"));
            }

            let scale = scale_factor.clamp(0.1, 1.0) as f64;
            let mut scaled_config = config.clone();
            scaled_config.fx *= scale;
            scaled_config.fy *= scale;
            scaled_config.cx *= scale;
            scaled_config.cy *= scale;
            scaled_config.k1 = 0.0;
            scaled_config.k2 = 0.0;
            scaled_config.p1 = 0.0;
            scaled_config.p2 = 0.0;
            scaled_config.k3 = 0.0;

            Ok(Self {
                stream,
                payload,
                output_array,
                width: width as i32,
                height: height as i32,
                scaled_config,
            })
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn new(_width: usize, _height: usize, _config: &CameraConfig, _scale_factor: f32) -> Result<Self> {
        Err(anyhow!("GPU feature not enabled"))
    }

    #[cfg(feature = "gpu")]
    pub fn detect(&mut self, image_data: &[u8], width: usize, height: usize) -> Result<Vec<Detection>> {
        if width as i32 != self.width || height as i32 != self.height {
            return Err(anyhow!("Resolution changed, recreate GpuDetector"));
        }
        unsafe {
            let mut img: VPIImage = std::ptr::null_mut();
            let mut pitches = [0i32; 3];
            pitches[0] = width as i32;
            let mut planes = [std::ptr::null_mut(); 3];
            planes[0] = image_data.as_ptr() as *mut c_void;
            let data = vpi::VPIImageData {
                bufferType: 1,
                buffer: vpi::VPIImageBuffer {
                    pitch: vpi::VPIImageBufferPitch { planes, pitches },
                },
            };
            let params = vpi::VPIImageCreateParams {
                width: self.width,
                height: self.height,
                format: vpi::get_vpi_image_format_u8(),
            };
            if vpi::vpiImageCreateWrapper(&data, &params as *const _ as *const std::ffi::c_void, 0, &mut img) != 0 {
                return Err(anyhow!("Failed to wrap image"));
            }

            let mut status = vpi::vpiSubmitAprilTagDetector(
                self.stream,
                vpi::get_vpi_backend_cuda(),
                self.payload,
                100,
                img,
                self.output_array,
            );
            if status != 0 {
                status = vpi::vpiSubmitAprilTagDetector(
                    self.stream,
                    vpi::get_vpi_backend_cpu(),
                    self.payload,
                    100,
                    img,
                    self.output_array,
                );
            }
            if status != 0 {
                vpi::vpiImageDestroy(img);
                return Err(anyhow!("Failed to submit AprilTag detector"));
            }
            if vpi::vpiStreamSync(self.stream) != 0 {
                vpi::vpiImageDestroy(img);
                return Err(anyhow!("Failed to sync VPI stream"));
            }

            let mut size: i32 = 0;
            vpi::vpiArrayGetSize(self.output_array, &mut size);
            let mut out = Vec::new();
            if size > 0 {
                let mut stride: i32 = 0;
                let mut data_ptr: *mut c_void = std::ptr::null_mut();
                if vpi::vpiArrayLockData(self.output_array, 1, &mut stride, &mut data_ptr) == 0
                    && !data_ptr.is_null()
                {
                    for i in 0..size {
                        let p =
                            (data_ptr as *const u8).add((i * stride) as usize) as *const vpi::VPIAprilTagDetection;
                        let d = &*p;
                        out.push(Detection::AprilTag(AprilTagDetection {
                            id: d.id as usize,
                            center: [d.center.x as f64, d.center.y as f64],
                            corners: [
                                [d.corners[0].x as f64, d.corners[0].y as f64],
                                [d.corners[1].x as f64, d.corners[1].y as f64],
                                [d.corners[2].x as f64, d.corners[2].y as f64],
                                [d.corners[3].x as f64, d.corners[3].y as f64],
                            ],
                            cpu_detection: None,
                        }));
                    }
                    vpi::vpiArrayUnlock(self.output_array);
                }
            }

            vpi::vpiImageDestroy(img);
            Ok(out)
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub fn detect(&mut self, _image_data: &[u8], _width: usize, _height: usize) -> Result<Vec<Detection>> {
        Err(anyhow!("GPU feature not enabled"))
    }
}

#[cfg(feature = "gpu")]
impl Drop for GpuDetector {
    fn drop(&mut self) {
        unsafe {
            if !self.output_array.is_null() {
                vpi::vpiArrayDestroy(self.output_array);
            }
            if !self.payload.is_null() {
                vpi::vpiPayloadDestroy(self.payload);
            }
            if !self.stream.is_null() {
                vpi::vpiStreamDestroy(self.stream);
            }
        }
    }
}
