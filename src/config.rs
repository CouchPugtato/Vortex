use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Serialize};

fn default_smoothing_alpha() -> f64 { 0.1 }
fn default_resolution_scale_factor() -> f32 { 1.0 }
fn default_black_level_offset() -> f64 { 7.0 }
fn default_sensor_gain() -> f64 { 2.0 }
fn default_red_balance() -> f64 { 1200.0 }
fn default_blue_balance() -> f64 { 1976.0 }
fn default_yolo_obj_width_m() -> f64 { 0.3 }
fn default_yolo_obj_height_m() -> f64 { 0.3 }
fn default_confidence_threshold() -> f64 { 0.25 }
fn default_use_nn() -> bool { true }

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct CameraConfig {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
    pub tag_size_m: f64,

    // camera to robot center
    #[serde(default)]
    pub x_offset: f64,
    #[serde(default)]
    pub y_offset: f64,
    #[serde(default)]
    pub z_offset: f64,

    #[serde(default)]
    pub pitch_deg: f64,
    #[serde(default)]
    pub yaw_deg: f64,
    #[serde(default)]
    pub roll_deg: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct ProcessingConfig {
    #[serde(default = "default_smoothing_alpha")]
    pub smoothing_alpha: f64,
    #[serde(default = "default_resolution_scale_factor", alias = "gpu_scale_factor")]
    pub resolution_scale_factor: f32,

    #[serde(default = "default_black_level_offset")]
    pub black_level_offset: f64,
    #[serde(default = "default_sensor_gain")]
    pub sensor_gain: f64,
    #[serde(default = "default_red_balance")]
    pub red_balance: f64,
    #[serde(default = "default_blue_balance")]
    pub blue_balance: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct ObjectDetectionConfig {
    #[serde(default = "default_use_nn")]
    pub use_nn: bool,
    #[serde(default = "default_yolo_obj_width_m", alias = "processing_yolo_obj_width_m")]
    pub yolo_obj_width_m: f64,
    #[serde(default = "default_yolo_obj_height_m", alias = "processing_yolo_obj_height_m")]
    pub yolo_obj_height_m: f64,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct RuntimeConfig {
    pub camera: CameraConfig,
    #[serde(default)]
    pub processing: ProcessingConfig,
    #[serde(default)]
    pub object_detection: ObjectDetectionConfig,
}

impl RuntimeConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let value: serde_json::Value = serde_json::from_reader(reader)?;
        let mut config: Self = serde_json::from_value(value.clone())?;
        if value.get("object_detection").is_none() {
            if let Some(proc_cfg) = value.get("processing") {
                if let Some(v) = proc_cfg.get("yolo_obj_width_m").and_then(|x| x.as_f64()) {
                    config.object_detection.yolo_obj_width_m = v;
                }
                if let Some(v) = proc_cfg.get("yolo_obj_height_m").and_then(|x| x.as_f64()) {
                    config.object_detection.yolo_obj_height_m = v;
                }
            }
        }
        Ok(config)
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            smoothing_alpha: 0.1,
            resolution_scale_factor: 1.0,
            black_level_offset: 7.0,
            sensor_gain: 2.0,
            red_balance: 1200.0,
            blue_balance: 1976.0,
        }
    }
}

impl Default for ObjectDetectionConfig {
    fn default() -> Self {
        Self {
            use_nn: true,
            yolo_obj_width_m: 0.3,
            yolo_obj_height_m: 0.3,
            confidence_threshold: 0.25,
        }
    }
}
