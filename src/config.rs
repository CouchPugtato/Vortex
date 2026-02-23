use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Serialize};

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
    #[serde(
        default = "default_resolution_scale_factor",
        alias = "gpu_scale_factor"
    )]
    pub resolution_scale_factor: f32,
    #[serde(default = "default_yolo_obj_width_m")]
    pub yolo_obj_width_m: f64,
    #[serde(default = "default_yolo_obj_height_m")]
    pub yolo_obj_height_m: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct RuntimeConfig {
    pub camera: CameraConfig,
    #[serde(default)]
    pub processing: ProcessingConfig,
}

impl RuntimeConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            smoothing_alpha: 0.1,
            resolution_scale_factor: 1.0,
            yolo_obj_width_m: 0.3,
            yolo_obj_height_m: 0.3,
        }
    }
}

