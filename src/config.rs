use serde::Deserialize;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Clone, Copy, Deserialize)]
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

impl CameraConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }
}
