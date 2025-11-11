use image::{GrayImage};
use image::imageops::filter3x3;

pub fn contrast_stretch(gray: &GrayImage) -> GrayImage {
    let data = gray.as_raw();
    let (mut min_v, mut max_v) = (u8::MAX, u8::MIN);
    for &v in data.iter() {
        if v < min_v { min_v = v; }
        if v > max_v { max_v = v; }
    }
    if max_v <= min_v { return gray.clone(); }
    let range = (max_v - min_v) as f32;
    let mut out = Vec::with_capacity(data.len());
    for &v in data.iter() {
        let nv = ((v.saturating_sub(min_v) as f32) * 255.0 / range).round() as u8;
        out.push(nv);
    }
    GrayImage::from_raw(gray.width(), gray.height(), out).unwrap_or_else(|| gray.clone())
}

pub fn gamma_correct(gray: &GrayImage, gamma: f32) -> GrayImage {
    if gamma <= 0.0 { return gray.clone(); }
    let inv = 1.0 / gamma;
    let mut out = Vec::with_capacity(gray.as_raw().len());
    for &v in gray.as_raw().iter() {
        let nf = (v as f32) / 255.0;
        let cf = nf.powf(inv);
        let nv = (cf * 255.0).round().clamp(0.0, 255.0) as u8;
        out.push(nv);
    }
    GrayImage::from_raw(gray.width(), gray.height(), out).unwrap_or_else(|| gray.clone())
}

pub fn linear_contrast(gray: &GrayImage, gain: f32) -> GrayImage {
    let mut out = Vec::with_capacity(gray.as_raw().len());
    for &v in gray.as_raw().iter() {
        let nv = ((v as f32 - 127.5) * (1.0 + gain) + 127.5).round().clamp(0.0, 255.0) as u8;
        out.push(nv);
    }
    GrayImage::from_raw(gray.width(), gray.height(), out).unwrap_or_else(|| gray.clone())
}

pub fn sharpen3x3(gray: &GrayImage) -> GrayImage {
    let kernel: [f32; 9] = [
        0.0, -1.0, 0.0,
        -1.0, 5.0, -1.0,
        0.0, -1.0, 0.0,
    ];
    filter3x3(gray, &kernel)
}