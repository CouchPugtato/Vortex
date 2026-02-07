use image::{GrayImage, Luma};

pub fn contrast_stretch(img: &GrayImage) -> GrayImage {
    let mut min_val = 255u8;
    let mut max_val = 0u8;

    for p in img.pixels() {
        let val = p.0[0];
        if val < min_val { min_val = val; }
        if val > max_val { max_val = val; }
    }

    if min_val >= max_val {
        return img.clone();
    }

    let mut out = GrayImage::new(img.width(), img.height());
    let range = (max_val - min_val) as f32;
    let scale = 255.0 / range;

    for (x, y, p) in img.enumerate_pixels() {
        let val = p.0[0];
        let new_val = ((val as f32 - min_val as f32) * scale).clamp(0.0, 255.0) as u8;
        out.put_pixel(x, y, Luma([new_val]));
    }
    out
}
