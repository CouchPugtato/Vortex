use image::{RgbaImage, Rgba};

pub fn draw_detections(overlay: &mut RgbaImage, corners_list: &Vec<[(f64, f64); 4]>) {
    let color = Rgba([0, 255, 0, 255]);
    for corners in corners_list.iter() {
        let (x0, y0) = (corners[0].0.round() as i32, corners[0].1.round() as i32);
        let (x1, y1) = (corners[1].0.round() as i32, corners[1].1.round() as i32);
        let (x2, y2) = (corners[2].0.round() as i32, corners[2].1.round() as i32);
        let (x3, y3) = (corners[3].0.round() as i32, corners[3].1.round() as i32);
        draw_line_thick(overlay, x0, y0, x1, y1, color);
        draw_line_thick(overlay, x1, y1, x2, y2, color);
        draw_line_thick(overlay, x2, y2, x3, y3, color);
        draw_line_thick(overlay, x3, y3, x0, y0, color);
        draw_square(overlay, x0, y0, 3, Rgba([255, 255, 255, 255]));
        draw_square(overlay, x1, y1, 3, Rgba([255, 255, 255, 255]));
        draw_square(overlay, x2, y2, 3, Rgba([255, 255, 255, 255]));
        draw_square(overlay, x3, y3, 3, Rgba([255, 255, 255, 255]));
    }
}

fn draw_line_thick(img: &mut RgbaImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        paint_thick(img, x0, y0, color);
        if x0 == x1 && y0 == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x0 += sx; }
        if e2 <= dx { err += dx; y0 += sy; }
    }
}

fn paint_thick(img: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>) {
    for dy in -2..=2 {
        for dx in -2..=2 {
            let px = x + dx;
            let py = y + dy;
            if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}

fn draw_square(img: &mut RgbaImage, x: i32, y: i32, radius: i32, color: Rgba<u8>) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = x + dx;
            let py = y + dy;
            if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}