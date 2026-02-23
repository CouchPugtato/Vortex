use anyhow::{anyhow, Result};
use crate::detector::{Detection, YoloDetection};

#[cfg(feature = "tensorrt")]
mod trt {
    use std::ffi::{c_char, c_int, c_void};

    #[repr(C)]
    pub struct YoloTrtDims {
        pub nb_dims: c_int,
        pub dims: [c_int; 8],
    }

    extern "C" {
        pub fn yolo_trt_create(
            engine_path: *const c_char,
            input_dims: *mut YoloTrtDims,
            output_dims: *mut YoloTrtDims,
        ) -> *mut c_void;

        pub fn yolo_trt_infer(
            handle: *mut c_void,
            input: *const f32,
            output: *mut f32,
            output_len: usize,
        ) -> c_int;

        pub fn yolo_trt_destroy(handle: *mut c_void);
    }
}

pub struct YoloDetector {
    #[cfg(feature = "tensorrt")]
    handle: *mut std::ffi::c_void,
    #[cfg(feature = "tensorrt")]
    input_w: usize,
    #[cfg(feature = "tensorrt")]
    input_h: usize,
    #[cfg(feature = "tensorrt")]
    input_c: usize,
    #[cfg(feature = "tensorrt")]
    output_len: usize,
    #[cfg(feature = "tensorrt")]
    output_dims: Vec<usize>,
    #[cfg(feature = "tensorrt")]
    class_names: Vec<String>,
    #[cfg(feature = "tensorrt")]
    conf_thresh: f32,
    #[cfg(feature = "tensorrt")]
    iou_thresh: f32,
}

impl YoloDetector {
    #[cfg(feature = "tensorrt")]
    pub fn new() -> Result<Self> {
        use std::ffi::CString;
        use std::path::Path;

        let engine_path = std::env::var("YOLO_ENGINE")
            .unwrap_or_else(|_| "models/rockpaperscizzors.engine".to_string());

        if !Path::new(&engine_path).exists() {
            return Err(anyhow!("YOLO engine not found at {}. Build it from the ONNX model first.", engine_path));
        }

        let class_names = std::env::var("YOLO_CLASSES")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec!["robot".to_string()]);

        let conf_thresh = std::env::var("YOLO_CONF")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.25);

        let iou_thresh = std::env::var("YOLO_IOU")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.45);

        let mut input_dims = trt::YoloTrtDims { nb_dims: 0, dims: [0; 8] };
        let mut output_dims = trt::YoloTrtDims { nb_dims: 0, dims: [0; 8] };

        let c_path = CString::new(engine_path)?;
        let handle = unsafe { trt::yolo_trt_create(c_path.as_ptr(), &mut input_dims, &mut output_dims) };
        if handle.is_null() {
            return Err(anyhow!("Failed to create TensorRT YOLO engine"));
        }

        let input_dims_vec = dims_to_vec(&input_dims)?;
        let output_dims_vec = dims_to_vec(&output_dims)?;

        let (input_c, input_h, input_w) = match input_dims_vec.as_slice() {
            [1, c, h, w] => (*c, *h, *w),
            [c, h, w] => (*c, *h, *w),
            _ => return Err(anyhow!("Unsupported input dims from engine: {:?}", input_dims_vec)),
        };

        let output_len = output_dims_vec.iter().product::<usize>();

        println!(
            "YOLO Detector Initialized: input={}x{}x{} output_len={}",
            input_c, input_h, input_w, output_len
        );

        Ok(Self {
            handle,
            input_w,
            input_h,
            input_c,
            output_len,
            output_dims: output_dims_vec,
            class_names,
            conf_thresh,
            iou_thresh,
        })
    }

    #[cfg(not(feature = "tensorrt"))]
    pub fn new() -> Result<Self> {
        Err(anyhow!("TensorRT feature not enabled"))
    }

    #[cfg(feature = "tensorrt")]
    pub fn detect(&mut self, data: &[u8], width: usize, height: usize) -> Result<Vec<Detection>> {
        let (input, scale, pad_x, pad_y) = preprocess_letterbox_gray_to_chw(
            data,
            width,
            height,
            self.input_w,
            self.input_h,
            self.input_c,
        )?;

        let mut output = vec![0f32; self.output_len];
        let status = unsafe { trt::yolo_trt_infer(self.handle, input.as_ptr(), output.as_mut_ptr(), output.len()) };
        if status != 0 {
            return Err(anyhow!("TensorRT inference failed with status {}", status));
        }

        let dets = postprocess_yolo(
            &output,
            &self.output_dims,
            self.input_w,
            self.input_h,
            width,
            height,
            scale,
            pad_x,
            pad_y,
            &self.class_names,
            self.conf_thresh,
            self.iou_thresh,
        );

        Ok(dets.into_iter().map(Detection::Yolo).collect())
    }

    #[cfg(not(feature = "tensorrt"))]
    pub fn detect(&mut self, _data: &[u8], _width: usize, _height: usize) -> Result<Vec<Detection>> {
        Err(anyhow!("TensorRT feature not enabled"))
    }
}

#[cfg(feature = "tensorrt")]
impl Drop for YoloDetector {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                trt::yolo_trt_destroy(self.handle);
            }
        }
    }
}

#[cfg(feature = "tensorrt")]
fn dims_to_vec(dims: &trt::YoloTrtDims) -> Result<Vec<usize>> {
    if dims.nb_dims <= 0 || dims.nb_dims as usize > dims.dims.len() {
        return Err(anyhow!("Invalid dims from TensorRT: nb_dims={}", dims.nb_dims));
    }
    let mut v = Vec::with_capacity(dims.nb_dims as usize);
    for i in 0..dims.nb_dims as usize {
        let d = dims.dims[i];
        if d <= 0 {
            return Err(anyhow!("Invalid dimension value {}", d));
        }
        v.push(d as usize);
    }
    Ok(v)
}

#[cfg(feature = "tensorrt")]
fn preprocess_letterbox_gray_to_chw(
    data: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    dst_c: usize,
) -> Result<(Vec<f32>, f32, f32, f32)> {
    if dst_c != 3 {
        return Err(anyhow!("YOLO input must be 3-channel (got {})", dst_c));
    }
    if data.len() < src_w * src_h {
        return Err(anyhow!("Input buffer too small"));
    }

    let scale = (dst_w as f32 / src_w as f32)
        .min(dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round().max(1.0) as usize;
    let new_h = (src_h as f32 * scale).round().max(1.0) as usize;
    let pad_x = ((dst_w - new_w) / 2) as f32;
    let pad_y = ((dst_h - new_h) / 2) as f32;

    let mut out = vec![0f32; dst_c * dst_w * dst_h];

    for y in 0..new_h {
        let src_y = (y as f32 / scale).min((src_h - 1) as f32);
        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let ly = src_y - y0 as f32;

        for x in 0..new_w {
            let src_x = (x as f32 / scale).min((src_w - 1) as f32);
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let lx = src_x - x0 as f32;

            let p00 = data[y0 * src_w + x0] as f32;
            let p01 = data[y0 * src_w + x1] as f32;
            let p10 = data[y1 * src_w + x0] as f32;
            let p11 = data[y1 * src_w + x1] as f32;

            let top = p00 + (p01 - p00) * lx;
            let bottom = p10 + (p11 - p10) * lx;
            let value = (top + (bottom - top) * ly) / 255.0;

            let dst_x = x + pad_x as usize;
            let dst_y = y + pad_y as usize;
            let idx = dst_y * dst_w + dst_x;

            out[idx] = value;
            out[dst_w * dst_h + idx] = value;
            out[2 * dst_w * dst_h + idx] = value;
        }
    }

    Ok((out, scale, pad_x, pad_y))
}

#[cfg(feature = "tensorrt")]
fn postprocess_yolo(
    output: &[f32],
    output_dims: &[usize],
    input_w: usize,
    input_h: usize,
    orig_w: usize,
    orig_h: usize,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    class_names: &[String],
    conf_thresh: f32,
    iou_thresh: f32,
) -> Vec<YoloDetection> {
    let num_classes = class_names.len().max(1);
    let attrs_a = 4 + num_classes;
    let attrs_b = 5 + num_classes;

    let (num_boxes, attrs, channels_first) = match output_dims {
        [_, d1, d2] => {
            if *d1 == attrs_a || *d1 == attrs_b {
                (*d2, *d1, true)
            } else if *d2 == attrs_a || *d2 == attrs_b {
                (*d1, *d2, false)
            } else {
                (d2.min(d1).to_owned(), d1.max(d2).to_owned(), true)
            }
        }
        [d1, d2] => (*d1, *d2, false),
        _ => {
            return Vec::new();
        }
    };

    let mut use_sigmoid = false;
    for &v in output.iter().take(1024) {
        if v < 0.0 || v > 1.0 {
            use_sigmoid = true;
            break;
        }
    }

    let mut candidates = Vec::new();
    for i in 0..num_boxes {
        let mut get = |k: usize| -> f32 {
            if channels_first {
                let idx = k * num_boxes + i;
                output.get(idx).copied().unwrap_or(0.0)
            } else {
                let idx = i * attrs + k;
                output.get(idx).copied().unwrap_or(0.0)
            }
        };

        let mut x = get(0);
        let mut y = get(1);
        let mut w = get(2);
        let mut h = get(3);

        let mut obj = 1.0f32;
        let mut class_start = 4;
        if attrs == attrs_b {
            obj = get(4);
            class_start = 5;
        }

        if use_sigmoid {
            obj = sigmoid(obj);
        }

        let mut best_class = 0usize;
        let mut best_score = 0f32;
        for c in 0..num_classes {
            let mut score = get(class_start + c);
            if use_sigmoid {
                score = sigmoid(score);
            }
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }

        let conf = best_score * obj;
        if conf < conf_thresh {
            continue;
        }

        if x <= 1.0 && w <= 1.0 {
            x *= input_w as f32;
            w *= input_w as f32;
        }
        if y <= 1.0 && h <= 1.0 {
            y *= input_h as f32;
            h *= input_h as f32;
        }

        let mut left = x - w / 2.0;
        let mut top = y - h / 2.0;

        left = (left - pad_x) / scale;
        top = (top - pad_y) / scale;
        let mut box_w = w / scale;
        let mut box_h = h / scale;

        if left < 0.0 { left = 0.0; }
        if top < 0.0 { top = 0.0; }
        if left + box_w > orig_w as f32 { box_w = (orig_w as f32 - left).max(0.0); }
        if top + box_h > orig_h as f32 { box_h = (orig_h as f32 - top).max(0.0); }

        candidates.push((
            conf,
            YoloDetection {
                class_name: class_names.get(best_class).cloned().unwrap_or_else(|| "object".to_string()),
                confidence: conf as f64,
                bbox: [left as f64, top as f64, box_w as f64, box_h as f64],
            },
        ));
    }

    nms(candidates, iou_thresh)
}

#[cfg(feature = "tensorrt")]
fn nms(mut dets: Vec<(f32, YoloDetection)>, iou_thresh: f32) -> Vec<YoloDetection> {
    dets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<YoloDetection> = Vec::new();

    for (_, det) in dets.into_iter() {
        let mut suppressed = false;
        for kept in &keep {
            if iou(&det.bbox, &kept.bbox) > iou_thresh as f64 {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            keep.push(det);
        }
    }
    keep
}

#[cfg(feature = "tensorrt")]
fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let (ax1, ay1, aw, ah) = (a[0], a[1], a[2], a[3]);
    let (bx1, by1, bw, bh) = (b[0], b[1], b[2], b[3]);
    let ax2 = ax1 + aw;
    let ay2 = ay1 + ah;
    let bx2 = bx1 + bw;
    let by2 = by1 + bh;

    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);
    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;
    let area_a = aw * ah;
    let area_b = bw * bh;
    if area_a + area_b - inter_area <= 0.0 {
        return 0.0;
    }
    inter_area / (area_a + area_b - inter_area)
}

#[cfg(feature = "tensorrt")]
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
