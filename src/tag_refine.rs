//! Image alignment for small tags whose corner-only planar pose is ambiguous.
use crate::{
    config::CameraConfig,
    pose::{self, Pose, PoseEstimate},
};
use image::{GrayImage, Luma};
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

const CELL_PIXELS: usize = 32;
const TEMPLATE_SIDE: usize = 10 * CELL_PIXELS;

pub struct TagPatch {
    image: GrayImage,
    origin: [usize; 2],
    samples: Vec<[usize; 2]>,
    template: GrayImage,
    black: f64,
    contrast: f64,
    pixels_per_template_pixel: f64,
    bits: Vec<bool>,
    noise_floor: f64,
}

impl std::fmt::Debug for TagPatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TagPatch")
            .field("width", &self.image.width())
            .field("height", &self.image.height())
            .field("samples", &self.samples.len())
            .finish_non_exhaustive()
    }
}

impl TagPatch {
    pub fn new(data: &[u8], width: usize, height: usize, corners: [[f64; 2]; 4]) -> Option<Self> {
        Self::with_pattern(data, width, height, corners, None)
    }

    pub fn with_pattern(
        data: &[u8],
        width: usize,
        height: usize,
        corners: [[f64; 2]; 4],
        pattern: Option<&[bool]>,
    ) -> Option<Self> {
        if pattern.is_some_and(|bits| bits.len() != 64) {
            return None;
        }
        if width < 2 || height < 2 || width.checked_mul(height) != Some(data.len()) {
            return None;
        }
        let lengths: Vec<_> = (0..4)
            .map(|i| {
                (corners[i][0] - corners[(i + 1) % 4][0])
                    .hypot(corners[i][1] - corners[(i + 1) % 4][1])
            })
            .collect();
        let side = lengths.iter().sum::<f64>() / 4.0;
        if !(7.0..=40.0).contains(&side) {
            return None;
        }
        let lo = [
            corners
                .iter()
                .map(|p| p[0])
                .fold(f64::INFINITY, f64::min)
                .floor() as isize
                - 6,
            corners
                .iter()
                .map(|p| p[1])
                .fold(f64::INFINITY, f64::min)
                .floor() as isize
                - 6,
        ];
        let hi = [
            corners
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil() as isize
                + 6,
            corners
                .iter()
                .map(|p| p[1])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil() as isize
                + 6,
        ];
        if lo[0] < 0 || lo[1] < 0 || hi[0] >= width as isize || hi[1] >= height as isize {
            return None;
        }
        let origin = [lo[0] as usize, lo[1] as usize];
        let mut image = GrayImage::new((hi[0] - lo[0] + 1) as u32, (hi[1] - lo[1] + 1) as u32);
        let mut samples = Vec::new();
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let point = [
                (origin[0] + x as usize) as f64,
                (origin[1] + y as usize) as f64,
            ];
            pixel[0] = data[(origin[1] + y as usize) * width + origin[0] + x as usize];
            if (0..4).all(|i| {
                let a = corners[i];
                let b = corners[(i + 1) % 4];
                let d = [b[0] - a[0], b[1] - a[1]];
                (-d[1] * (point[0] - a[0]) + d[0] * (point[1] - a[1])) / lengths[i] <= 1.0
            }) {
                samples.push([x as usize, y as usize]);
            }
        }
        if samples.len() < 32 {
            return None;
        }
        let mut rows = Vec::new();
        let mut rhs = Vec::new();
        for (grid, target) in [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]]
            .iter()
            .zip([corners[3], corners[2], corners[1], corners[0]])
        {
            let [x, y] = *grid;
            let [u, v] = target;
            rows.extend([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]);
            rhs.push(u);
            rows.extend([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]);
            rhs.push(v);
        }
        let h = DMatrix::from_row_slice(8, 8, &rows)
            .lu()
            .solve(&DVector::from_vec(rhs))?;
        let homography = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0);
        let mut cells = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                let p = homography * Vector3::new(x as f64 + 0.5, y as f64 + 0.5, 1.0);
                cells.push(sample(
                    &image,
                    p.x / p.z - origin[0] as f64,
                    p.y / p.z - origin[1] as f64,
                )?);
            }
        }
        let mut sorted = cells.clone();
        sorted.sort_by(f64::total_cmp);
        let black = sorted[12];
        let contrast = sorted[51] - black;
        if contrast < 60.0 {
            return None;
        }
        let mut template =
            GrayImage::from_pixel(TEMPLATE_SIDE as u32, TEMPLATE_SIDE as u32, Luma([255]));
        for y in 0..8 {
            for x in 0..8 {
                let value = if x == 0
                    || x == 7
                    || y == 0
                    || y == 7
                    || pattern.map_or(cells[y * 8 + x] < black + contrast * 0.5, |p| p[y * 8 + x])
                {
                    0
                } else {
                    255
                };
                for py in 0..CELL_PIXELS {
                    for px in 0..CELL_PIXELS {
                        template.put_pixel(
                            ((x + 1) * CELL_PIXELS + px) as u32,
                            ((y + 1) * CELL_PIXELS + py) as u32,
                            Luma([value]),
                        );
                    }
                }
            }
        }
        let bits = (0..64)
            .map(|i| {
                let x = i % 8;
                let y = i / 8;
                x == 0
                    || x == 7
                    || y == 0
                    || y == 7
                    || pattern.map_or(cells[i] < black + contrast * 0.5, |p| p[i])
            })
            .collect();
        let noise = cells
            .iter()
            .map(|v| {
                let normalized = (v - black) / contrast;
                normalized.abs().min((normalized - 1.0).abs()).powi(2)
            })
            .sum::<f64>()
            / 64.0;
        let noise_floor = (2.0 * noise.sqrt() + 0.002).powi(2);
        Some(Self {
            image,
            origin,
            samples,
            template,
            black,
            contrast,
            bits,
            noise_floor,
            pixels_per_template_pixel: side / (8 * CELL_PIXELS) as f64,
        })
    }

    pub fn refine(
        &self,
        candidates: &[PoseEstimate],
        camera: &CameraConfig,
        corners: &[(f64, f64); 4],
        distortion: bool,
    ) -> Option<PoseEstimate> {
        if camera.fx <= 0.0 || camera.fy <= 0.0 || camera.tag_size_m <= 0.0 {
            return None;
        }
        let rays: Vec<_> = self
            .samples
            .iter()
            .map(|p| {
                let pixel = (
                    (self.origin[0] + p[0]) as f64,
                    (self.origin[1] + p[1]) as f64,
                );
                let (x, y) = if distortion {
                    pose::undistort_normalized_point(pixel, camera)
                } else {
                    (
                        (pixel.0 - camera.cx) / camera.fx,
                        (pixel.1 - camera.cy) / camera.fy,
                    )
                };
                Vector3::new(x, y, 1.0)
            })
            .collect();
        let levels: Vec<_> = [1.0, 0.5, 0.2, 0.0]
            .into_iter()
            .map(|sigma| {
                let im = if sigma == 0.0 {
                    self.image.clone()
                } else {
                    image::imageops::blur(&self.image, sigma)
                };
                let template = if sigma == 0.0 {
                    self.template.clone()
                } else {
                    image::imageops::blur(
                        &self.template,
                        (sigma as f64 / self.pixels_per_template_pixel) as f32,
                    )
                };
                let observed: Vec<_> = self
                    .samples
                    .iter()
                    .map(|p| {
                        (im.get_pixel(p[0] as u32, p[1] as u32)[0] as f64 - self.black)
                            / self.contrast
                    })
                    .collect();
                (template, observed, sigma == 0.0)
            })
            .collect();
        let mut seeds = Vec::<Pose>::new();
        for candidate in candidates {
            let t = candidate.pose.translation;
            if t.norm() < 1e-9 {
                continue;
            }
            let view = t / t.norm();
            // Planar tags can have a second minimum even when native orthogonal
            // iteration returns only one. Reflect the normal about the view ray
            // while preserving a proper rotation and the tag's in-plane axes.
            let reflection = Matrix3::identity() - 2.0 * view * view.transpose();
            let flip_z = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0));
            for rotation in [
                candidate.pose.rotation,
                reflection * candidate.pose.rotation * flip_z,
            ] {
                if !seeds.iter().any(|old| {
                    (old.rotation - rotation).norm() < 1e-6 && (old.translation - t).norm() < 1e-6
                }) {
                    seeds.push(Pose {
                        rotation,
                        translation: t,
                    });
                }
            }
        }
        let mut best: Option<(f64, PoseEstimate)> = None;
        let initial_count = seeds.len();
        let mut cursor = 0;
        let mut expanded = false;
        while cursor < seeds.len() || !expanded {
            if cursor == initial_count && !expanded {
                expanded = true;
                // If the fit remains above the observed noise floor, probe a
                // wider angular basin rather than accept a poor local minimum.
                if let Some((score, estimate)) = &best {
                    if *score > self.noise_floor {
                        let t = estimate.pose.translation;
                        let view = t / t.norm();
                        let reflection = Matrix3::identity() - 2.0 * view * view.transpose();
                        let flip_z = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0));
                        for base in [
                            estimate.pose.rotation,
                            reflection * estimate.pose.rotation * flip_z,
                        ] {
                            for x in [-0.1, 0.0, 0.1] {
                                for y in [-0.1, 0.0, 0.1] {
                                    if x == 0.0 && y == 0.0 {
                                        continue;
                                    }
                                    let rotation =
                                        Rotation3::from_euler_angles(x, y, 0.0).into_inner() * base;
                                    seeds.push(Pose {
                                        rotation,
                                        translation: t,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            if cursor >= seeds.len() {
                break;
            }
            let candidate = seeds[cursor].clone();
            cursor += 1;
            let r = Rotation3::from_matrix_unchecked(candidate.rotation).scaled_axis();
            let t = candidate.translation;
            let prior = Vector3::new(r.x, r.y, r.z);
            let mut params = DVector::from_vec(vec![
                r.x,
                r.y,
                r.z,
                t.x,
                t.y,
                t.z,
                (1.0 / CELL_PIXELS as f64).ln(),
            ]);
            for (template, observed, sharp) in &levels {
                let bits = if *sharp {
                    Some(self.bits.as_slice())
                } else {
                    None
                };
                let noise_scale = if *sharp {
                    self.noise_floor.sqrt().max(0.002)
                } else {
                    1.0
                };
                let mut lambda = 1e-3;
                let Some(mut residual) = residuals(
                    &params,
                    &prior,
                    template,
                    observed,
                    &rays,
                    camera.tag_size_m,
                    bits,
                    noise_scale,
                ) else {
                    break;
                };
                let mut cost = residual.norm_squared();
                for _ in 0..40 {
                    let mut jacobian = DMatrix::zeros(residual.len(), 7);
                    for j in 0..7 {
                        let mut shifted = params.clone();
                        shifted[j] += 1e-5;
                        let Some(values) = residuals(
                            &shifted,
                            &prior,
                            template,
                            observed,
                            &rays,
                            camera.tag_size_m,
                            bits,
                            noise_scale,
                        ) else {
                            continue;
                        };
                        jacobian.set_column(j, &((values - &residual) / 1e-5));
                    }
                    let jt = jacobian.transpose();
                    let mut normal = &jt * &jacobian;
                    for j in 0..7 {
                        normal[(j, j)] += lambda;
                    }
                    let Some(delta) = normal.lu().solve(&(-&jt * &residual)) else {
                        break;
                    };
                    if delta.norm() < 1e-8 {
                        break;
                    }
                    let next = &params + delta;
                    let Some(values) = residuals(
                        &next,
                        &prior,
                        template,
                        observed,
                        &rays,
                        camera.tag_size_m,
                        bits,
                        noise_scale,
                    ) else {
                        lambda *= 10.0;
                        continue;
                    };
                    let next_cost = values.norm_squared();
                    if next_cost < cost {
                        params = next;
                        residual = values;
                        cost = next_cost;
                        lambda = (lambda * 0.3).max(1e-6);
                    } else {
                        lambda = (lambda * 10.0).min(1e8);
                    }
                }
            }
            let (template, observed, _) = levels.last()?;
            let Some(values) = residuals(
                &params,
                &prior,
                template,
                observed,
                &rays,
                camera.tag_size_m,
                Some(&self.bits),
                self.noise_floor.sqrt().max(0.002),
            ) else {
                continue;
            };
            // The weak rotation prior stabilizes fitting, but does not rank hypotheses.
            let score = values.rows(0, rays.len()).norm_squared() / rays.len() as f64
                * self.noise_floor.max(0.002_f64.powi(2));
            let pose = Pose {
                rotation: Rotation3::new(Vector3::new(params[0], params[1], params[2]))
                    .into_inner(),
                translation: Vector3::new(params[3], params[4], params[5]),
            };
            let Some(rmse) =
                pose::reprojection_rmse_px_for_pose(&pose, corners, camera, distortion)
            else {
                continue;
            };
            if score.sqrt() > 0.06 || rmse > 1.0 || !score.is_finite() {
                continue;
            }
            if best.as_ref().map_or(true, |(old, _)| score < *old) {
                best = Some((
                    score,
                    PoseEstimate {
                        pose,
                        reprojection_rmse_px: rmse,
                    },
                ));
            }
        }
        best.map(|(_, estimate)| estimate)
    }
}

/// Joint pixel alignment on a calibrated, level chassis. Unlike independent
/// square PnP, all patches share one X/Y/yaw; only edge spread varies per tag.
/// The inputs are decoded tag patches, mapped tag transforms and camera mount.
pub fn refine_planar_field_pose(
    patches: &[(&TagPatch, Pose)],
    candidates: &[PoseEstimate],
    camera: &CameraConfig,
    mount: &pose::RobotMount,
    field_points: &[Vector3<f64>],
    image_points: &[(f64, f64)],
) -> Option<PoseEstimate> {
    if patches.is_empty() || candidates.is_empty() {
        return None;
    }
    let models: Vec<_> = patches
        .iter()
        .map(|(patch, transform)| {
            let rays: Vec<_> = patch
                .samples
                .iter()
                .map(|p| {
                    let (u, v) = pose::undistort_normalized_point(
                        (
                            (patch.origin[0] + p[0]) as f64,
                            (patch.origin[1] + p[1]) as f64,
                        ),
                        camera,
                    );
                    Vector3::new(u, v, 1.)
                })
                .collect();
            let observed: Vec<_> = patch
                .samples
                .iter()
                .map(|p| {
                    (patch.image.get_pixel(p[0] as u32, p[1] as u32)[0] as f64 - patch.black)
                        / patch.contrast
                })
                .collect();
            (*patch, transform, rays, observed)
        })
        .collect();
    let residual = |params: &DVector<f64>| -> Option<DVector<f64>> {
        if !params.iter().all(|v| v.is_finite()) {
            return None;
        }
        let camera_pose = mount.field_pose(params);
        let mut values = Vec::new();
        for (index, (patch, transform, rays, observed)) in models.iter().enumerate() {
            let width = params[3 + index].exp();
            if !(0.01..=1.5).contains(&width) {
                return None;
            }
            let r = camera_pose.rotation * transform.rotation;
            let t = camera_pose.rotation * transform.translation + camera_pose.translation;
            if t.z <= 0. {
                return None;
            }
            let inverse =
                Matrix3::from_columns(&[r.column(0).into_owned(), r.column(1).into_owned(), t])
                    .try_inverse()?;
            for (ray, observed) in rays.iter().zip(observed) {
                let local = inverse * ray;
                if local.z <= 0. {
                    return None;
                }
                let u = (local.x / local.z / camera.tag_size_m + 0.5) * 8.;
                let v = (local.y / local.z / camera.tag_size_m + 0.5) * 8.;
                values.push(soft_cell_value(&patch.bits, u, v, width) - observed);
            }
        }
        Some(DVector::from_vec(values))
    };
    let mut best: Option<(f64, PoseEstimate)> = None;
    for candidate in candidates.iter().take(6) {
        let robot_rotation = candidate.pose.rotation.transpose() * mount.rotation().transpose();
        let yaw = robot_rotation[(1, 0)].atan2(robot_rotation[(0, 0)]);
        let position = -candidate.pose.rotation.transpose() * candidate.pose.translation
            - Rotation3::from_axis_angle(&Vector3::z_axis(), yaw) * mount.offset();
        let mut params = DVector::from_vec(vec![position.x, position.y, yaw]);
        params = params.insert_rows(3, models.len(), 0.4_f64.ln());
        let Some(mut values) = residual(&params) else {
            continue;
        };
        let mut cost = values.norm_squared();
        let mut lambda = 1e-3;
        for _ in 0..60 {
            let mut jacobian = DMatrix::zeros(values.len(), params.len());
            for j in 0..params.len() {
                let mut shifted = params.clone();
                shifted[j] += 1e-5;
                if let Some(next) = residual(&shifted) {
                    jacobian.set_column(j, &((next - &values) / 1e-5));
                }
            }
            let jt = jacobian.transpose();
            let mut normal = &jt * &jacobian;
            for j in 0..params.len() {
                normal[(j, j)] += lambda;
            }
            let Some(delta) = normal.lu().solve(&(-&jt * &values)) else {
                break;
            };
            if delta.norm() < 1e-8 {
                break;
            }
            let shifted = &params + delta;
            if let Some(next) = residual(&shifted) {
                let next_cost = next.norm_squared();
                if next_cost < cost {
                    params = shifted;
                    values = next;
                    cost = next_cost;
                    lambda = (lambda * 0.3).max(1e-8);
                    continue;
                }
            }
            lambda = (lambda * 10.).min(1e10);
        }
        let score = cost / values.len() as f64;
        if score.sqrt() > 0.15 || !score.is_finite() {
            continue;
        }
        let result = mount.field_pose(&params);
        let origin = -result.rotation.transpose() * result.translation;
        if patches
            .iter()
            .any(|(_, tag)| (origin - tag.translation).dot(&(-tag.rotation.column(2))) <= 0.)
        {
            continue;
        }
        let Some(rmse) =
            pose::correspondence_reprojection_rmse(&result, field_points, image_points, camera)
        else {
            continue;
        };
        if rmse > 2. {
            continue;
        }
        if best.as_ref().map_or(true, |(old, _)| score < *old) {
            best = Some((
                score,
                PoseEstimate {
                    pose: result,
                    reprojection_rmse_px: rmse,
                },
            ));
        }
    }
    best.map(|(_, p)| p)
}

fn residuals(
    params: &DVector<f64>,
    prior: &Vector3<f64>,
    template: &GrayImage,
    observed: &[f64],
    rays: &[Vector3<f64>],
    tag_size: f64,
    bits: Option<&[bool]>,
    noise_scale: f64,
) -> Option<DVector<f64>> {
    if !params.iter().all(|v| v.is_finite()) || params[5] <= 0.0 {
        return None;
    }
    let width = params[6].exp();
    if !(0.005..=0.25).contains(&width) {
        return None;
    }
    let r = Rotation3::new(Vector3::new(params[0], params[1], params[2])).into_inner();
    let h = Matrix3::from_columns(&[
        r.column(0).into_owned(),
        r.column(1).into_owned(),
        Vector3::new(params[3], params[4], params[5]),
    ]);
    let inverse = h.try_inverse()?;
    let mut values = Vec::with_capacity(rays.len() + 3);
    for (ray, value) in rays.iter().zip(observed) {
        let local = inverse * ray;
        if local.z <= 0.0 {
            return None;
        }
        let u = ((local.x / local.z) / tag_size + 0.5) * 8.0;
        let v = ((local.y / local.z) / tag_size + 0.5) * 8.0;
        let prediction = if let Some(bits) = bits {
            soft_cell_value(bits, u, v, width)
        } else {
            let x = (u * CELL_PIXELS as f64 + CELL_PIXELS as f64 - 0.5)
                .clamp(0.0, TEMPLATE_SIDE as f64 - 1.01);
            let y = (v * CELL_PIXELS as f64 + CELL_PIXELS as f64 - 0.5)
                .clamp(0.0, TEMPLATE_SIDE as f64 - 1.01);
            sample(template, x, y)? / 255.0
        };
        // A contrast-normalized residual is not unit-variance noise. Weight the
        // sharp-image likelihood by observed noise so weak initialization priors
        // cannot overpower trustworthy pixel evidence in small, distant tags.
        values.push((prediction - value) / noise_scale);
    }
    // Coarse levels need an orientation anchor to stay in a useful basin. At
    // full resolution the image evidence must be allowed to correct biased
    // small-tag corner estimates rather than inherit their orientation.
    let orientation_std = if bits.is_some() { 1.0 } else { 0.2 };
    for i in 0..3 {
        values.push((params[i] - prior[i]) / orientation_std);
    }
    // Fit edge spread as a nuisance parameter instead of assuming a renderer's
    // sampling density or a particular camera's sharpness.
    values.push((params[6] - (1.0 / CELL_PIXELS as f64).ln()) / 2.0);
    Some(DVector::from_vec(values))
}

fn soft_cell_value(bits: &[bool], x: f64, y: f64, width: f64) -> f64 {
    fn axis(value: f64, width: f64) -> (isize, isize, f64) {
        let edge = value.round();
        if (value - edge).abs() < width * 0.5 {
            (
                edge as isize - 1,
                edge as isize,
                (value - edge) / width + 0.5,
            )
        } else {
            let cell = value.floor() as isize;
            (cell, cell, 0.0)
        }
    }
    let (x0, x1, fx) = axis(x, width);
    let (y0, y1, fy) = axis(y, width);
    let value = |x: isize, y: isize| {
        if (0..8).contains(&x) && (0..8).contains(&y) && bits[y as usize * 8 + x as usize] {
            0.0
        } else {
            1.0
        }
    };
    let top = value(x0, y0) * (1.0 - fx) + value(x1, y0) * fx;
    let bottom = value(x0, y1) * (1.0 - fx) + value(x1, y1) * fx;
    top * (1.0 - fy) + bottom * fy
}

fn sample(image: &GrayImage, x: f64, y: f64) -> Option<f64> {
    if x < 0.0 || y < 0.0 || x >= image.width() as f64 - 1.0 || y >= image.height() as f64 - 1.0 {
        return None;
    }
    let ix = x.floor() as u32;
    let iy = y.floor() as u32;
    let fx = x - ix as f64;
    let fy = y - iy as f64;
    let a =
        image.get_pixel(ix, iy)[0] as f64 * (1.0 - fx) + image.get_pixel(ix + 1, iy)[0] as f64 * fx;
    let b = image.get_pixel(ix, iy + 1)[0] as f64 * (1.0 - fx)
        + image.get_pixel(ix + 1, iy + 1)[0] as f64 * fx;
    Some(a * (1.0 - fy) + b * fy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{CpuDetector, Detection};

    #[test]
    fn bilinear_interpolation_uses_all_four_pixels() {
        let image = GrayImage::from_raw(2, 2, vec![0, 100, 200, 255]).unwrap();
        assert!((sample(&image, 0.5, 0.5).unwrap() - 138.75).abs() < 1e-9);
        assert!(sample(&image, -0.1, 0.0).is_none());
    }

    #[test]
    fn edge_spread_preserves_cell_centers_and_boundary_intensity() {
        let mut bits = vec![true; 64];
        bits[3 * 8 + 4] = false;
        for width in [0.015, 0.03125, 0.10] {
            assert_eq!(soft_cell_value(&bits, 3.5, 3.5, width), 0.0);
            assert_eq!(soft_cell_value(&bits, 4.5, 3.5, width), 1.0);
            assert!((soft_cell_value(&bits, 4.0, 3.5, width) - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn image_alignment_recovers_an_independent_small_tag_pose() {
        let reference = image::load_from_memory(include_bytes!(
            "../simulation_tests/reference/tag36h11_1.png"
        ))
        .unwrap()
        .to_luma8();
        // A different template sampling density, camera, pose and pixel phase
        // from the Python fixtures guards against matching only that renderer.
        let printed =
            image::imageops::resize(&reference, 160, 160, image::imageops::FilterType::Nearest);
        let camera = CameraConfig {
            fx: 520.0,
            fy: 535.0,
            cx: 310.3,
            cy: 225.7,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
            tag_size_m: 0.16,
            x_offset: 0.0,
            y_offset: 0.0,
            z_offset: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
            roll_deg: 0.0,
        };
        let rotation = Rotation3::from_euler_angles(0.12, 0.20, 0.05).into_inner();
        let translation = Vector3::new(0.025, -0.020, 3.0);
        let h = Matrix3::from_columns(&[
            rotation.column(0).into_owned(),
            rotation.column(1).into_owned(),
            translation,
        ]);
        let inverse = h.try_inverse().unwrap();
        let mut image = GrayImage::from_pixel(641, 481, Luma([180]));
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let ray = Vector3::new(
                (x as f64 - camera.cx) / camera.fx,
                (y as f64 - camera.cy) / camera.fy,
                1.0,
            );
            let local = inverse * ray;
            let tx = (local.x / local.z / camera.tag_size_m + 0.5) * 128.0 + 15.5;
            let ty = (local.y / local.z / camera.tag_size_m + 0.5) * 128.0 + 15.5;
            if let Some(value) = sample(&printed, tx, ty) {
                pixel[0] = value.round() as u8;
            }
        }
        let mut detector = CpuDetector::new(1).unwrap();
        let detections = detector.detect(image.as_raw(), 641, 481).unwrap();
        let tag = detections
            .iter()
            .find_map(|d| match d {
                Detection::AprilTag(tag) if tag.id == 1 => Some(tag),
                _ => None,
            })
            .unwrap();
        assert!(tag.image_patch.is_some());
        let best = tag
            .cpu_pose_candidates(&camera, true, 50)
            .into_iter()
            .min_by(|a, b| a.reprojection_rmse_px.total_cmp(&b.reprojection_rmse_px))
            .unwrap();
        let truth = -rotation.transpose() * translation;
        let estimate = -best.pose.rotation.transpose() * best.pose.translation;
        assert!(
            (truth - estimate).norm() < 0.15,
            "camera error {}",
            (truth - estimate).norm()
        );
    }
}
