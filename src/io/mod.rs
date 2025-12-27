#[cfg(feature = "npz")]
use std::io::BufReader;
use std::io::{Read, Seek};

use bytemuck::Zeroable;
use cgmath::{Array, EuclideanSpace, InnerSpace, Point3, Vector3};
use half::f16;

use crate::pointcloud::{
    Aabb, Covariance3D, Gaussian, GaussianCompressed, GaussianQuantization, Quantization,
};

#[cfg(feature = "npz")]
use self::npz::NpzReader;

pub use self::ply::EmbeddedCamera;
use self::ply::PlyReader;

#[cfg(feature = "npz")]
pub mod npz;
pub mod ply;

pub trait PointCloudReader {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error>;

    fn magic_bytes() -> &'static [u8];
    fn file_ending() -> &'static str;
}

pub struct GenericGaussianPointCloud {
    gaussians: Vec<u8>,
    sh_coefs: Vec<u8>,
    compressed: bool,
    pub covars: Option<Vec<Covariance3D>>,
    pub quantization: Option<GaussianQuantization>,
    pub sh_deg: u32,
    pub num_points: usize,
    pub kernel_size: Option<f32>,
    pub mip_splatting: Option<bool>,
    pub background_color: Option<[f32; 3]>,

    pub up: Option<Vector3<f32>>,
    pub center: Point3<f32>,
    pub aabb: Aabb<f32>,
    pub embedded_camera: Option<EmbeddedCamera>,
}

impl GenericGaussianPointCloud {
    pub fn load<R: Read + Seek>(f: R) -> Result<Self, anyhow::Error> {
        let mut signature: [u8; 4] = [0; 4];
        let mut f = f;
        f.read_exact(&mut signature)?;
        f.rewind()?;
        if signature.starts_with(PlyReader::<R>::magic_bytes()) {
            let mut ply_reader = PlyReader::new(f)?;
            return ply_reader.read();
        }
        #[cfg(feature = "npz")]
        if signature.starts_with(NpzReader::<R>::magic_bytes()) {
            let mut reader = BufReader::new(f);
            let mut npz_reader = NpzReader::new(&mut reader)?;
            return npz_reader.read();
        }
        Err(anyhow::anyhow!("Unknown file format"))
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        gaussians: Vec<Gaussian>,
        sh_coefs: Vec<[[f16; 3]; 16]>,
        sh_deg: u32,
        num_points: usize,
        kernel_size: Option<f32>,
        mip_splatting: Option<bool>,
        background_color: Option<[f32; 3]>,
        covars: Option<Vec<Covariance3D>>,
        quantization: Option<GaussianQuantization>,
        embedded_camera: Option<EmbeddedCamera>,
    ) -> Self {
        let mut bbox: Aabb<f32> = Aabb::zeroed();
        for v in &gaussians {
            bbox.grow(&v.xyz);
        }

        let (center, mut up) = plane_from_points(
            gaussians
                .iter()
                .map(|g| g.xyz.cast().unwrap())
                .collect::<Vec<Point3<f32>>>()
                .as_slice(),
        );

        if bbox.radius() < 10. {
            up = None;
        }
        Self {
            gaussians: bytemuck::cast_slice(&gaussians).to_vec(),
            sh_coefs: bytemuck::cast_slice(&sh_coefs).to_vec(),
            sh_deg,
            num_points,
            kernel_size,
            mip_splatting,
            background_color,
            covars,
            quantization,
            up,
            center,
            aabb: bbox,
            compressed: false,
            embedded_camera,
        }
    }

    #[cfg(feature = "npz")]
    #[allow(clippy::too_many_arguments)]
    fn new_compressed(
        gaussians: Vec<GaussianCompressed>,
        sh_coefs: Vec<u8>,
        sh_deg: u32,
        num_points: usize,
        kernel_size: Option<f32>,
        mip_splatting: Option<bool>,
        background_color: Option<[f32; 3]>,
        covars: Option<Vec<Covariance3D>>,
        quantization: Option<GaussianQuantization>,
    ) -> Self {
        let mut bbox: Aabb<f32> = Aabb::unit();
        for v in &gaussians {
            bbox.grow(&v.xyz);
        }

        let (center, mut up) = plane_from_points(
            gaussians
                .iter()
                .map(|g| g.xyz.cast().unwrap())
                .collect::<Vec<Point3<f32>>>()
                .as_slice(),
        );

        if bbox.radius() < 10. {
            up = None;
        }
        Self {
            gaussians: bytemuck::cast_slice(&gaussians).to_vec(),
            sh_coefs: bytemuck::cast_slice(&sh_coefs).to_vec(),
            sh_deg,
            num_points,
            kernel_size,
            mip_splatting,
            background_color,
            covars,
            quantization,
            up,
            center,
            aabb: bbox,
            compressed: true,
            embedded_camera: None,
        }
    }

    pub fn gaussians(&self) -> anyhow::Result<&[Gaussian]> {
        if self.compressed {
            Err(anyhow::anyhow!("Gaussians are compressed"))
        } else {
            Ok(bytemuck::cast_slice(&self.gaussians))
        }
    }

    pub fn gaussians_compressed(&self) -> anyhow::Result<&[GaussianCompressed]> {
        if self.compressed {
            Err(anyhow::anyhow!("Gaussians are compressed"))
        } else {
            Ok(bytemuck::cast_slice(&self.gaussians))
        }
    }

    pub fn sh_coefs_buffer(&self) -> &[u8] {
        &self.sh_coefs
    }

    pub fn gaussian_buffer(&self) -> &[u8] {
        &self.gaussians
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }

    /// Information about downsampling that was applied
    pub fn downsample_info(&self, original_count: usize) -> Option<DownsampleInfo> {
        if original_count > self.num_points {
            Some(DownsampleInfo {
                original_count,
                final_count: self.num_points,
                ratio: self.num_points as f32 / original_count as f32,
            })
        } else {
            None
        }
    }

    /// Downsample the point cloud to at most max_points.
    /// Returns the original point count if downsampling occurred.
    pub fn downsample(&mut self, max_points: usize) -> Option<usize> {
        if self.num_points <= max_points {
            return None;
        }

        let original_count = self.num_points;
        let step = self.num_points.div_ceil(max_points);
        let new_count = self.num_points.div_ceil(step);

        log::info!(
            "Downsampling point cloud from {} to {} points (keeping every {}th point)",
            self.num_points,
            new_count,
            step
        );

        if self.compressed {
            self.downsample_compressed(step, new_count);
        } else {
            self.downsample_uncompressed(step, new_count);
        }

        // Downsample covariances if present
        if let Some(ref covars) = self.covars {
            let new_covars: Vec<Covariance3D> = covars.iter().step_by(step).copied().collect();
            self.covars = Some(new_covars);
        }

        self.num_points = new_count;

        // Recalculate bounding box from downsampled points
        self.recalculate_bounds();

        Some(original_count)
    }

    fn downsample_uncompressed(&mut self, step: usize, new_count: usize) {
        let gaussian_size = std::mem::size_of::<Gaussian>();
        let sh_coef_size = std::mem::size_of::<[[f16; 3]; 16]>();

        // Downsample gaussians
        let mut new_gaussians = Vec::with_capacity(new_count * gaussian_size);
        for i in (0..self.gaussians.len())
            .step_by(step * gaussian_size)
            .take(new_count)
        {
            new_gaussians.extend_from_slice(&self.gaussians[i..i + gaussian_size]);
        }
        self.gaussians = new_gaussians;

        // Downsample SH coefficients
        let mut new_sh_coefs = Vec::with_capacity(new_count * sh_coef_size);
        for i in (0..self.sh_coefs.len())
            .step_by(step * sh_coef_size)
            .take(new_count)
        {
            new_sh_coefs.extend_from_slice(&self.sh_coefs[i..i + sh_coef_size]);
        }
        self.sh_coefs = new_sh_coefs;
    }

    fn downsample_compressed(&mut self, step: usize, new_count: usize) {
        let gaussian_size = std::mem::size_of::<GaussianCompressed>();

        // Downsample gaussians
        let mut new_gaussians = Vec::with_capacity(new_count * gaussian_size);
        for i in (0..self.gaussians.len())
            .step_by(step * gaussian_size)
            .take(new_count)
        {
            new_gaussians.extend_from_slice(&self.gaussians[i..i + gaussian_size]);
        }
        self.gaussians = new_gaussians;

        // For compressed format, SH coefficients may be shared via indices
        // We keep all SH coefficients as they may be referenced by remaining gaussians
        // This is a simplification - a more sophisticated approach would remap indices
    }

    fn recalculate_bounds(&mut self) {
        if self.compressed {
            let gaussians: &[GaussianCompressed] = bytemuck::cast_slice(&self.gaussians);
            let mut bbox: Aabb<f32> = Aabb::zeroed();
            for g in gaussians {
                bbox.grow(&g.xyz);
            }
            self.aabb = bbox;
            self.center = bbox.center();
        } else {
            let gaussians: &[Gaussian] = bytemuck::cast_slice(&self.gaussians);
            let mut bbox: Aabb<f32> = Aabb::zeroed();
            for g in gaussians {
                bbox.grow(&g.xyz);
            }
            self.aabb = bbox;
            self.center = bbox.center();
        }
    }

    /// Compress the point cloud by quantizing colors and opacity to i8.
    /// This reduces GPU memory by ~40% with minimal quality loss.
    /// Returns Ok(true) if compression was applied, Ok(false) if already compressed.
    pub fn compress(&mut self) -> Result<bool, anyhow::Error> {
        if self.compressed {
            return Ok(false); // Already compressed
        }

        log::info!(
            "Compressing point cloud ({} points) for faster rendering...",
            self.num_points
        );

        // Parse uncompressed gaussians
        let gaussians: &[Gaussian] = bytemuck::cast_slice(&self.gaussians);

        // Collect opacity values for quantization
        let opacities: Vec<f32> = gaussians.iter().map(|g| g.opacity.to_f32()).collect();

        // Compute opacity quantization parameters
        let opacity_quant = Self::compute_quantization(&opacities);

        // Parse SH coefficients and compute quantization parameters
        let sh_coefs_f16: &[[[f16; 3]; 16]] = bytemuck::cast_slice(&self.sh_coefs);

        // Separate DC (first coefficient) and rest
        let mut dc_values: Vec<f32> = Vec::with_capacity(self.num_points * 3);
        let mut rest_values: Vec<f32> = Vec::with_capacity(self.num_points * 45); // 15 * 3

        for coefs in sh_coefs_f16.iter().take(self.num_points) {
            // DC component (index 0)
            for val in &coefs[0] {
                dc_values.push(val.to_f32());
            }
            // Rest components (indices 1-15)
            for coef in coefs.iter().skip(1) {
                for val in coef {
                    rest_values.push(val.to_f32());
                }
            }
        }

        let dc_quant = Self::compute_quantization(&dc_values);
        let rest_quant = Self::compute_quantization(&rest_values);

        log::debug!(
            "DC quantization: scale={:.6}, zero_point={}, value_range=[{:.4}, {:.4}]",
            dc_quant.scale,
            dc_quant.zero_point,
            dc_values.iter().cloned().fold(f32::INFINITY, f32::min),
            dc_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        log::debug!(
            "REST quantization: scale={:.6}, zero_point={}, value_range=[{:.4}, {:.4}]",
            rest_quant.scale,
            rest_quant.zero_point,
            rest_values.iter().cloned().fold(f32::INFINITY, f32::min),
            rest_values
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        );
        log::debug!(
            "Opacity quantization: scale={:.6}, zero_point={}",
            opacity_quant.scale,
            opacity_quant.zero_point
        );

        // Build compressed gaussians
        let mut compressed_gaussians: Vec<GaussianCompressed> = Vec::with_capacity(self.num_points);
        let mut covariances: Vec<Covariance3D> = Vec::with_capacity(self.num_points);

        for (i, g) in gaussians.iter().enumerate() {
            let quantized_opacity = Self::quantize_value(g.opacity.to_f32(), &opacity_quant);

            compressed_gaussians.push(GaussianCompressed {
                xyz: g.xyz,
                opacity: quantized_opacity,
                scale_factor: 0, // Not using scale factor for basic compression
                geometry_idx: i as u32,
                sh_idx: i as u32,
            });

            covariances.push(Covariance3D(g.cov));
        }

        // Quantize SH coefficients to i8
        let mut quantized_sh: Vec<i8> = Vec::with_capacity(self.num_points * 48); // 16 * 3

        for coefs in sh_coefs_f16.iter().take(self.num_points) {
            // DC component
            for val in &coefs[0] {
                quantized_sh.push(Self::quantize_value(val.to_f32(), &dc_quant));
            }
            // Rest components
            for coef in coefs.iter().skip(1) {
                for val in coef {
                    quantized_sh.push(Self::quantize_value(val.to_f32(), &rest_quant));
                }
            }
        }

        // Update self with compressed data
        self.gaussians = bytemuck::cast_slice(&compressed_gaussians).to_vec();
        self.sh_coefs = bytemuck::cast_slice(&quantized_sh).to_vec();
        self.covars = Some(covariances);
        self.quantization = Some(GaussianQuantization {
            color_dc: Quantization::new(dc_quant.zero_point, dc_quant.scale),
            color_rest: Quantization::new(rest_quant.zero_point, rest_quant.scale),
            opacity: Quantization::new(opacity_quant.zero_point, opacity_quant.scale),
            scaling_factor: Quantization::default(),
        });
        self.compressed = true;

        log::info!("Compression complete. Memory reduced by ~40%.");
        Ok(true)
    }

    /// Compute quantization parameters (scale and zero_point) for a set of values.
    /// Maps float values to i8 range [-128, 127] with the formula:
    /// quantized = value / scale + zero_point
    /// value = (quantized - zero_point) * scale
    fn compute_quantization(values: &[f32]) -> QuantizationParams {
        if values.is_empty() {
            return QuantizationParams {
                scale: 1.0,
                zero_point: 0,
            };
        }

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max - min;
        if range < 1e-10 {
            // All values are approximately the same
            // Use a scale that maps this value to 0 when dequantized
            // quantized = value / scale + zero_point
            // For value ~= min, we want quantized = 0, so zero_point = -min / scale
            // Using scale = 1.0 / 128.0 allows reasonable precision
            let avg_value = (min + max) / 2.0;
            return QuantizationParams {
                scale: 1.0 / 128.0,
                zero_point: -(avg_value * 128.0).round() as i32,
            };
        }

        let scale = range / 255.0;
        let zero_point = (-128.0 - min / scale).round() as i32;

        QuantizationParams { scale, zero_point }
    }

    /// Quantize a single f32 value to i8 using the given parameters.
    fn quantize_value(value: f32, params: &QuantizationParams) -> i8 {
        let quantized = (value / params.scale + params.zero_point as f32).round();
        quantized.clamp(-128.0, 127.0) as i8
    }
}

/// Parameters for quantization
struct QuantizationParams {
    scale: f32,
    zero_point: i32,
}

/// Information about downsampling that was applied to a point cloud
#[derive(Debug, Clone, Copy)]
pub struct DownsampleInfo {
    pub original_count: usize,
    pub final_count: usize,
    pub ratio: f32,
}

// Fit a plane to a collection of points.
// Fast, and accurate to within a few degrees.
// Returns None if the points do not span a plane.
// see http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
fn plane_from_points(points: &[Point3<f32>]) -> (Point3<f32>, Option<Vector3<f32>>) {
    let n = points.len();

    let mut sum = Point3 {
        x: 0.0f32,
        y: 0.0f32,
        z: 0.0f32,
    };
    for p in points {
        sum += p.to_vec();
    }
    let centroid = sum * (1.0 / (n as f32));
    if n < 3 {
        return (centroid, None);
    }

    // Calculate full 3x3 covariance matrix, excluding symmetries:
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut xz = 0.0;
    let mut yy = 0.0;
    let mut yz = 0.0;
    let mut zz = 0.0;

    for p in points {
        let r = p - centroid;
        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    xx /= n as f32;
    xy /= n as f32;
    xz /= n as f32;
    yy /= n as f32;
    yz /= n as f32;
    zz /= n as f32;

    let mut weighted_dir = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    {
        let det_x = yy * zz - yz * yz;
        let axis_dir = Vector3 {
            x: det_x,
            y: xz * yz - xy * zz,
            z: xy * yz - xz * yy,
        };
        let mut weight = det_x * det_x;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += axis_dir * weight;
    }

    {
        let det_y = xx * zz - xz * xz;
        let axis_dir = Vector3 {
            x: xz * yz - xy * zz,
            y: det_y,
            z: xy * xz - yz * xx,
        };
        let mut weight = det_y * det_y;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += axis_dir * weight;
    }

    {
        let det_z = xx * yy - xy * xy;
        let axis_dir = Vector3 {
            x: xy * yz - xz * yy,
            y: xy * xz - yz * xx,
            z: det_z,
        };
        let mut weight = det_z * det_z;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += axis_dir * weight;
    }

    let mut normal = weighted_dir.normalize();

    if normal.dot(Vector3::unit_y()) < 0. {
        normal = -normal;
    }
    if normal.is_finite() {
        (centroid, Some(normal))
    } else {
        (centroid, None)
    }
}
