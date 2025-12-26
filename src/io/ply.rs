use anyhow::Ok;
use half::f16;
use ply_rs::ply;

use std::io::{self, BufReader, Read, Seek};

use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use cgmath::{InnerSpace, Point3, Quaternion, Vector3};

use crate::{
    pointcloud::Gaussian,
    utils::{build_cov, sh_deg_from_num_coefs, sigmoid},
};

use super::{GenericGaussianPointCloud, PointCloudReader};

/// Camera data embedded in ML-SHARP PLY files
#[derive(Debug, Clone)]
pub struct EmbeddedCamera {
    /// 4x4 camera-to-world extrinsic matrix (row-major)
    pub extrinsic: [[f32; 4]; 4],
    /// 3x3 intrinsic matrix (row-major): [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    pub intrinsic: [[f32; 3]; 3],
    /// Image dimensions (width, height)
    pub image_size: (u32, u32),
}

pub struct PlyReader<R: Read + Seek> {
    header: ply_rs::ply::Header,
    reader: BufReader<R>,
    sh_deg: u32,
    num_points: usize,
    mip_splatting: Option<bool>,
    kernel_size: Option<f32>,
    background_color: Option<[f32; 3]>,
    has_normals: bool,
    has_ml_sharp_camera: bool,
}

impl<R: io::Read + io::Seek> PlyReader<R> {
    pub fn new(reader: R) -> Result<Self, anyhow::Error> {
        let mut reader = BufReader::new(reader);
        let parser = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let header = parser.read_header(&mut reader).unwrap();
        let sh_deg = Self::file_sh_deg(&header)?;
        let num_points = Self::num_points(&header)?;
        let mip_splatting = Self::mip_splatting(&header)?;
        let kernel_size = Self::kernel_size(&header)?;
        let background_color = Self::background_color(&header)
            .map_err(|e| log::warn!("could not parse background_color: {}", e))
            .unwrap_or_default();
        let has_normals = Self::has_normals(&header);
        let has_ml_sharp_camera = Self::has_ml_sharp_camera(&header);
        Ok(Self {
            header,
            reader,
            sh_deg,
            num_points,
            mip_splatting,
            kernel_size,
            background_color,
            has_normals,
            has_ml_sharp_camera,
        })
    }

    fn has_normals(header: &ply::Header) -> bool {
        header.elements["vertex"].properties.contains_key("nx")
    }

    fn has_ml_sharp_camera(header: &ply::Header) -> bool {
        header.elements.contains_key("extrinsic")
            && header.elements.contains_key("intrinsic")
            && header.elements.contains_key("image_size")
    }

    fn read_ml_sharp_camera<B: ByteOrder>(&mut self) -> anyhow::Result<EmbeddedCamera> {
        // Read 16 floats for 4x4 extrinsic matrix
        let mut extrinsic = [[0f32; 4]; 4];
        for row in &mut extrinsic {
            self.reader.read_f32_into::<B>(row)?;
        }

        // Read 9 floats for 3x3 intrinsic matrix
        let mut intrinsic = [[0f32; 3]; 3];
        for row in &mut intrinsic {
            self.reader.read_f32_into::<B>(row)?;
        }

        // Read 2 uints for image size (width, height)
        let width = self.reader.read_u32::<B>()?;
        let height = self.reader.read_u32::<B>()?;

        // Skip remaining ML-SHARP elements: frame (2 ints), disparity (2 floats),
        // color_space (1 uchar), version (3 uchars)
        let mut _frame = [0i32; 2];
        self.reader.read_i32_into::<B>(&mut _frame)?;
        let mut _disparity = [0f32; 2];
        self.reader.read_f32_into::<B>(&mut _disparity)?;
        let mut _color_space = [0u8; 1];
        self.reader.read_exact(&mut _color_space)?;
        let mut _version = [0u8; 3];
        self.reader.read_exact(&mut _version)?;

        Ok(EmbeddedCamera {
            extrinsic,
            intrinsic,
            image_size: (width, height),
        })
    }

    fn read_line<B: ByteOrder>(
        &mut self,
        sh_deg: usize,
        has_normals: bool,
    ) -> anyhow::Result<(Gaussian, [[f16; 3]; 16])> {
        let mut pos = [0.; 3];
        self.reader.read_f32_into::<B>(&mut pos)?;

        // skip normals if present
        if has_normals {
            let mut _normals = [0.; 3];
            self.reader.read_f32_into::<B>(&mut _normals)?;
        }

        let mut sh: [[f32; 3]; 16] = [[0.; 3]; 16];
        self.reader.read_f32_into::<B>(&mut sh[0])?;
        let mut sh_rest = [0.; 15 * 3];
        let num_coefs = (sh_deg + 1) * (sh_deg + 1);
        self.reader
            .read_f32_into::<B>(&mut sh_rest[..(num_coefs - 1) * 3])?;

        // higher order coefficients are stored with channel first (shape:[N,3,C])
        for i in 0..(num_coefs - 1) {
            for j in 0..3 {
                sh[i + 1][j] = sh_rest[j * (num_coefs - 1) + i];
            }
        }

        let opacity = sigmoid(self.reader.read_f32::<B>()?);

        let scale_1 = self.reader.read_f32::<B>()?.exp();
        let scale_2 = self.reader.read_f32::<B>()?.exp();
        let scale_3 = self.reader.read_f32::<B>()?.exp();
        let scale = Vector3::new(scale_1, scale_2, scale_3);

        let rot_0 = self.reader.read_f32::<B>()?;
        let rot_1 = self.reader.read_f32::<B>()?;
        let rot_2 = self.reader.read_f32::<B>()?;
        let rot_3 = self.reader.read_f32::<B>()?;
        let rot = Quaternion::new(rot_0, rot_1, rot_2, rot_3).normalize();

        let cov = build_cov(rot, scale);

        Ok((
            Gaussian::new(
                Point3::from(pos).cast().unwrap(),
                f16::from_f32(opacity),
                cov.map(f16::from_f32),
            ),
            sh.map(|x| x.map(f16::from_f32)),
        ))
    }

    fn file_sh_deg(header: &ply::Header) -> Result<u32, anyhow::Error> {
        let num_sh_coefs = header.elements["vertex"]
            .properties
            .keys()
            .filter(|k| k.starts_with("f_"))
            .count();

        let file_sh_deg = sh_deg_from_num_coefs(num_sh_coefs as u32 / 3).ok_or(anyhow::anyhow!(
            "number of sh coefficients {num_sh_coefs} cannot be mapped to sh degree"
        ))?;
        Ok(file_sh_deg)
    }

    fn num_points(header: &ply::Header) -> Result<usize, anyhow::Error> {
        Ok(header
            .elements
            .get("vertex")
            .ok_or(anyhow::anyhow!("missing element vertex"))?
            .count as usize)
    }

    fn mip_splatting(header: &ply::Header) -> Result<Option<bool>, anyhow::Error> {
        Ok(header
            .comments
            .iter()
            .find(|c| c.contains("mip"))
            .map(|c| c.split('=').next_back().unwrap().parse::<bool>())
            .transpose()?)
    }
    fn kernel_size(header: &ply::Header) -> Result<Option<f32>, anyhow::Error> {
        Ok(header
            .comments
            .iter()
            .find(|c| c.contains("kernel_size"))
            .map(|c| c.split('=').next_back().unwrap().parse::<f32>())
            .transpose()?)
    }

    fn background_color(header: &ply::Header) -> anyhow::Result<Option<[f32; 3]>> {
        header
            .comments
            .iter()
            .find(|c| c.contains("background_color"))
            .map(|c| {
                let value = c.split('=').next_back();
                let parts = value.map(|c| {
                    c.split(",")
                        .map(|v| v.parse::<f32>())
                        .collect::<Result<Vec<f32>, _>>()
                });
                parts.map_or_else(
                    || Err(anyhow::anyhow!("could not parse:")),
                    |x| {
                        x.map_err(|e| anyhow::anyhow!("could not parse: {}", e))
                            .map(|x| [x[0], x[1], x[2]])
                    },
                )
            })
            .transpose()
    }
}

impl<R: io::Read + io::Seek> PointCloudReader for PlyReader<R> {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error> {
        let mut gaussians = Vec::with_capacity(self.num_points);
        let mut sh_coefs = Vec::with_capacity(self.num_points);
        let has_normals = self.has_normals;
        let has_ml_sharp_camera = self.has_ml_sharp_camera;

        let embedded_camera = match self.header.encoding {
            ply_rs::ply::Encoding::Ascii => todo!("ascii ply format not supported"),
            ply_rs::ply::Encoding::BinaryBigEndian => {
                for _ in 0..self.num_points {
                    let (g, s) = self.read_line::<BigEndian>(self.sh_deg as usize, has_normals)?;
                    gaussians.push(g);
                    sh_coefs.push(s);
                }
                if has_ml_sharp_camera {
                    Some(self.read_ml_sharp_camera::<BigEndian>()?)
                } else {
                    None
                }
            }
            ply_rs::ply::Encoding::BinaryLittleEndian => {
                for _ in 0..self.num_points {
                    let (g, s) =
                        self.read_line::<LittleEndian>(self.sh_deg as usize, has_normals)?;
                    gaussians.push(g);
                    sh_coefs.push(s);
                }
                if has_ml_sharp_camera {
                    Some(self.read_ml_sharp_camera::<LittleEndian>()?)
                } else {
                    None
                }
            }
        };

        Ok(GenericGaussianPointCloud::new(
            gaussians,
            sh_coefs,
            self.sh_deg,
            self.num_points,
            self.kernel_size,
            self.mip_splatting,
            self.background_color,
            None,
            None,
            embedded_camera,
        ))
    }

    fn magic_bytes() -> &'static [u8] {
        "ply".as_bytes()
    }

    fn file_ending() -> &'static str {
        "ply"
    }
}
