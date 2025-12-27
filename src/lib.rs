use std::{
    io::{Read, Seek},
    path::PathBuf,
    sync::Arc,
};

#[cfg(target_arch = "wasm32")]
use std::io::Cursor;

#[cfg(target_arch = "wasm32")]
use once_cell::sync::Lazy;
#[cfg(target_arch = "wasm32")]
use std::sync::Mutex;

use renderer::Display;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};
use wgpu::Backends;

use cgmath::{Deg, EuclideanSpace, InnerSpace, Point3, Quaternion, UlpsEq, Vector2, Vector3};
use egui::FullOutput;
use num_traits::One;

use utils::key_to_num;
#[cfg(not(target_arch = "wasm32"))]
use utils::RingBuffer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
extern crate js_sys;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{DeviceEvent, ElementState, Event, TouchPhase as WinitTouchPhase, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

mod animation;
mod ui;
pub use animation::{
    Animation, Sampler, TrackingShot, TrajectoryAnimation, TrajectoryParams, TrajectoryType,
    Transition,
};
mod camera;
pub use camera::{Camera, PerspectiveCamera, PerspectiveProjection};
mod controller;
pub use controller::CameraController;
mod pointcloud;
pub use pointcloud::PointCloud;

pub mod io;

mod renderer;
pub use renderer::{GaussianRenderer, SplattingArgs};

mod scene;
use crate::utils::GPUStopwatch;

pub use self::scene::{Scene, SceneCamera, Split};

pub mod gpu_rs;
mod ui_renderer;
mod uniform;
mod utils;

pub struct RenderConfig {
    pub no_vsync: bool,
    pub hdr: bool,
}

/// Pending file data for in-place reload (WASM only)
#[cfg(target_arch = "wasm32")]
struct PendingFile {
    data: Vec<u8>,
    filename: String,
    compress: bool,
}

/// Global state for pending file to load (WASM only)
/// This allows JavaScript to queue a file load that the event loop will pick up
#[cfg(target_arch = "wasm32")]
static PENDING_FILE: Lazy<Mutex<Option<PendingFile>>> = Lazy::new(|| Mutex::new(None));

/// Global state for target canvas size from embedded camera (WASM only)
/// This allows JavaScript to resize the canvas to match the original image dimensions
#[cfg(target_arch = "wasm32")]
static TARGET_CANVAS_SIZE: Lazy<Mutex<Option<(u32, u32)>>> = Lazy::new(|| Mutex::new(None));

/// Global state for pending trajectory to start (WASM only)
#[cfg(target_arch = "wasm32")]
static PENDING_TRAJECTORY: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

/// Global state for pending reset and replay (WASM only)
#[cfg(target_arch = "wasm32")]
static PENDING_RESET_AND_PLAY: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

/// Device performance tier for determining resource limits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTier {
    Low,    // <= 4GB RAM: 1M splats max
    Medium, // <= 8GB RAM: 2M splats max
    High,   // > 8GB RAM: 4M splats max
}

impl DeviceTier {
    /// Maximum number of splats recommended for this device tier
    pub fn max_splats(&self) -> u32 {
        match self {
            DeviceTier::Low => 1_000_000,
            DeviceTier::Medium => 2_000_000,
            DeviceTier::High => 4_000_000,
        }
    }
}

/// Device capabilities for determining resource limits
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    pub device_tier: DeviceTier,
    pub max_splats: u32,
    pub device_name: String,
}

impl DeviceCapabilities {
    /// Bytes per splat including all buffers (Gaussian + 2D splat + sorting overhead)
    const BYTES_PER_SPLAT: u64 = 100; // Conservative estimate

    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let limits = adapter.limits();
        let info = adapter.get_info();

        // Detect device tier from system memory (WASM) or buffer limits (native)
        let device_tier = Self::detect_device_tier(&info, &limits);

        // Calculate max splats based on buffer limits and device tier
        let buffer_based_max = (limits.max_buffer_size / Self::BYTES_PER_SPLAT) as u32;
        let tier_based_max = device_tier.max_splats();

        // Use the more conservative limit
        let max_splats = buffer_based_max.min(tier_based_max);

        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            device_tier,
            max_splats,
            device_name: info.name.clone(),
        }
    }

    fn detect_device_tier(_info: &wgpu::AdapterInfo, limits: &wgpu::Limits) -> DeviceTier {
        // Try to get device memory from JavaScript on WASM
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(memory_gb) = Self::get_device_memory_wasm() {
                return if memory_gb <= 4.0 {
                    DeviceTier::Low
                } else if memory_gb <= 8.0 {
                    DeviceTier::Medium
                } else {
                    DeviceTier::High
                };
            }
        }

        // Fallback: estimate tier from buffer limits
        // max_buffer_size is typically proportional to available GPU memory
        let max_buffer_mb = limits.max_buffer_size / (1024 * 1024);
        if max_buffer_mb <= 256 {
            DeviceTier::Low
        } else if max_buffer_mb <= 1024 {
            DeviceTier::Medium
        } else {
            DeviceTier::High
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn get_device_memory_wasm() -> Option<f64> {
        let window = web_sys::window()?;
        let navigator = window.navigator();
        // navigator.deviceMemory returns memory in GB (2, 4, 8, etc.)
        let device_memory = js_sys::Reflect::get(&navigator, &"deviceMemory".into()).ok()?;
        device_memory.as_f64()
    }
}

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
    pub capabilities: DeviceCapabilities,
}

impl WGPUContext {
    pub async fn new_instance() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        return WGPUContext::new(&instance, None).await;
    }

    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();
        log::info!("using {}", adapter.get_info().name);

        #[cfg(target_arch = "wasm32")]
        let required_features = wgpu::Features::default();
        #[cfg(not(target_arch = "wasm32"))]
        let required_features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;

        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features,
                #[cfg(not(target_arch = "wasm32"))]
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    max_storage_buffers_per_shader_stage: 12,
                    max_compute_workgroup_storage_size: 1 << 15,
                    ..adapter_limits
                },

                #[cfg(target_arch = "wasm32")]
                required_limits: wgpu::Limits {
                    max_compute_workgroup_storage_size: 1 << 15,
                    ..adapter_limits
                },
                label: None,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .unwrap();

        // Detect device capabilities for resource limiting
        let capabilities = DeviceCapabilities::from_adapter(&adapter);
        log::info!(
            "Device capabilities: tier={:?}, max_splats={}",
            capabilities.device_tier,
            capabilities.max_splats
        );

        Self {
            device,
            queue,
            adapter,
            capabilities,
        }
    }
}

pub struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    scale_factor: f32,

    pc: PointCloud,
    pointcloud_file_path: Option<PathBuf>,
    renderer: GaussianRenderer,
    animation: Option<(Animation<PerspectiveCamera>, bool)>,
    controller: CameraController,
    scene: Option<Scene>,
    scene_file_path: Option<PathBuf>,
    current_view: Option<usize>,
    ui_renderer: ui_renderer::EguiWGPU,
    fps: f32,
    ui_visible: bool,

    #[cfg(not(target_arch = "wasm32"))]
    history: RingBuffer<(Duration, Duration, Duration)>,
    display: Display,

    splatting_args: SplattingArgs,

    saved_cameras: Vec<SceneCamera>,
    #[cfg(feature = "video")]
    #[allow(dead_code)]
    cameras_save_path: String,
    stopwatch: Option<GPUStopwatch>,
    initial_camera: Option<PerspectiveCamera>,

    /// If point cloud was downsampled, stores the original count
    downsampled_from: Option<usize>,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<R: Read + Seek>(
        window: Window,
        pc_file: R,
        render_config: &RenderConfig,
        compress: bool,
    ) -> anyhow::Result<Self> {
        let mut size = window.inner_size();
        if size == PhysicalSize::new(0, 0) {
            size = PhysicalSize::new(800, 600);
        }

        let window = Arc::new(window);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        let device = &wgpu_context.device;
        let queue = &wgpu_context.queue;

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        let surface_format = *surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .unwrap_or(&surface_caps.formats[0]);

        let render_format = if render_config.hdr {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format.remove_srgb_suffix()],
        };
        surface.configure(device, &config);

        let mut pc_raw = io::GenericGaussianPointCloud::load(pc_file)?;

        // Downsample if point cloud exceeds device capabilities
        let max_splats = wgpu_context.capabilities.max_splats as usize;
        let downsampled_from = pc_raw.downsample(max_splats);

        // Compress if requested and not already compressed
        if compress && !pc_raw.compressed() {
            if let Err(e) = pc_raw.compress() {
                log::warn!("Failed to compress point cloud: {:?}", e);
            }
        }

        // Extract embedded camera before creating PointCloud (which consumes pc_raw)
        let initial_camera: Option<PerspectiveCamera> =
            pc_raw.embedded_camera.clone().map(|c| c.into());

        // Update target canvas size from embedded camera (WASM only)
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(ref embedded) = pc_raw.embedded_camera {
                if let Ok(mut size) = TARGET_CANVAS_SIZE.lock() {
                    *size = Some(embedded.image_size);
                    log::info!(
                        "Set target canvas size from embedded camera: {}x{}",
                        embedded.image_size.0,
                        embedded.image_size.1
                    );
                }
            }
        }

        let pc = PointCloud::new(device, pc_raw)?;
        log::info!(
            "loaded point cloud with {:} points (compressed: {})",
            pc.num_points(),
            pc.compressed()
        );
        if let Some(original) = downsampled_from {
            log::warn!(
                "Point cloud was downsampled from {} to {} points due to device limits",
                original,
                pc.num_points()
            );
        }

        let renderer =
            GaussianRenderer::new(device, queue, render_format, pc.sh_deg(), pc.compressed()).await;

        let aabb = pc.bbox();
        let aspect = size.width as f32 / size.height as f32;

        // Use embedded camera if available, otherwise compute default view
        let view_camera = if let Some(mut cam) = initial_camera {
            cam.fit_near_far(aabb);
            cam
        } else {
            PerspectiveCamera::new(
                aabb.center() - Vector3::new(1., 1., 1.) * aabb.radius() * 0.5,
                Quaternion::one(),
                PerspectiveProjection::new(
                    Vector2::new(size.width, size.height),
                    Vector2::new(Deg(45.), Deg(45. / aspect)),
                    0.01,
                    1000.,
                ),
            )
        };

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = pc.center();
        // controller.up = pc.up;
        let ui_renderer = ui_renderer::EguiWGPU::new(device, surface_format, &window);

        let display = Display::new(
            device,
            render_format,
            surface_format.remove_srgb_suffix(),
            size.width,
            size.height,
        );

        let stopwatch = if cfg!(not(target_arch = "wasm32")) {
            Some(GPUStopwatch::new(device, Some(3)))
        } else {
            None
        };

        Ok(Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            renderer,
            splatting_args: SplattingArgs {
                camera: view_camera,
                viewport: Vector2::new(size.width, size.height),
                gaussian_scaling: 1.,
                max_sh_deg: pc.sh_deg(),
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: Duration::ZERO,
                scene_center: None,
                scene_extend: None,
                background_color: wgpu::Color::BLACK,
            },
            pc,
            // camera: view_camera,
            controller,
            ui_renderer,
            fps: 0.,
            #[cfg(not(target_arch = "wasm32"))]
            history: RingBuffer::new(512),
            ui_visible: true,
            display,
            saved_cameras: Vec::new(),
            #[cfg(feature = "video")]
            cameras_save_path: "cameras_saved.json".to_string(),
            animation: None,
            scene: None,
            current_view: None,
            pointcloud_file_path: None,
            scene_file_path: None,

            stopwatch,
            initial_camera,
            downsampled_from,
        })
    }

    fn reload(&mut self) -> anyhow::Result<()> {
        if let Some(file_path) = &self.pointcloud_file_path {
            log::info!("reloading volume from {:?}", file_path);
            let file = std::fs::File::open(file_path)?;
            let mut pc_raw = io::GenericGaussianPointCloud::load(file)?;

            // Downsample if point cloud exceeds device capabilities
            let max_splats = self.wgpu_context.capabilities.max_splats as usize;
            self.downsampled_from = pc_raw.downsample(max_splats);

            self.pc = PointCloud::new(&self.wgpu_context.device, pc_raw)?;
        } else {
            return Err(anyhow::anyhow!("no pointcloud file path present"));
        }
        if let Some(scene_path) = &self.scene_file_path {
            log::info!("reloading scene from {:?}", scene_path);
            let file = std::fs::File::open(scene_path)?;

            self.set_scene(Scene::from_json(file)?);
        }
        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface
                .configure(&self.wgpu_context.device, &self.config);
            self.display
                .resize(&self.wgpu_context.device, new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
            self.splatting_args.viewport = Vector2::new(new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    /// returns whether redraw is required
    fn ui(&mut self) -> (bool, egui::FullOutput) {
        self.ui_renderer.begin_frame(&self.window);
        let request_redraw = ui::ui(self);

        let shapes = self.ui_renderer.end_frame(&self.window);

        (request_redraw, shapes)
    }

    /// returns whether the sceen changed and we need a redraw
    fn update(&mut self, dt: Duration) {
        // ema fps update

        if self.splatting_args.walltime < Duration::from_secs(5) {
            self.splatting_args.walltime += dt;
        }
        if let Some((next_camera, playing)) = &mut self.animation {
            if self.controller.user_inptut {
                self.cancle_animation()
            } else {
                let dt = if *playing { dt } else { Duration::ZERO };
                self.splatting_args.camera = next_camera.update(dt);
                self.splatting_args
                    .camera
                    .projection
                    .resize(self.config.width, self.config.height);
                if next_camera.done() {
                    self.animation.take();
                    self.controller.reset_to_camera(self.splatting_args.camera);
                }
            }
        } else {
            self.controller
                .update_camera(&mut self.splatting_args.camera, dt);

            // check if camera moved out of selected view
            if let Some(idx) = self.current_view {
                if let Some(scene) = &self.scene {
                    if let Some(camera) = scene.camera(idx) {
                        let scene_camera: PerspectiveCamera = camera.into();
                        if !self.splatting_args.camera.position.ulps_eq(
                            &scene_camera.position,
                            1e-4,
                            f32::default_max_ulps(),
                        ) || !self.splatting_args.camera.rotation.ulps_eq(
                            &scene_camera.rotation,
                            1e-4,
                            f32::default_max_ulps(),
                        ) {
                            self.current_view.take();
                        }
                    }
                }
            }
        }

        let aabb = self.pc.bbox();
        self.splatting_args.camera.fit_near_far(aabb);
    }

    fn render(
        &mut self,
        redraw_scene: bool,
        shapes: Option<FullOutput>,
    ) -> Result<(), wgpu::SurfaceError> {
        if let Some(s) = self.stopwatch.as_mut() {
            s.reset()
        }

        let output = self.surface.get_current_texture()?;
        let view_rgb = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let view_srgb = output.texture.create_view(&Default::default());
        // do prepare stuff

        let mut encoder =
            self.wgpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render command encoder"),
                });

        if redraw_scene {
            self.renderer.prepare(
                &mut encoder,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &self.pc,
                self.splatting_args,
                &mut self.stopwatch,
            );
        }

        let ui_state = shapes.map(|shapes| {
            self.ui_renderer.prepare(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &mut encoder,
                shapes,
            )
        });

        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.start(&mut encoder, "rasterization").unwrap();
        }
        if redraw_scene {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.display.texture(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.splatting_args.background_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            self.renderer.render(&mut render_pass, &self.pc);
        }
        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.stop(&mut encoder, "rasterization").unwrap();
        }

        self.display.render(
            &mut encoder,
            &view_rgb,
            self.splatting_args.background_color,
            self.renderer.camera(),
            self.renderer.render_settings(),
        );
        if let Some(s) = self.stopwatch.as_mut() {
            s.end(&mut encoder)
        }

        if let Some(state) = &ui_state {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass ui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view_srgb,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.ui_renderer.render(&mut render_pass, state);
        }

        if let Some(ui_state) = ui_state {
            self.ui_renderer.cleanup(ui_state)
        }
        self.wgpu_context.queue.submit([encoder.finish()]);

        output.present();
        self.splatting_args.viewport = Vector2::new(self.config.width, self.config.height);
        Ok(())
    }

    fn set_scene(&mut self, scene: Scene) {
        self.splatting_args.scene_extend = Some(scene.extend());
        let mut center = Point3::origin();
        for c in scene.cameras(None) {
            let z_axis: Vector3<f32> = c.rotation[2].into();
            center += Vector3::from(c.position) + z_axis * 2.;
        }
        center /= scene.num_cameras() as f32;

        self.controller.center = center;
        self.scene.replace(scene);
        if self.saved_cameras.is_empty() {
            self.saved_cameras = self
                .scene
                .as_ref()
                .unwrap()
                .cameras(Some(Split::Test))
                .clone();
        }
    }

    fn start_tracking_shot(&mut self) {
        if self.saved_cameras.len() > 1 {
            let shot = TrackingShot::from_cameras(self.saved_cameras.clone());
            let a = Animation::new(
                Duration::from_secs_f32(self.saved_cameras.len() as f32 * 2.),
                true,
                Box::new(shot),
            );
            self.animation = Some((a, true));
        }
    }

    fn cancle_animation(&mut self) {
        self.animation.take();
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn stop_animation(&mut self) {
        if let Some((_animation, playing)) = &mut self.animation {
            *playing = false;
        }
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn set_scene_camera(&mut self, i: usize) {
        if let Some(scene) = &self.scene {
            self.current_view.replace(i);
            log::info!("view moved to camera {i}");
            if let Some(camera) = scene.camera(i) {
                self.set_camera(camera, Duration::from_millis(200));
            } else {
                log::error!("camera {i} not found");
            }
        }
    }

    pub fn set_camera<C: Into<PerspectiveCamera>>(
        &mut self,
        camera: C,
        animation_duration: Duration,
    ) {
        let camera: PerspectiveCamera = camera.into();
        if animation_duration.is_zero() {
            self.update_camera(camera)
        } else {
            let target_camera = camera;
            let a = Animation::new(
                animation_duration,
                false,
                Box::new(Transition::new(
                    self.splatting_args.camera,
                    target_camera,
                    smoothstep,
                )),
            );
            self.animation = Some((a, true));
        }
    }

    fn update_camera(&mut self, camera: PerspectiveCamera) {
        self.splatting_args.camera = camera;
        self.splatting_args
            .camera
            .projection
            .resize(self.config.width, self.config.height);
    }

    /// Reset camera to the initial view (embedded camera from PLY or computed default)
    pub fn reset_view(&mut self) {
        if let Some(cam) = self.initial_camera {
            self.set_camera(cam, Duration::from_millis(300));
        }
    }

    /// Check if an initial camera (embedded in PLY) is available
    pub fn has_initial_camera(&self) -> bool {
        self.initial_camera.is_some()
    }

    /// Start a trajectory animation
    pub fn start_trajectory(&mut self, trajectory_type: TrajectoryType) {
        let params = TrajectoryParams {
            trajectory_type,
            ..Default::default()
        };

        // Compute min_depth from AABB
        let aabb = self.pc.bbox();
        let camera_to_center = self.splatting_args.camera.position - aabb.center();
        let min_depth = (camera_to_center.magnitude() - aabb.radius()).max(aabb.radius() / 10.0);

        // Compute viewport diagonal in normalized device coordinates
        let fovx = self.splatting_args.camera.projection.fovx.0;
        let fovy = self.splatting_args.camera.projection.fovy.0;
        let viewport_diagonal = ((fovx.tan()).powi(2) + (fovy.tan()).powi(2)).sqrt();

        let trajectory = TrajectoryAnimation::new(
            params.clone(),
            self.splatting_args.camera,
            aabb.center(),
            min_depth,
            viewport_diagonal,
        );

        let duration =
            Duration::from_secs_f32(params.duration_per_cycle_seconds * params.num_repeats as f32);

        let animation = Animation::new(duration, false, Box::new(trajectory));

        self.animation = Some((animation, true));
        log::info!("Started trajectory animation: {:?}", trajectory_type);
    }

    /// Reset to initial camera and optionally start a trajectory animation
    pub fn reset_and_replay(&mut self, trajectory_type: Option<TrajectoryType>) {
        if let Some(cam) = self.initial_camera {
            self.splatting_args.camera = cam;
            self.controller.reset_to_camera(cam);

            if let Some(traj_type) = trajectory_type {
                self.start_trajectory(traj_type);
            }
        }
    }

    fn save_view(&mut self) {
        let max_scene_id = if let Some(scene) = &self.scene {
            scene.cameras(None).iter().map(|c| c.id).max().unwrap_or(0)
        } else {
            0
        };
        let max_id = self.saved_cameras.iter().map(|c| c.id).max().unwrap_or(0);
        let id = max_id.max(max_scene_id) + 1;
        self.saved_cameras.push(SceneCamera::from_perspective(
            self.splatting_args.camera,
            id.to_string(),
            id,
            Vector2::new(self.config.width, self.config.height),
            Split::Test,
        ));
    }

    /// Load a new point cloud in-place, properly dropping the old one first
    #[cfg(target_arch = "wasm32")]
    fn load_new_pointcloud(&mut self, data: Vec<u8>, filename: String, compress: bool) {
        log::info!(
            "Loading new point cloud: {} ({} bytes, compress={})",
            filename,
            data.len(),
            compress
        );

        let device = &self.wgpu_context.device;

        // Parse the new point cloud
        let pc_reader = Cursor::new(data);
        let mut pc_raw = match io::GenericGaussianPointCloud::load(pc_reader) {
            Ok(pc) => pc,
            Err(e) => {
                log::error!("Failed to load point cloud: {:?}", e);
                return;
            }
        };

        // Downsample if point cloud exceeds device capabilities
        let max_splats = self.wgpu_context.capabilities.max_splats as usize;
        let new_downsampled_from = pc_raw.downsample(max_splats);

        // Compress if requested and not already compressed
        if compress && !pc_raw.compressed() {
            if let Err(e) = pc_raw.compress() {
                log::warn!("Failed to compress point cloud: {:?}", e);
            }
        }

        // Extract embedded camera before creating PointCloud
        let new_initial_camera: Option<PerspectiveCamera> =
            pc_raw.embedded_camera.clone().map(|c| c.into());

        // Update target canvas size from embedded camera
        if let Some(ref embedded) = pc_raw.embedded_camera {
            if let Ok(mut size) = TARGET_CANVAS_SIZE.lock() {
                *size = Some(embedded.image_size);
                log::info!(
                    "Set target canvas size from embedded camera: {}x{}",
                    embedded.image_size.0,
                    embedded.image_size.1
                );
            }
        } else {
            // Clear target canvas size if no embedded camera
            if let Ok(mut size) = TARGET_CANVAS_SIZE.lock() {
                *size = None;
            }
        }

        // Create new PointCloud (old one will be dropped)
        let new_pc = match PointCloud::new(device, pc_raw) {
            Ok(pc) => pc,
            Err(e) => {
                log::error!("Failed to create PointCloud: {:?}", e);
                return;
            }
        };

        log::info!(
            "Loaded point cloud with {:} points (compressed: {})",
            new_pc.num_points(),
            new_pc.compressed()
        );
        if let Some(original) = new_downsampled_from {
            log::warn!(
                "Point cloud was downsampled from {} to {} points due to device limits",
                original,
                new_pc.num_points()
            );
        }

        // Check if we need to recreate the renderer (compression state changed)
        let needs_new_renderer =
            self.pc.compressed() != new_pc.compressed() || self.pc.sh_deg() != new_pc.sh_deg();

        if needs_new_renderer {
            log::warn!(
                "Compression or SH degree changed (compressed: {} -> {}, sh_deg: {} -> {}). \
                 Please reload the page for proper rendering.",
                self.pc.compressed(),
                new_pc.compressed(),
                self.pc.sh_deg(),
                new_pc.sh_deg()
            );
            // Note: We can't easily recreate the renderer in WASM because
            // GaussianRenderer::new is async and we're in a sync context.
            // For now, we continue with the old renderer which may cause visual artifacts.
        }

        // Update state
        self.pc = new_pc;
        self.initial_camera = new_initial_camera;
        self.downsampled_from = new_downsampled_from;
        self.controller.center = self.pc.center();

        // Reset view to new point cloud
        let aabb = self.pc.bbox();
        if let Some(mut cam) = self.initial_camera {
            cam.fit_near_far(aabb);
            self.splatting_args.camera = cam;
        } else {
            let aspect = self.config.width as f32 / self.config.height as f32;
            self.splatting_args.camera = PerspectiveCamera::new(
                aabb.center() - Vector3::new(1., 1., 1.) * aabb.radius() * 0.5,
                Quaternion::one(),
                PerspectiveProjection::new(
                    Vector2::new(self.config.width, self.config.height),
                    Vector2::new(Deg(45.), Deg(45. / aspect)),
                    0.01,
                    1000.,
                ),
            );
        }
        self.controller.reset_to_camera(self.splatting_args.camera);

        // Reset animation and scene
        self.animation = None;
        self.scene = None;
        self.current_view = None;
        self.splatting_args.walltime = Duration::ZERO;

        log::info!("Point cloud reload complete");
    }
}

pub fn smoothstep(x: f32) -> f32 {
    x * x * (3.0 - 2.0 * x)
}

pub async fn open_window<R: Read + Seek + Send + Sync + 'static>(
    file: R,
    scene_file: Option<R>,
    config: RenderConfig,
    pointcloud_file_path: Option<PathBuf>,
    scene_file_path: Option<PathBuf>,
) {
    open_window_with_options(
        file,
        scene_file,
        config,
        pointcloud_file_path,
        scene_file_path,
        false,
    )
    .await
}

pub async fn open_window_with_options<R: Read + Seek + Send + Sync + 'static>(
    file: R,
    scene_file: Option<R>,
    config: RenderConfig,
    pointcloud_file_path: Option<PathBuf>,
    scene_file_path: Option<PathBuf>,
    compress: bool,
) {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let scene = scene_file.and_then(|f| match Scene::from_json(f) {
        Ok(s) => Some(s),
        Err(err) => {
            log::error!("cannot load scene: {:?}", err);
            None
        }
    });

    // let window_size = if let Some(scene) = &scene {
    //     let camera = scene.camera(0).unwrap();
    //     let factor = 1200. / camera.width as f32;
    //     LogicalSize::new(
    //         (camera.width as f32 * factor) as u32,
    //         (camera.height as f32 * factor) as u32,
    //     )
    // } else {
    //     LogicalSize::new(800, 600)
    // };
    let window_size = LogicalSize::new(800, 600);
    let window_attributes = Window::default_attributes()
        .with_inner_size(window_size)
        .with_title(format!(
            "{} ({})",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ));

    #[allow(deprecated)]
    let window = event_loop.create_window(window_attributes).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the viewer container
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                doc.get_element_by_id("loading-display")
                    .unwrap()
                    .set_text_content(Some("Unpacking"));
                doc.get_element_by_id("viewer-container")
            })
            .and_then(|container| {
                let canvas = window.canvas().unwrap();
                canvas.set_id("window-canvas");
                // Use container dimensions instead of body
                let container_element = container
                    .dyn_ref::<web_sys::HtmlElement>()
                    .expect("viewer-container should be an HtmlElement");
                // Get device pixel ratio for high-DPI displays
                let device_pixel_ratio = web_sys::window()
                    .map(|w| w.device_pixel_ratio())
                    .unwrap_or(1.0);
                canvas.set_width(
                    (container_element.client_width() as f64 * device_pixel_ratio) as u32,
                );
                canvas.set_height(
                    (container_element.client_height() as f64 * device_pixel_ratio) as u32,
                );
                let elm = web_sys::Element::from(canvas);
                elm.set_attribute("style", "width: 100%; height: 100%; display: block;")
                    .unwrap();
                container.append_child(&elm).ok()
            })
            .expect("couldn't append canvas to viewer-container");
    }

    // limit the redraw rate to the monitor refresh rate
    let min_wait = window
        .current_monitor()
        .map(|m| {
            let hz = m.refresh_rate_millihertz().unwrap_or(60_000);
            Duration::from_millis(1000000 / hz as u64)
        })
        .unwrap_or(Duration::from_millis(17));

    // Minimum frame time to prevent GPU overload (60 FPS cap)
    // This is used as a safety limit even when vsync is disabled
    const MIN_FRAME_TIME: Duration = Duration::from_millis(16);

    let mut state = WindowContext::new(window, file, &config, compress)
        .await
        .unwrap();
    state.pointcloud_file_path = pointcloud_file_path;

    if let Some(scene) = scene {
        state.set_scene(scene);
        state.set_scene_camera(0);
        state.scene_file_path = scene_file_path;
    }

    #[cfg(target_arch = "wasm32")]
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            doc.get_element_by_id("spinner")
                .unwrap()
                .set_attribute("style", "display:none;")
                .unwrap();
            doc.body()
        });

    let mut last = Instant::now();

    #[allow(deprecated)]
    event_loop.run(move |event, target| match event {
            Event::NewEvents(winit::event::StartCause::ResumeTimeReached { .. }) => {
                state.window.request_redraw();
            },
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() && !state.ui_renderer.on_event(&state.window,event) => match event {
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size, None);
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                ..
            } => {
                state.scale_factor = *scale_factor as f32;
            }
            WindowEvent::CloseRequested => {log::info!("close!");target.exit()},
            WindowEvent::ModifiersChanged(m)=>{
                state.controller.alt_pressed = m.state().alt_key();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key{
                if event.state == ElementState::Released{

                    if key == KeyCode::KeyT{
                        if state.animation.is_none(){
                            state.start_tracking_shot();
                        }else{
                            state.stop_animation()
                        }
                    }else if key == KeyCode::KeyU{
                        state.ui_visible = !state.ui_visible;
                    }else if key == KeyCode::KeyC{
                        state.save_view();
                    }else if key == KeyCode::KeyH{
                        // H for Home - reset to initial view
                        state.reset_view();
                    } else  if key == KeyCode::KeyR && state.controller.alt_pressed{
                        if let Err(err) = state.reload(){
                            log::error!("failed to reload volume: {:?}", err);
                        }
                    }else if let Some(scene) = &state.scene{
                        let new_camera = if let Some(num) = key_to_num(key){
                            Some(num as usize)
                        }
                        else if key == KeyCode::KeyR{
                            Some((rand::random::<u32>() as usize)%scene.num_cameras())
                        }else if key == KeyCode::KeyN{
                            scene.nearest_camera(state.splatting_args.camera.position,None)
                        }else if key == KeyCode::PageUp{
                            Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                        }else if key == KeyCode::KeyT{
                            Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                        }
                        else if key == KeyCode::PageDown{
                            Some(state.current_view.map_or(0, |v|v-1) % scene.num_cameras())
                        }else{None};

                        if let Some(new_camera) = new_camera{
                            state.set_scene_camera(new_camera);
                        }
                    }
                }
                state
                    .controller
                    .process_keyboard(key, event.state == ElementState::Pressed);
            }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                    state.controller.process_scroll(*dy )
                }
                winit::event::MouseScrollDelta::PixelDelta(p) => {
                    state.controller.process_scroll(p.y as f32 / 100.)
                }
            },
            WindowEvent::MouseInput { state:button_state, button, .. }=>{
                match button {
                    winit::event::MouseButton::Left =>                         state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
                    winit::event::MouseButton::Right => state.controller.right_mouse_pressed = *button_state == ElementState::Pressed,
                    _=>{}
                }
            }
            WindowEvent::Touch(touch) => {
                let touch_phase = match touch.phase {
                    WinitTouchPhase::Started => controller::TouchPhase::Started,
                    WinitTouchPhase::Moved => controller::TouchPhase::Moved,
                    WinitTouchPhase::Ended => controller::TouchPhase::Ended,
                    WinitTouchPhase::Cancelled => controller::TouchPhase::Cancelled,
                };

                let controller_touch = controller::Touch {
                    id: touch.id,
                    position: (touch.location.x as f32, touch.location.y as f32),
                    phase: touch_phase,
                };

                state.controller.process_touch(controller_touch);
            }
            WindowEvent::RedrawRequested => {
                // Check for pending file to load (WASM only)
                #[cfg(target_arch = "wasm32")]
                if let Ok(mut pending) = PENDING_FILE.try_lock() {
                    if let Some(file) = pending.take() {
                        state.load_new_pointcloud(file.data, file.filename, file.compress);
                    }
                }

                // Check for pending trajectory to start (WASM only)
                #[cfg(target_arch = "wasm32")]
                if let Ok(mut pending) = PENDING_TRAJECTORY.try_lock() {
                    if let Some(traj_type_str) = pending.take() {
                        if let Some(traj_type) = TrajectoryType::parse(&traj_type_str) {
                            state.start_trajectory(traj_type);
                        } else {
                            log::warn!("Unknown trajectory type: {}", traj_type_str);
                        }
                    }
                }

                // Check for pending reset and play (WASM only)
                #[cfg(target_arch = "wasm32")]
                if let Ok(mut pending) = PENDING_RESET_AND_PLAY.try_lock() {
                    if let Some(traj_type_str) = pending.take() {
                        let traj_type = TrajectoryType::parse(&traj_type_str);
                        state.reset_and_replay(traj_type);
                    }
                }

                // Always enforce a minimum frame time to prevent GPU overload
                // Use monitor refresh rate when vsync is enabled, otherwise use 60 FPS cap
                let frame_wait = if config.no_vsync {
                    MIN_FRAME_TIME
                } else {
                    min_wait.max(MIN_FRAME_TIME)
                };
                target.set_control_flow(ControlFlow::wait_duration(frame_wait));

                let now = Instant::now();
                let dt = now - last;
                last = now;

                let old_settings = state.splatting_args;
                state.update(dt);

                let (redraw_ui, shapes) = state.ui();

                let resolution_change = state.splatting_args.viewport != Vector2::new(state.config.width, state.config.height);

                let request_redraw = old_settings != state.splatting_args || resolution_change;

                if request_redraw || redraw_ui {
                    state.fps = (1. / dt.as_secs_f32()) * 0.05 + state.fps * 0.95;
                    match state.render(request_redraw, state.ui_visible.then_some(shapes)) {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size(), None),
                        // The system is out of memory, we should probably quit
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => println!("error: {:?}", e),
                    }
                }
            }
            _ => {}
        },
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion{ delta, },
            .. // We're not using device_id currently
        } => {
            state.controller.process_mouse(delta.0 as f32, delta.1 as f32)
        }
        _ => {},
    }).unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn run_wasm(
    pc: Vec<u8>,
    scene: Option<Vec<u8>>,
    pc_file: Option<String>,
    scene_file: Option<String>,
    compress: bool,
) {
    use std::str::FromStr;

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    let pc_reader = Cursor::new(pc);
    let scene_reader = scene.map(|d: Vec<u8>| Cursor::new(d));

    wasm_bindgen_futures::spawn_local(open_window_with_options(
        pc_reader,
        scene_reader,
        RenderConfig {
            no_vsync: false,
            hdr: false,
        },
        pc_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
        scene_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
        compress,
    ));
}

/// Load a new file in-place without page reload (WASM only)
/// This function queues the file data for the event loop to pick up
/// The old PointCloud will be properly dropped before the new one is created
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn load_new_file(data: Vec<u8>, filename: String, compress: bool) {
    log::info!(
        "Queueing new file for load: {} ({} bytes)",
        filename,
        data.len()
    );
    if let Ok(mut pending) = PENDING_FILE.lock() {
        *pending = Some(PendingFile {
            data,
            filename,
            compress,
        });
    } else {
        log::error!("Failed to lock PENDING_FILE mutex");
    }
}

/// Check if there's a target canvas size from the embedded camera
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn has_target_canvas_size() -> bool {
    if let Ok(size) = TARGET_CANVAS_SIZE.try_lock() {
        size.is_some()
    } else {
        false
    }
}

/// Get the target canvas size [width, height] from the embedded camera
/// Returns null if no embedded camera size is available
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_target_canvas_size() -> js_sys::Array {
    let arr = js_sys::Array::new();
    if let Ok(size) = TARGET_CANVAS_SIZE.try_lock() {
        if let Some((width, height)) = *size {
            arr.push(&wasm_bindgen::JsValue::from(width));
            arr.push(&wasm_bindgen::JsValue::from(height));
        }
    }
    arr
}

/// Get the list of available trajectory types
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_trajectory_types() -> js_sys::Array {
    let arr = js_sys::Array::new();
    arr.push(&wasm_bindgen::JsValue::from_str("rotate_forward"));
    arr.push(&wasm_bindgen::JsValue::from_str("swipe"));
    arr.push(&wasm_bindgen::JsValue::from_str("shake"));
    arr.push(&wasm_bindgen::JsValue::from_str("rotate"));
    arr.push(&wasm_bindgen::JsValue::from_str("forward"));
    arr
}

/// Start a trajectory animation
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start_trajectory(trajectory_type: String) {
    log::info!("Queueing trajectory: {}", trajectory_type);
    if let Ok(mut pending) = PENDING_TRAJECTORY.lock() {
        *pending = Some(trajectory_type);
    } else {
        log::error!("Failed to lock PENDING_TRAJECTORY mutex");
    }
}

/// Reset view and start a trajectory animation
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn reset_view_and_play(trajectory_type: String) {
    log::info!("Queueing reset and play: {}", trajectory_type);
    if let Ok(mut pending) = PENDING_RESET_AND_PLAY.lock() {
        *pending = Some(trajectory_type);
    } else {
        log::error!("Failed to lock PENDING_RESET_AND_PLAY mutex");
    }
}
