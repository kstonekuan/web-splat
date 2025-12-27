use splines::{Interpolate, Key};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

use cgmath::{EuclideanSpace, InnerSpace, Matrix3, Point3, Quaternion, Rad, Vector3, VectorSpace};
use std::f32::consts::PI;

use crate::{camera::PerspectiveCamera, PerspectiveProjection};

/// Types of camera trajectory animations (matching ML#)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryType {
    /// Spiral/dolly motion: circular orbit + forward motion (default)
    RotateForward,
    /// Left-to-right horizontal pan
    Swipe,
    /// Horizontal shake then vertical shake (sinusoidal)
    Shake,
    /// Circular orbit around scene
    Rotate,
    /// Pure dolly forward/backward motion
    Forward,
}

impl TrajectoryType {
    /// Parse a trajectory type from a string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rotate_forward" | "rotateforward" => Some(Self::RotateForward),
            "swipe" => Some(Self::Swipe),
            "shake" => Some(Self::Shake),
            "rotate" => Some(Self::Rotate),
            "forward" => Some(Self::Forward),
            _ => None,
        }
    }
}

/// Parameters for trajectory animation (matching ML#)
#[derive(Clone, Debug)]
pub struct TrajectoryParams {
    /// Type of trajectory
    pub trajectory_type: TrajectoryType,
    /// Maximum lateral camera offset as fraction of depth (default: 0.08)
    pub max_disparity: f32,
    /// Maximum zoom/forward motion as fraction of depth (default: 0.15)
    pub max_zoom: f32,
    /// Number of animation steps (default: 60)
    pub num_steps: u32,
    /// Number of times to repeat the pattern (default: 1)
    pub num_repeats: u32,
    /// Duration per cycle in seconds (default: 2.0)
    pub duration_per_cycle_seconds: f32,
}

impl Default for TrajectoryParams {
    fn default() -> Self {
        Self {
            trajectory_type: TrajectoryType::RotateForward,
            max_disparity: 0.08,
            max_zoom: 0.15,
            num_steps: 60,
            num_repeats: 1,
            duration_per_cycle_seconds: 2.0,
        }
    }
}

/// Camera trajectory animation that samples camera positions along a trajectory
pub struct TrajectoryAnimation {
    params: TrajectoryParams,
    base_camera: PerspectiveCamera,
    /// Maximum offset in camera-local space [lateral_x, lateral_y, medial_z]
    max_offset: Vector3<f32>,
    /// Point to look at (scene center)
    look_at_point: Point3<f32>,
    /// Camera's local coordinate frame (columns are right, up, forward vectors)
    camera_frame: Matrix3<f32>,
}

impl TrajectoryAnimation {
    /// Create a new trajectory animation
    ///
    /// # Arguments
    /// * `params` - Trajectory parameters
    /// * `base_camera` - Starting camera position/orientation
    /// * `scene_center` - Center of the scene to look at
    /// * `min_depth` - Minimum depth from camera to scene (for offset scaling)
    /// * `viewport_diagonal` - Diagonal of viewport in normalized device coordinates
    pub fn new(
        params: TrajectoryParams,
        base_camera: PerspectiveCamera,
        scene_center: Point3<f32>,
        min_depth: f32,
        viewport_diagonal: f32,
    ) -> Self {
        // Compute max offsets like ML#:
        // max_lateral = max_disparity * diagonal * min_depth
        // max_medial = max_zoom * min_depth
        let max_lateral = params.max_disparity * viewport_diagonal * min_depth;
        let max_medial = params.max_zoom * min_depth;
        let max_offset = Vector3::new(max_lateral, max_lateral, max_medial);

        // Get camera's local coordinate frame from rotation
        // The rotation transforms from camera space to world space
        let camera_frame: Matrix3<f32> = base_camera.rotation.into();

        Self {
            params,
            base_camera,
            max_offset,
            look_at_point: scene_center,
            camera_frame,
        }
    }

    /// Compute the eye offset in camera-local space for a given normalized time
    fn compute_eye_offset(&self, t: f32) -> Vector3<f32> {
        match self.params.trajectory_type {
            TrajectoryType::Swipe => self.compute_swipe_offset(t),
            TrajectoryType::Shake => self.compute_shake_offset(t),
            TrajectoryType::Rotate => self.compute_rotate_offset(t),
            TrajectoryType::RotateForward => self.compute_rotate_forward_offset(t),
            TrajectoryType::Forward => self.compute_forward_offset(t),
        }
    }

    /// Swipe: Linear left-to-right horizontal pan
    fn compute_swipe_offset(&self, t: f32) -> Vector3<f32> {
        // Linear motion along X-axis: [-max, +max]
        let x = self.max_offset.x * (2.0 * t - 1.0);
        Vector3::new(x, 0.0, 0.0)
    }

    /// Shake: Horizontal shake then vertical shake (sinusoidal)
    fn compute_shake_offset(&self, t: f32) -> Vector3<f32> {
        if t < 0.5 {
            // First half: horizontal sine wave
            let phase = t * 2.0;
            let x = self.max_offset.x * (2.0 * PI * phase).sin();
            Vector3::new(x, 0.0, 0.0)
        } else {
            // Second half: vertical sine wave
            let phase = (t - 0.5) * 2.0;
            let y = self.max_offset.y * (2.0 * PI * phase).sin();
            Vector3::new(0.0, y, 0.0)
        }
    }

    /// Rotate: Circular orbit around scene
    fn compute_rotate_offset(&self, t: f32) -> Vector3<f32> {
        let angle = 2.0 * PI * t;
        Vector3::new(
            self.max_offset.x * angle.sin(),
            self.max_offset.y * angle.cos(),
            0.0,
        )
    }

    /// RotateForward: Spiral/dolly motion (circular + forward)
    fn compute_rotate_forward_offset(&self, t: f32) -> Vector3<f32> {
        let angle = 2.0 * PI * t;
        Vector3::new(
            self.max_offset.x * angle.sin(),
            0.0,
            self.max_offset.z * (1.0 - angle.cos()) / 2.0,
        )
    }

    /// Forward: Pure dolly forward/backward motion
    fn compute_forward_offset(&self, t: f32) -> Vector3<f32> {
        Vector3::new(0.0, 0.0, self.max_offset.z * t)
    }

    /// Compute look-at rotation for a given position
    fn compute_look_at_rotation(&self, position: Point3<f32>) -> Quaternion<f32> {
        let forward = (self.look_at_point - position).normalize();
        // Use the camera's original up vector
        let original_up = self.camera_frame.y;

        // Compute right vector
        let right = forward.cross(original_up).normalize();
        // Recompute up to ensure orthogonality
        let up = right.cross(forward).normalize();

        // Build rotation matrix (camera looks along -Z in camera space)
        let rotation_matrix = Matrix3::from_cols(right, up, -forward);

        // Convert to quaternion
        Quaternion::from(rotation_matrix)
    }
}

impl Sampler for TrajectoryAnimation {
    type Sample = PerspectiveCamera;

    fn sample(&self, t: f32) -> PerspectiveCamera {
        // Handle multiple repeats: t goes from 0 to 1 for the full animation
        // We want each repeat to go through the full cycle
        let cycle_t = (t * self.params.num_repeats as f32).fract();
        // Handle the edge case where t = 1.0 exactly
        let cycle_t = if t >= 1.0 { 1.0 } else { cycle_t };

        // Get offset in camera-local space
        let local_offset = self.compute_eye_offset(cycle_t);

        // Transform offset from camera-local to world space
        let world_offset = self.camera_frame * local_offset;

        // Compute new position
        let new_position = self.base_camera.position + world_offset;

        // Compute new rotation looking at scene center
        let new_rotation = self.compute_look_at_rotation(new_position);

        PerspectiveCamera {
            position: new_position,
            rotation: new_rotation,
            projection: self.base_camera.projection,
        }
    }
}

pub trait Lerp {
    fn lerp(&self, other: &Self, amount: f32) -> Self;
}

pub trait Sampler {
    type Sample;

    fn sample(&self, v: f32) -> Self::Sample;
}

pub struct Transition<T> {
    from: T,
    to: T,
    interp_fn: fn(f32) -> f32,
}
impl<T: Lerp + Clone> Transition<T> {
    pub fn new(from: T, to: T, interp_fn: fn(f32) -> f32) -> Self {
        Self {
            from,
            to,
            interp_fn,
        }
    }
}

impl<T: Lerp + Clone> Sampler for Transition<T> {
    type Sample = T;
    fn sample(&self, v: f32) -> Self::Sample {
        self.from.lerp(&self.to, (self.interp_fn)(v))
    }
}

pub struct TrackingShot {
    spline: splines::Spline<f32, PerspectiveCamera>,
}

impl TrackingShot {
    pub fn from_cameras<C>(cameras: Vec<C>) -> Self
    where
        C: Into<PerspectiveCamera>,
    {
        let cameras: Vec<PerspectiveCamera> = cameras.into_iter().map(|c| c.into()).collect();

        let last_two = cameras.iter().skip(cameras.len() - 2).take(2);
        let first_two = cameras.iter().take(2);
        let spline = splines::Spline::from_iter(
            last_two
                .chain(cameras.iter())
                .chain(first_two)
                .enumerate()
                .map(|(i, c)| {
                    let v = (i as f32 - 1.) / (cameras.len()) as f32;
                    Key::new(v, *c, splines::Interpolation::CatmullRom)
                }),
        );

        Self { spline }
    }

    pub fn num_control_points(&self) -> usize {
        self.spline.len()
    }
}

impl Sampler for TrackingShot {
    type Sample = PerspectiveCamera;
    fn sample(&self, v: f32) -> Self::Sample {
        match self.spline.sample(v) {
            Some(p) => p,
            None => panic!("spline sample failed at {}", v),
        }
    }
}

impl Interpolate<f32> for PerspectiveCamera {
    fn step(t: f32, threshold: f32, a: Self, b: Self) -> Self {
        if t < threshold {
            a
        } else {
            b
        }
    }

    fn lerp(t: f32, a: Self, b: Self) -> Self {
        Self {
            position: Point3::from_vec(a.position.to_vec().lerp(b.position.to_vec(), t)),
            rotation: a.rotation.slerp(b.rotation, t),
            projection: a.projection.lerp(&b.projection, t),
        }
    }

    fn cosine(_t: f32, _a: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_hermite(
        t: f32,
        x: (f32, Self),
        a: (f32, Self),
        b: (f32, Self),
        y: (f32, Self),
    ) -> Self {
        // unroll quaternion rotations so that the animation always takes the shortest path
        // this is just a hack...
        let q_unrolled = unroll([x.1.rotation, a.1.rotation, b.1.rotation, y.1.rotation]);
        Self {
            position: Point3::from_vec(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.position.to_vec()),
                (a.0, a.1.position.to_vec()),
                (b.0, b.1.position.to_vec()),
                (y.0, y.1.position.to_vec()),
            )),
            rotation: Interpolate::cubic_hermite(
                t,
                (x.0, q_unrolled[0]),
                (a.0, q_unrolled[1]),
                (b.0, q_unrolled[2]),
                (y.0, q_unrolled[3]),
            )
            .normalize(),
            projection: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.projection),
                (a.0, a.1.projection),
                (b.0, b.1.projection),
                (y.0, y.1.projection),
            ),
        }
    }

    fn quadratic_bezier(_t: f32, _a: Self, _u: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier_mirrored(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }
}

impl Interpolate<f32> for PerspectiveProjection {
    fn step(t: f32, threshold: f32, a: Self, b: Self) -> Self {
        if t < threshold {
            a
        } else {
            b
        }
    }

    fn lerp(t: f32, a: Self, b: Self) -> Self {
        a.lerp(&b, t)
    }

    fn cosine(_t: f32, _a: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_hermite(
        t: f32,
        x: (f32, Self),
        a: (f32, Self),
        b: (f32, Self),
        y: (f32, Self),
    ) -> Self {
        Self {
            fovx: Rad(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fovx.0),
                (a.0, a.1.fovx.0),
                (b.0, b.1.fovx.0),
                (y.0, y.1.fovx.0),
            )),
            fovy: Rad(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fovy.0),
                (a.0, a.1.fovy.0),
                (b.0, b.1.fovy.0),
                (y.0, y.1.fovy.0),
            )),
            znear: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.znear),
                (a.0, a.1.znear),
                (b.0, b.1.znear),
                (y.0, y.1.znear),
            ),
            zfar: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.zfar),
                (a.0, a.1.zfar),
                (b.0, b.1.zfar),
                (y.0, y.1.zfar),
            ),
            fov2view_ratio: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fov2view_ratio),
                (a.0, a.1.fov2view_ratio),
                (b.0, b.1.fov2view_ratio),
                (y.0, y.1.fov2view_ratio),
            ),
        }
    }

    fn quadratic_bezier(_t: f32, _a: Self, _u: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier_mirrored(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }
}

pub struct Animation<T> {
    duration: Duration,
    time_left: Duration,
    looping: bool,
    sampler: Box<dyn Sampler<Sample = T>>,
}

impl<T> Animation<T> {
    pub fn new(duration: Duration, looping: bool, sampler: Box<dyn Sampler<Sample = T>>) -> Self {
        Self {
            duration,
            time_left: duration,
            looping,
            sampler,
        }
    }

    pub fn done(&self) -> bool {
        if self.looping {
            false
        } else {
            self.time_left.is_zero()
        }
    }

    pub fn update(&mut self, dt: Duration) -> T {
        match self.time_left.checked_sub(dt) {
            Some(new_left) => {
                // set time left
                self.time_left = new_left;
            }
            None => {
                if self.looping {
                    self.time_left = self.duration + self.time_left - dt;
                } else {
                    self.time_left = Duration::ZERO;
                }
            }
        }
        self.sampler.sample(self.progress())
    }

    pub fn progress(&self) -> f32 {
        1. - self.time_left.as_secs_f32() / self.duration.as_secs_f32()
    }

    pub fn set_progress(&mut self, v: f32) {
        self.time_left = self.duration.mul_f32(1. - v);
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }

    pub fn set_duration(&mut self, duration: Duration) {
        let progress = self.progress();
        self.duration = duration;
        self.set_progress(progress);
    }
}

/// unroll quaternion rotations so that the animation always takes the shortest path
fn unroll(rot: [Quaternion<f32>; 4]) -> [Quaternion<f32>; 4] {
    let mut rot = rot;
    if rot[0].s < 0. {
        rot[0] = -rot[0];
    }
    for i in 1..4 {
        if rot[i].dot(rot[i - 1]) < 0. {
            rot[i] = -rot[i];
        }
    }
    rot
}
