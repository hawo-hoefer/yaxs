use super::linalg::{Vec3, Vec4};

#[derive(Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quaternion {
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    pub fn from_axis_angle(x: f32, y: f32, z: f32, alpha: f32) -> Self
where {
        let v = Vec3::new(x, y, z).normalize();

        let alpha_half_sin = (0.5 * alpha).sin();

        Self::new(
            (alpha / 2.0).cos(),
            v[0] * alpha_half_sin,
            v[1] * alpha_half_sin,
            v[2] * alpha_half_sin,
        )
    }
    /// treating self as quaternion, compute the quaternion conjugate
    pub fn conjugate(&self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    pub fn unit_recip_unchecked(&self) -> Self {
        self.conjugate()
    }

    pub fn magnitude(&self) -> f32 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn scale_inplace(&mut self, s: f32) {
        self.w *= s;
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }

    /// compute quaternion reciprocal
    ///
    pub fn recip(&self) -> Self {
        let mut conjug = self.conjugate();
        let mag = conjug.magnitude();
        // TODO: do something here, maybe sqinv or something
        conjug.scale_inplace(1.0 / (mag * mag));
        conjug
    }

    /// rotate v by self
    ///
    /// $$v' = q v q^{-1}$$
    ///
    /// * `v`: vector to rotate
    pub fn quaternion_transform(&self, v: &Vec3<f32>) -> Vec3<f32> {
        let v = Self::new(0.0, v[0], v[1], v[2]);
        let q_tf = self.quaternion_transform_quaternion(&v);
        Vec3::new(q_tf.x, q_tf.y, q_tf.z)
    }

    /// rotate v by self without normalization
    ///
    /// $$v' = q v q^{-1}$$
    ///
    /// * `v`: vector to rotate
    pub fn unit_transform_unchecked(&self, v: &Vec3<f32>) -> Vec3<f32> {
        let v = Self::new(0.0, v[0], v[1], v[2]);

        let q_tf = self.unit_quat_tf_unchecked(&v);

        Vec3::new(q_tf.x, q_tf.y, q_tf.z)
    }

    /// interpreting self as a quaternion, rotate v, using the algorithm shown below
    ///
    /// is somehow slower than what I had before.
    ///
    /// <https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/>
    ///
    /// $$v' = q v q^{-1}$$
    ///
    /// * `v`: vector to rotate
    #[deprecated]
    pub fn unit_quaternion_transform_unchecked_alt(&self, v: &Vec3<f32>) -> Vec3<f32> {
        let qxyz = Vec3::new(self.x, self.y, self.z);
        let t = qxyz.cross(v);
        let v_ = v + &t.scale(self.w) + qxyz.cross(&t);

        v_
    }

    pub fn unit_quat_tf_unchecked(&self, v: &Self) -> Self {
        let recip = self.unit_recip_unchecked();

        self.hamilton_product(&v).hamilton_product(&recip)
    }

    pub fn quaternion_transform_quaternion(&self, v: &Self) -> Self {
        let recip = self.recip();

        self.hamilton_product(&v).hamilton_product(&recip)
    }

    pub fn get_ptr(&self) -> *const f32 {
        &self.w as *const f32
    }

    pub fn get_mut_ptr(&mut self) -> *mut f32 {
        &mut self.w as *mut f32
    }

    pub fn hamilton_product(&self, rhs: &Self) -> Self {
        Self::new(
            self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        )
    }
}

impl From<Vec4<f32>> for Quaternion {
    fn from(v: Vec4<f32>) -> Self {
        // TODO: reinterpret?
        Self::new(v[0], v[1], v[2], v[3])
    }
}

impl From<Vec4<f64>> for Quaternion {
    fn from(v: Vec4<f64>) -> Self {
        // TODO: reinterpret?
        Self::new(v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32)
    }
}

impl std::fmt::Display for Quaternion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(6);
        let precision = f.precision().unwrap_or(3);
        write!(
            f,
            "Quaternion({:w$.p$}, {:w$.p$}, {:w$.p$}, {:w$.p$})",
            self.w,
            self.x,
            self.y,
            self.z,
            w = width,
            p = precision
        )
    }
}

#[cfg(test)]
mod test {
    use crate::math::linalg::Vec3;

    use super::Quaternion;

    #[test]
    fn quaternion_conjugate_inverse() {
        let q = Quaternion::new(1.0, -1.0, 2.4, 3.3);
        let qconj = q.conjugate();
        assert_eq!(qconj.conjugate(), q);
    }

    #[test]
    fn quaternion_conjugate_norm() {
        let q = Quaternion::new(1.0, -1.0, 2.4, 3.3);
        let qconj = q.conjugate();
        let prod = q.hamilton_product(&qconj);
        let atol = 1e-7;
        assert!(prod.x.abs() < atol);
        assert!(prod.y.abs() < atol);
        assert!(prod.z.abs() < atol);

        assert!((prod.w - q.magnitude().powi(2)).abs() < 1e-7)
    }

    #[test]
    fn quaternion_reciprocal_identity() {
        let r0 = Quaternion::from_axis_angle(0.3, 2.0, 1.0, 32.0f32.to_radians());
        let r1 = r0.recip();
        let prod = r0.hamilton_product(&r1);
        assert!((prod.w - 1.0).abs() < 1e-7);
        assert!((prod.x - 0.0).abs() < 1e-7);
        assert!((prod.y - 0.0).abs() < 1e-7);
        assert!((prod.z - 0.0).abs() < 1e-7);
    }

    #[test]
    fn quaternion_angle_axis() {
        let atol = 1e-3;
        let q = Quaternion::from_axis_angle(0.0, 1.0, 0.0, 32.0f32.to_radians());
        assert!((q.w - 0.961).abs() < atol, "{}, {}", q.w, 0.961);
        assert!((q.x - 0.0).abs() < atol, "{}, {}", q.x, 0.0);
        assert!((q.y - 0.276).abs() < atol, "{}, {}", q.y, 0.276);
        assert!((q.z - 0.0).abs() < atol, "{}, {}", q.w, 0.0);
    }

    #[test]
    fn quaternion_rot() {
        let atol = 1e-6;
        let q = Quaternion::from_axis_angle(0.0, 1.0, 0.0, 32.0f32.to_radians());
        let rot = q.quaternion_transform(&Vec3::<f32>::new(5.0, 7.0, 1.0));

        assert!((rot[0] - 4.77016).abs() < atol, "{}, {}", rot[0], 4.77016);
        assert!((rot[1] - 7.0).abs() < atol, "{}, {}", rot[1], 7.0);
        assert!((rot[2] - -1.801548).abs() < atol, "{}, {}", rot[2], -1.8015);
    }
}
