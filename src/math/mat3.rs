use std::ops::{Add, Div, Mul, Neg, Sub};

use num_traits::{One, Zero};

use super::Vec3;

/// 3x3 matrix in row-major order
///
/// * `v`: matrix content
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mat3<T> {
    v: [T; 9],
}

pub struct Mat3Rows<'a, T> {
    mat: &'a Mat3<T>,
    n: usize,
}

impl<T> Iterator for Mat3Rows<'_, T>
where
    T: Copy,
{
    type Item = Vec3<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n >= 3 {
            return None;
        }

        let ret = Some(self.mat.row(self.n));
        self.n += 1;
        ret
    }
}

impl<T> Mat3<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        r0c0: T,
        r0c1: T,
        r0c2: T,
        r1c0: T,
        r1c1: T,
        r1c2: T,
        r2c0: T,
        r2c1: T,
        r2c2: T,
    ) -> Self {
        Self {
            v: [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2],
        }
    }

    pub fn identity() -> Self
    where
        T: Zero + One,
    {
        Self {
            #[rustfmt::skip]
                v: [
                    T::one(),  T::zero(), T::zero(),
                    T::zero(), T::one(),  T::zero(),
                    T::zero(), T::zero(), T::one(),
                ],
        }
    }

    pub fn zeros() -> Self
    where
        T: Zero,
    {
        Self {
            #[rustfmt::skip]
                v: [
                    T::zero(), T::zero(), T::zero(),
                    T::zero(), T::zero(), T::zero(),
                    T::zero(), T::zero(), T::zero(),
                ],
        }
    }

    pub fn transpose(&self) -> Self
    where
        T: Copy,
    {
        #[rustfmt::skip]
        let [
            a, b, c,
            d, e, f,
            g, h, i
        ] = self.v;

        Mat3::new(a, d, g, b, e, h, c, f, i)
    }

    #[rustfmt::skip]
    pub fn from_columns(cols: &[Vec3<T>; 3]) -> Mat3<T>
    where
        T: Copy,
    {
        Mat3::new(
            cols[0][0], cols[1][0], cols[2][0],
            cols[0][1], cols[1][1], cols[2][1],
            cols[0][2], cols[1][2], cols[2][2],
        )
    }

    #[rustfmt::skip]
    pub fn det(&self) -> T
    where T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Copy
    {
        let [
            a, b, c,
            d, e, f,
            g, h, i,
        ] = self.v;

        a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h
    }

    pub fn try_inverse(&self) -> Option<Mat3<T>>
    where
        T: Mul<T, Output = T>
            + Add<T, Output = T>
            + Sub<T, Output = T>
            + Copy
            + Zero
            + Div<T, Output = T>
            + Neg<Output = T>,
    {
        let det = self.det();
        if det.is_zero() {
            return None;
        }

        Some(self.adjugate() / det)
    }

    /// iterate the matrixes rows
    ///
    /// ```
    /// use yaxs::math::{Vec3, Mat3};
    ///
    /// let mat = Mat3::new(
    ///     0, 1, 2,
    ///     3, 4, 5,
    ///     6, 7, 8,
    /// );
    /// let mut ri = mat.row_iter();
    /// assert_eq!(ri.next(), Some(Vec3::new(0, 1, 2)));
    /// assert_eq!(ri.next(), Some(Vec3::new(3, 4, 5)));
    /// assert_eq!(ri.next(), Some(Vec3::new(6, 7, 8)));
    /// ```
    pub fn row_iter(&self) -> Mat3Rows<T> {
        Mat3Rows { mat: self, n: 0 }
    }

    pub fn row(&self, i: usize) -> Vec3<T>
    where
        T: Copy,
    {
        Vec3::new(self[(i, 0)], self[(i, 1)], self[(i, 2)])
    }

    #[rustfmt::skip]
    fn adjugate(&self) -> Mat3<T>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Neg<Output = T> + Copy,
    {
        let Mat3 {
            v: [
            a1, a2, a3,
            b1, b2, b3,
            c1, c2, c3
        ],
        } = *self;

        Mat3::new(
              b2 * c3 - c2 * b3 , -(a2 * c3 - c2 * a3),   a2 * b3 - b2 * a3,
            -(b1 * c3 - c1 * b3),   a1 * c3 - c1 * a3 , -(a1 * b3 - b1 * a3),
              b1 * c2 - c1 * b2 , -(a1 * c2 - c1 * a2),   a1 * b2 - b1 * a2,
        )
    }
}

impl<T> std::ops::Index<(usize, usize)> for Mat3<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < 3);
        assert!(col < 3);

        &self.v[row * 3 + col]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for Mat3<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < 3);
        assert!(col < 3);

        &mut self.v[row * 3 + col]
    }
}

impl<T> Div<T> for Mat3<T>
where
    T: Div<T, Output = T> + Copy,
{
    type Output = Mat3<T>;

    fn div(mut self, rhs: T) -> Self::Output {
        for v in self.v.iter_mut() {
            *v = *v / rhs;
        }

        self
    }
}

impl<T> std::ops::Mul<Vec3<T>> for Mat3<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Copy,
{
    type Output = Vec3<T>;

    fn mul(self, rhs: Vec3<T>) -> Self::Output {
        let [a, b, c, d, e, f, g, h, i] = self.v;
        // +-----+   +-+
        // |a,b,c|   |x|
        // |d,e,f| * |y|
        // |g,h,i|   |z|
        // +-----+   +-+
        Vec3::new(
            a * rhs.x + b * rhs.y + c * rhs.z,
            d * rhs.x + e * rhs.y + f * rhs.z,
            g * rhs.x + h * rhs.y + i * rhs.z,
        )
    }
}

impl<T> std::ops::Mul<&Vec3<T>> for Mat3<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Copy,
{
    type Output = Vec3<T>;

    fn mul(self, rhs: &Vec3<T>) -> Self::Output {
        // +-----+   +-+
        // |a,b,c|   |x|
        // |d,e,f| * |y|
        // |g,h,i|   |z|
        // +-----+   +-+
        Vec3::new(
            self[(0, 0)] * rhs.x + self[(0, 1)] * rhs.y + self[(0, 2)] * rhs.z,
            self[(1, 0)] * rhs.x + self[(1, 1)] * rhs.y + self[(1, 2)] * rhs.z,
            self[(2, 0)] * rhs.x + self[(2, 1)] * rhs.y + self[(2, 2)] * rhs.z,
        )
    }
}

impl<T> std::ops::Mul<T> for Mat3<T>
where
    T: Mul<T, Output = T> + Copy,
{
    type Output = Mat3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut ret = self;
        for v in ret.v.iter_mut() {
            *v = *v * rhs;
        }
        ret
    }
}

impl<'a, T> std::ops::Mul for &'a Mat3<T>
where
    &'a T: Mul<&'a T, Output = T> + Add<&'a T, Output = T> + Copy,
    T: Mul<T, Output = T> + Add<T, Output = T> + Copy,
{
    type Output = Mat3<T>;

    fn mul(self, rhs: &'a Mat3<T>) -> Self::Output {
        let [a1, b1, c1, d1, e1, f1, g1, h1, i1] = &self.v;
        let [a2, b2, c2, d2, e2, f2, g2, h2, i2] = &rhs.v;
        // +-----+   +-----+
        // |a,b,c|   |a,b,c|
        // |d,e,f| * |d,e,f|
        // |g,h,i|   |g,h,i|
        // +-----+   +-----+
        Mat3::new(
            a1 * a2 + b1 * d2 + c1 * g2,
            a1 * b2 + b1 * e2 + c1 * h2,
            a1 * c2 + b1 * f2 + c1 * i2,
            d1 * a2 + e1 * d2 + f1 * g2,
            d1 * b2 + e1 * e2 + f1 * h2,
            d1 * c2 + e1 * f2 + f1 * i2,
            g1 * a2 + h1 * d2 + i1 * g2,
            g1 * b2 + h1 * e2 + i1 * h2,
            g1 * c2 + h1 * f2 + i1 * i2,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn adjugate() {
        #[rustfmt::skip]
        let m = Mat3::new(
            -3.0,  2.0, -5.0,
            -1.0,  0.0, -2.0,
             3.0, -4.0,  1.0
        );
        let expected = Mat3::new(-8.0, 18.0, -4.0, -5.0, 12.0, -1.0, 4.0, -6.0, 2.0);
        assert_eq!(m.adjugate(), expected)
    }

    #[test]
    fn inverse_identity() {
        let m = Mat3::<f64>::identity();
        let inv = m.try_inverse().expect("identity has inverse");
        assert_eq!(inv, m)
    }

    #[test]
    fn double_inverse() {
        #[rustfmt::skip]
        let m = Mat3::<f64>::new(
            -3.0,  2.0, -5.0,
            -1.0,  0.0, -2.0,
             3.0, -4.0,  1.0
        );
        let inv = m.try_inverse().expect("matrix has inverse");
        let hopefully_m = inv.try_inverse().expect("matrix has inverse");
        for (a, b) in m.v.iter().zip(hopefully_m.v) {
            println!("{}, {}", a, b);
            assert!((a - b).abs() < 1e-12f64);
        }
    }

    #[test]
    fn vec_mat_mul_ident() {
        let v = Vec3::new(1, 2, 3);
        let res = Mat3::identity() * v;
        assert_eq!(v, res);
    }

    #[test]
    fn vec_mat_mul() {
        let v = Vec3::new(1, 2, 3);
        #[rustfmt::skip]
        let res = Mat3::new(
            1, 2, 1,
            2, 3, 1,
            4, 2, 2,
        ) * v;

        let expected = Vec3::new(8, 11, 14);
        assert_eq!(expected, res);
    }

    #[test]
    fn mat_mat_mul() {
        #[rustfmt::skip]
        let m1 = Mat3::new(
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        );
        #[rustfmt::skip]
        let m2 = Mat3::new(
            5, 4, 6,
            8, 7, 9,
            2, 1, 3,
        );

        #[rustfmt::skip]
        let expected = Mat3::new(
             27, 21,  33,
             72, 57,  87,
            117, 93, 141,
        );
        let res = &m1 * &m2;
        assert_eq!(res, expected);
    }

    #[test]
    fn transpose() {
        #[rustfmt::skip]
        let mat = Mat3::new(
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        );

        #[rustfmt::skip]
        let expected = Mat3::new(
            1, 4, 7,
            2, 5, 8,
            3, 6, 9,
        );

        let t = mat.transpose();
        assert_eq!(t, expected);
    }

    #[test]
    fn index() {
        #[rustfmt::skip]
        let m = Mat3::new(
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        );

        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(0, 1)], 2);
        assert_eq!(m[(0, 2)], 3);

        assert_eq!(m[(1, 0)], 4);
        assert_eq!(m[(1, 1)], 5);
        assert_eq!(m[(1, 2)], 6);

        assert_eq!(m[(2, 0)], 7);
        assert_eq!(m[(2, 1)], 8);
        assert_eq!(m[(2, 2)], 9);
    }
}
