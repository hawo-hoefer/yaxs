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

impl<'a, T> Iterator for Mat3Rows<'a, T>
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

    #[rustfmt::skip]
    pub fn transpose(&self) -> Self
    where
        T: Copy,
    {
        Mat3::new(
            self[(0, 0)], self[(1, 0)], self[(2, 0)],
            self[(0, 1)], self[(1, 1)], self[(2, 1)],
            self[(0, 2)], self[(1, 2)], self[(2, 2)],
        )
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

        &self.v[row + col * 3]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for Mat3<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < 3);
        assert!(col < 3);

        &mut self.v[row + col * 3]
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
        return Vec3::new(
            a * rhs.x + b * rhs.y + c * rhs.z,
            d * rhs.x + e * rhs.y + f * rhs.z,
            g * rhs.x + h * rhs.y + i * rhs.z,
        );
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
}
