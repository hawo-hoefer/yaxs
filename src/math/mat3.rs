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
        self[(0, 0)] * self[(1, 1)] * self[(2, 2)] 
        + self[(0, 1)] * self[(1, 2)] * self[(2, 0)]
        + self[(0, 2)] * self[(1, 0)] * self[(2, 1)]

        - self[(0, 2)] * self[(1, 1)] * self[(2, 0)]
        - self[(0, 1)] * self[(1, 0)] * self[(2, 2)]
        - self[(0, 0)] * self[(1, 2)] * self[(2, 1)]
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
        Mat3Rows {
            mat: self,
            n: 0,
        }
    }

    pub fn row(&self, i: usize) -> Vec3<T>
    where
        T: Copy,
    {
        Vec3::new(self[(i, 0)], self[(i, 1)], self[(i, 2)])
    }

    fn adjugate(&self) -> Mat3<T>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Neg<Output = T> + Copy,
    {
        let Mat3 {
            v: [a1, a2, a3, b1, b2, b3, c1, c2, c3],
        } = *self;

        Mat3::new(
            b2 * c3 - c2 * b3,
            -(b1 * c3 - c1 * b3),
            b1 * c2 - c1 * b2,
            -(a2 * c3 - c2 * a3),
            a1 * c3 - c1 * a3,
            -(a1 * c2 - c1 * a2),
            a2 * b3 - b2 * a3,
            -(a1 * b3 + b1 * a3),
            a1 * b2 - b1 * a2,
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
