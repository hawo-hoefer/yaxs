use std::ops::{Add, Mul};

use num_traits::{One, Zero};

use super::Vec3;

#[derive(Debug, PartialEq, Eq)]
pub struct Mat4<T> {
    v: [T; 16],
}

impl<T> Mat4<T> {
    pub fn new(
        r0c0: T,
        r0c1: T,
        r0c2: T,
        r0c3: T,
        r1c0: T,
        r1c1: T,
        r1c2: T,
        r1c3: T,
        r2c0: T,
        r2c1: T,
        r2c2: T,
        r2c3: T,
        r3c0: T,
        r3c1: T,
        r3c2: T,
        r3c3: T,
    ) -> Self {
        Self {
            v: [
                r0c0, r0c1, r0c2, r0c3, r1c0, r1c1, r1c2, r1c3, r2c0, r2c1, r2c2, r2c3, r3c0, r3c1,
                r3c2, r3c3,
            ],
        }
    }


        #[rustfmt::skip]
    pub fn identity() -> Mat4<T> 
            where T: Zero + One
        {
            Mat4::new(
                T::one(),  T::zero(), T::zero(), T::zero(),
                T::zero(), T::one(),  T::zero(), T::zero(),
                T::zero(), T::zero(), T::one(),  T::zero(),
                T::zero(), T::zero(), T::zero(), T::one(),
            )
        }
    pub fn homog_mul(&self, rhs: Vec3<T>) -> Vec3<T>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Copy,
    {
        Vec3::new(
            self[(0, 0)] * rhs.x + self[(0, 1)] * rhs.y + self[(0, 2)] * rhs.z + rhs.x,
            self[(1, 0)] * rhs.x + self[(1, 1)] * rhs.y + self[(1, 2)] * rhs.z + rhs.y,
            self[(2, 0)] * rhs.x + self[(2, 1)] * rhs.y + self[(2, 2)] * rhs.z + rhs.z,
        )
    }
}

impl<T> std::ops::Index<(usize, usize)> for Mat4<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < 4);
        assert!(col < 4);

        &self.v[row + col * 4]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for Mat4<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < 4);
        assert!(col < 4);

        &mut self.v[row + col * 4]
    }
}
