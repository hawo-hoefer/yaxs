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
    where
        T: Zero + One,
    {
        Mat4::new(
            T::one(), T::zero(), T::zero(), T::zero(),
            T::zero(), T::one(), T::zero(), T::zero(),
            T::zero(), T::zero(), T::one(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::one(),
        )
    }

    pub fn set_homog_translation(&mut self, x_new: T, y_new: T, z_new: T) {
        #[rustfmt::skip]
        let [
            _, _, _, x,
            _, _, _, y,
            _, _, _, z,
            _, _, _, _
        ] = &mut self.v;

        *x = x_new;
        *y = y_new;
        *z = z_new;
    }

    pub fn homog_mul(&self, rhs: Vec3<T>) -> Vec3<T>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Copy,
    {
        #[rustfmt::skip]
        let [
            a, b, c, x,
            d, e, f, y,
            g, h, i, z,
            _, _, _, _
        ] = self.v;
        Vec3::new(
            a * rhs.x + b * rhs.y + c * rhs.z + x,
            d * rhs.x + e * rhs.y + f * rhs.z + y,
            g * rhs.x + h * rhs.y + i * rhs.z + z,
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn homog_mul_only_translation() {
        let mut m = Mat4::<isize>::identity();
        m.set_homog_translation(1, 2, 3);
        let v = m.homog_mul(Vec3::new(1, 1, 1));
        assert_eq!(v, Vec3::new(2, 3, 4));
    }

    #[test]
    fn homog_mul_rot90_z() {
        #[rustfmt::skip]
        let m = Mat4::new(
            0, -1, 0, 0,
            1,  0, 0, 0,
            0,  0, 1, 0,
            0,  0, 0, 1,
        );
        let v = m.homog_mul(Vec3::new(1, 2, 3));
        assert_eq!(v, Vec3::new(-2, 1, 3));
    }
}
