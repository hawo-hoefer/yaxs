use std::iter::Map;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::Mat3;
use itertools::Itertools;
use num_traits::{One, Zero};
use ordered_float::Float;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Copy, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub struct VecIter<'a, T> {
    vec: &'a Vec3<T>,
    n: usize,
}

impl<'a, T> Iterator for VecIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n >= 3 {
            return None;
        }

        let ret = &self.vec[self.n];
        self.n += 1;
        Some(ret)
    }
}

impl<'a, T> IntoIterator for &'a Vec3<T> {
    type IntoIter = VecIter<'a, T>;

    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        VecIter { vec: self, n: 0 }
    }
}

impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    pub fn map<V>(&self, p: impl Fn(&T) -> V) -> Vec3<V> {
        Vec3::new(p(&self.x), p(&self.y), p(&self.z))
    }

    pub fn dot(&self, rhs: &Self) -> T
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Copy,
    {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(&self, rhs: &Self) -> Self
    where
        T: Mul<T, Output = T> + Sub<T, Output = T> + Copy,
    {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn magnitude(&self) -> T
    where
        T: Float,
    {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl<T> std::ops::Index<usize> for Vec3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index must be smaller than 3"),
        }
    }
}

impl<T> std::ops::Add for Vec3<T>
where
    T: Add<T, Output = T>,
{
    type Output = Vec3<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl<T> std::ops::Add<T> for Vec3<T>
where
    T: Add<T, Output = T> + Copy,
{
    type Output = Vec3<T>;

    fn add(self, rhs: T) -> Self::Output {
        Vec3::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl<T> std::ops::Sub<T> for Vec3<T>
where
    T: Sub<T, Output = T> + Copy,
{
    type Output = Vec3<T>;

    fn sub(self, rhs: T) -> Self::Output {
        Vec3::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl<T> std::ops::Mul<T> for Vec3<T>
where
    T: Mul<T, Output = T> + Copy,
{
    type Output = Vec3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl<T> Neg for Vec3<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl<T> std::ops::Div<T> for Vec3<T>
where
    T: Div<T, Output = T> + Copy,
{
    type Output = Vec3<T>;

    fn div(self, rhs: T) -> Self::Output {
        Vec3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl<T> std::ops::Sub for Vec3<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Vec3<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
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
        return Vec3::new(
            self[(0, 0)] * rhs.x + self[(0, 1)] * rhs.y + self[(0, 2)] * rhs.z,
            self[(1, 0)] * rhs.x + self[(1, 1)] * rhs.y + self[(1, 2)] * rhs.z,
            self[(2, 0)] * rhs.x + self[(2, 1)] * rhs.y + self[(2, 2)] * rhs.z,
        );
    }
}

impl<T> std::ops::Mul<T> for Mat3<T>
where
    T: Mul<T, Output = T> + Copy,
{
    type Output = Mat3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Mat3::new(
            self[(0, 0)] * rhs,
            self[(0, 1)] * rhs,
            self[(0, 2)] * rhs,
            self[(1, 0)] * rhs,
            self[(1, 1)] * rhs,
            self[(1, 2)] * rhs,
            self[(2, 0)] * rhs,
            self[(2, 1)] * rhs,
            self[(2, 2)] * rhs,
        )
    }
}

impl<T> std::ops::Mul for Mat3<T>
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Copy,
{
    type Output = Mat3<T>;

    fn mul(self, rhs: Mat3<T>) -> Self::Output {
        // +-----+   +-----+
        // |a,b,c|   |a,b,c|
        // |d,e,f| * |d,e,f|
        // |g,h,i|   |g,h,i|
        // +-----+   +-----+
        Mat3::new(
            self[(0, 0)] * rhs[(0, 0)] + self[(0, 1)] * rhs[(1, 0)] + self[(0, 2)] * rhs[(2, 0)],
            self[(0, 0)] * rhs[(0, 1)] + self[(0, 1)] * rhs[(1, 1)] + self[(0, 2)] * rhs[(2, 1)],
            self[(0, 0)] * rhs[(0, 2)] + self[(0, 1)] * rhs[(1, 2)] + self[(0, 2)] * rhs[(2, 2)],
            self[(1, 0)] * rhs[(0, 0)] + self[(1, 1)] * rhs[(1, 0)] + self[(1, 2)] * rhs[(2, 0)],
            self[(1, 0)] * rhs[(0, 1)] + self[(1, 1)] * rhs[(1, 1)] + self[(1, 2)] * rhs[(2, 1)],
            self[(1, 0)] * rhs[(0, 2)] + self[(1, 1)] * rhs[(1, 2)] + self[(1, 2)] * rhs[(2, 2)],
            self[(2, 0)] * rhs[(0, 0)] + self[(2, 1)] * rhs[(1, 0)] + self[(2, 2)] * rhs[(2, 0)],
            self[(2, 0)] * rhs[(0, 1)] + self[(2, 1)] * rhs[(1, 1)] + self[(2, 2)] * rhs[(2, 1)],
            self[(2, 0)] * rhs[(0, 2)] + self[(2, 1)] * rhs[(1, 2)] + self[(2, 2)] * rhs[(2, 2)],
        )
    }
}

impl<'de, T> Deserialize<'de> for Vec3<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (a, b, c) = <(T, T, T)>::deserialize(deserializer)?;
        Ok(Vec3::new(a, b, c))
    }
}
