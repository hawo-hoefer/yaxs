use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

use serde::de::{self, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};

use num_traits::{Float, One, Zero};

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
#[repr(C, align(32))]
pub struct Mat<T, const ROWS: usize, const COLS: usize> {
    v: [[T; COLS]; ROWS],
}

pub type ColVec<T, const ROWS: usize> = Mat<T, ROWS, 1>;

pub type Vec3<T> = ColVec<T, 3>;
pub type Vec4<T> = ColVec<T, 4>;
pub type Mat3<T> = Mat<T, 3, 3>;
pub type Mat4<T> = Mat<T, 4, 4>;

impl<T, const ROWS: usize, const COLS: usize> std::fmt::Display for Mat<T, ROWS, COLS>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = f.precision().to_owned().unwrap_or(3);
        let w = f.width().to_owned().unwrap_or(6);
        write!(f, "Mat{ROWS}x{COLS}([\n")?;
        for row in 0..ROWS {
            write!(f, "  [")?;
            for col in 0..COLS {
                write!(f, "{v:w$.*}", p, w = w, v = self[(row, col)])?;
                if col != COLS - 1 {
                    f.write_str(", ")?;
                }
            }
            f.write_str("],\n")?;
        }
        f.write_str("])")?;

        Ok(())
    }
}

impl<T, const ROWS: usize> std::ops::Index<usize> for ColVec<T, ROWS> {
    type Output = T;

    fn index(&self, row: usize) -> &Self::Output {
        &self[(row, 0)]
    }
}

impl<T> Mat<T, 2, 2> {
    /// Compute the determinant
    pub fn det(&self) -> T
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Copy,
    {
        #[rustfmt::skip]
        let [
            [a, b],
            [c, d],
        ] = self.v;

        a * d - c * b
    }
}

impl<T> Mat<T, 3, 3> {
    /// Compute the determinant
    pub fn det(&self) -> T
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Copy,
    {
        #[rustfmt::skip]
        let [
            [a, b, c],
            [d, e, f],
            [g, h, i]
        ] = self.v;

        a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h
    }

    #[rustfmt::skip]
    /// Compute the adjugate Matrix
    fn adjugate(&self) -> Mat3<T>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Neg<Output = T> + Copy,
    {
        let Mat3 {
            v: [
            [a1, a2, a3],
            [b1, b2, b3],
            [c1, c2, c3],
        ],
        } = *self;

        Mat3::from_rows([
            [  b2 * c3 - c2 * b3 , -(a2 * c3 - c2 * a3),   a2 * b3 - b2 * a3 ],
            [-(b1 * c3 - c1 * b3),   a1 * c3 - c1 * a3 , -(a1 * b3 - b1 * a3)],
            [  b1 * c2 - c1 * b2 , -(a1 * c2 - c1 * a2),   a1 * b2 - b1 * a2 ],
        ])
    }

    /// try to invert self
    ///
    /// ```
    /// use yaxs::math::linalg::Mat;
    ///
    /// let m = {
    ///     let mut v = Mat::<f64, 3, 3>::zeros();
    ///     v[(0, 0)] = 12.0;
    ///     v[(1, 1)] = 6.0;
    ///     v[(2, 2)] = 3.0;
    ///     v
    /// };
    ///
    /// let inv = m.try_inverse().expect("v is invertible");
    /// assert_eq!(inv[(0, 0)], 1.0 / 12.0);
    /// assert_eq!(inv[(1, 1)], 1.0 / 6.0);
    /// assert_eq!(inv[(2, 2)], 1.0 / 3.0);
    /// ```
    pub fn try_inverse(&self) -> Option<Self>
    where
        T: Float,
    {
        let det = self.det();
        if det.is_zero() {
            return None;
        }

        Some(self.adjugate().scale(T::one() / det))
    }

    pub fn extend_to_homog(&self) -> Mat4<T>
    where
        T: Clone + Copy + Zero + One,
    {
        let mut ret = Mat4::zeros();

        for row in 0..3 {
            for col in 0..3 {
                ret[(row, col)] = self[(row, col)];
            }
        }

        ret[(3, 3)] = T::one();

        ret
    }
}

impl<T, const ROWS: usize> std::ops::IndexMut<usize> for ColVec<T, ROWS> {
    fn index_mut(&mut self, row: usize) -> &mut T {
        &mut self[(row, 0)]
    }
}

impl<T, const ROWS: usize> ColVec<T, ROWS> {
    /// Normalize the vector so that the magnitude is T::one()
    pub fn normalize(&self) -> Self
    where
        T: Float,
    {
        self.scale(T::one() / self.magnitude())
    }

    /// Normalize the vector inplace so that the magnitude is T::one()
    /// ```
    /// use yaxs::math::linalg::ColVec;
    ///
    /// let v = ColVec::from_cols([[1.0, 2.0, 3.0, 4.0]]).normalize();
    /// assert!((v.magnitude() - 1.0f64).abs() < 1e-6f64);
    /// ```
    pub fn normalize_inplace(&mut self)
    where
        T: Float + MulAssign,
    {
        self.scale_inplace(T::one() / self.magnitude());
    }

    /// Compute the dot product
    ///
    /// * `rhs`: rhs of dot product
    pub fn dot(&self, rhs: &ColVec<T, ROWS>) -> T
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Copy + Zero,
    {
        let mut res = T::zero();
        for i in 0..ROWS {
            res = res + self[i] * rhs[i];
        }

        res
    }

    /// Compute magnitude (L2-Norm) of the vector
    pub fn magnitude(&self) -> T
    where
        T: Float,
    {
        let mut sum = T::zero();
        for i in 0..ROWS {
            sum = sum + self[i] * self[i];
        }
        sum.sqrt()
    }

    pub fn zip<'a, S>(
        &'a self,
        other: &'a ColVec<S, ROWS>,
    ) -> impl Iterator<Item = (&'a T, &'a S)> {
        self.iter_values().zip(other.iter_values())
    }

    pub fn zip_mut<'a>(
        &'a mut self,
        other: &'a mut Self,
    ) -> impl Iterator<Item = (&'a mut T, &'a mut T)> {
        self.iter_values_mut().zip(other.iter_values_mut())
    }
}

impl<T, const ROWS: usize> ColVec<T, ROWS> {
    pub fn from_col(values: [T; ROWS]) -> Self {
        Mat::from_cols([values])
    }
}

impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Mat::from_rows([[x], [y], [z]])
    }

    /// extend self by putting v in the last position of self
    ///
    /// * `v`: value to extend by
    ///
    /// ```
    /// use yaxs::math::linalg::{Vec3, Vec4};
    ///
    /// let v = Vec3::new(1, 2, 3);
    ///
    /// assert_eq!(v.extend(4), Vec4::new(1, 2, 3, 4));
    /// ```
    pub fn extend(&self, v: T) -> Vec4<T>
    where
        T: Clone,
    {
        Vec4::new(self[0].clone(), self[1].clone(), self[2].clone(), v)
    }

    /// extend self by putting v in the last position of self
    ///
    /// * `v`: value to extend by
    ///
    /// ```
    /// use yaxs::math::linalg::{Vec3, Vec4};
    ///
    /// let v = Vec3::new(1, 2, 3);
    ///
    /// assert_eq!(v.extend_front(4), Vec4::new(4, 1, 2, 3));
    /// ```
    pub fn extend_front(&self, v: T) -> Vec4<T>
    where
        T: Clone,
    {
        Vec4::new(v, self[0].clone(), self[1].clone(), self[2].clone())
    }

    /// Compute the cross product with rhs
    ///
    /// * `rhs`: rhs of cross product
    pub fn cross(&self, rhs: &Self) -> Self
    where
        T: Mul<T, Output = T> + Sub<T, Output = T> + Copy,
    {
        Vec3::new(
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        )
    }
}

impl<T> Vec4<T> {
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Mat::from_cols([[x, y, z, w]])
    }
}

impl<T, const ROWS: usize, const COLS: usize> Mat<T, ROWS, COLS> {
    pub fn from_rows(v: [[T; COLS]; ROWS]) -> Self {
        Self { v }
    }

    /// Get a copy of a row
    ///
    /// * `idx`: row index
    /// ```
    /// use yaxs::math::linalg::{Mat, ColVec};
    ///
    /// let m = Mat::from_rows([
    ///     [1, 2, 3],
    ///     [3, 4, 5],
    /// ]);
    ///
    /// assert_eq!(m.row(0), ColVec::from_cols([[1, 2, 3]]));
    /// assert_eq!(m.row(1), ColVec::from_cols([[3, 4, 5]]));
    /// ```
    pub fn row(&self, idx: usize) -> ColVec<T, COLS>
    where
        T: Copy,
    {
        // SAFETY: we set the elements right after
        let mut ret: ColVec<_, COLS> = unsafe { MaybeUninit::uninit().assume_init() };

        for t in 0..COLS {
            ret[(t, 0)] = self[(idx, t)];
        }

        ret
    }

    /// Get a copy of a column
    ///
    /// * `idx`: row index
    /// ```
    /// use yaxs::math::linalg::{Mat, ColVec};
    ///
    /// let m = Mat::from_rows([
    ///     [1, 2, 3],
    ///     [3, 4, 5],
    /// ]);
    ///
    /// assert_eq!(m.row(0), ColVec::from_cols([[1, 2, 3]]));
    /// assert_eq!(m.row(1), ColVec::from_cols([[3, 4, 5]]));
    /// ```
    pub fn col(&self, idx: usize) -> ColVec<T, ROWS>
    where
        T: Copy,
    {
        // SAFETY: we set the elements right after
        let mut ret: Mat<_, ROWS, 1> = unsafe { MaybeUninit::uninit().assume_init() };

        for t in 0..COLS {
            ret[t] = self[(idx, t)];
        }

        ret
    }

    /// create self from columns
    ///
    /// * `v`: columns
    /// ```
    /// use yaxs::math::linalg::Mat;
    ///
    /// let v = Mat::from_cols([[1, 2, 3], [3, 4, 5]]);
    /// assert_eq!(v[(0, 0)], 1);
    /// assert_eq!(v[(1, 0)], 2);
    /// assert_eq!(v[(2, 0)], 3);
    ///
    /// assert_eq!(v[(0, 1)], 3);
    /// assert_eq!(v[(1, 1)], 4);
    /// assert_eq!(v[(2, 1)], 5);
    /// ```
    pub fn from_cols(v: [[T; ROWS]; COLS]) -> Self {
        let mut res: Mat<T, ROWS, COLS> = unsafe { MaybeUninit::uninit().assume_init() };
        for (ci, col) in v.into_iter().enumerate() {
            for (ri, val) in col.into_iter().enumerate() {
                res[(ri, ci)] = val;
            }
        }
        res
    }

    /// apply a function to the elements of a matrix
    ///
    /// * `p`: function to apply
    ///
    /// ```
    /// use yaxs::math::linalg::Mat;
    ///
    /// let v = Mat::from_rows([[1, 2, -3], [-4, 5, 6]]);
    /// let mapped = v.map(|&x| x > 0);
    ///
    /// assert_eq!(mapped[(0, 0)], true);
    /// assert_eq!(mapped[(0, 1)], true);
    /// assert_eq!(mapped[(0, 2)], false);
    /// assert_eq!(mapped[(1, 0)], false);
    /// assert_eq!(mapped[(1, 1)], true);
    /// assert_eq!(mapped[(1, 2)], true);
    /// ```
    pub fn map<V>(&self, mut p: impl FnMut(&T) -> V) -> Mat<V, ROWS, COLS>
    where
        T: Clone + Copy,
    {
        let mut ret: Mat<V, ROWS, COLS> = unsafe { MaybeUninit::uninit().assume_init() };

        for (rv, sv) in ret.iter_values_mut().zip(self.iter_values()) {
            *rv = p(sv);
        }

        ret
    }

    /// return whether all elements of self satisfy some condition p
    ///
    /// * `p`: condition to be evaluated on each element individually
    ///
    /// ```
    /// use yaxs::math::linalg::Mat;
    ///
    /// let v = Mat::from_rows([[8, 10, 12], [2, 4, 6]]);
    ///
    /// assert!(v.all(|&x| x % 2 == 0));
    /// ```
    pub fn all(&self, p: impl Fn(&T) -> bool) -> bool {
        self.iter_values().all(p)
    }

    /// return the number of rows
    pub fn rows(&self) -> usize {
        ROWS
    }

    /// return the number of columns
    pub fn cols(&self) -> usize {
        COLS
    }

    /// iterate values of the matix
    ///
    /// ```
    /// use yaxs::math::linalg::Mat;
    /// let v = Mat::from_cols([[1, 2, 3], [3, 4, 5]]);
    ///
    /// let mut iter = v.iter_values();
    /// assert_eq!(*iter.next().unwrap(), 1);
    /// assert_eq!(*iter.next().unwrap(), 3);
    /// assert_eq!(*iter.next().unwrap(), 2);
    /// assert_eq!(*iter.next().unwrap(), 4);
    /// assert_eq!(*iter.next().unwrap(), 3);
    /// assert_eq!(*iter.next().unwrap(), 5);
    /// ```
    pub fn iter_values(&self) -> impl Iterator<Item = &T> {
        self.v.iter().flatten()
    }

    /// iterate values of the matix
    ///
    /// ```
    /// use yaxs::math::linalg::Mat;
    /// let mut v = Mat::from_cols([[1, 2, 3], [3, 4, 5]]);
    ///
    /// let mut iter = v.iter_values_mut();
    /// assert_eq!(*iter.next().unwrap(), 1);
    /// assert_eq!(*iter.next().unwrap(), 3);
    /// let mut r = iter.next().unwrap();
    /// *r = 23;
    /// assert_eq!(*iter.next().unwrap(), 4);
    /// assert_eq!(*iter.next().unwrap(), 3);
    /// assert_eq!(*iter.next().unwrap(), 5);
    /// drop(iter);
    /// assert_eq!(v[(1, 0)], 23)
    /// ```
    pub fn iter_values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.v.iter_mut().flatten()
    }

    /// Scale self in place
    ///
    /// ```
    /// use yaxs::math::linalg::Mat;
    /// let mut v = Mat::from_cols([[1, 2, 3], [3, 4, 5]]);
    /// v.scale_inplace(2);
    /// assert_eq!(v[(0, 0)], 2);
    /// assert_eq!(v[(1, 0)], 4);
    /// assert_eq!(v[(2, 0)], 6);
    /// assert_eq!(v[(0, 1)], 6);
    /// assert_eq!(v[(1, 1)], 8);
    /// assert_eq!(v[(2, 1)], 10);
    /// ```
    pub fn scale_inplace(&mut self, s: T)
    where
        T: MulAssign + Copy,
    {
        for row in 0..ROWS {
            for col in 0..COLS {
                self[(row, col)] *= s;
            }
        }
    }

    /// determine whether self and other's elements all are within atol of each other
    ///
    /// * `other`: right hand side of fuzzy equality
    /// * `atol`: maximum allowed (absolute) difference
    pub fn isclose(&self, other: &Mat<T, ROWS, COLS>, atol: T) -> Option<(T, T)>
    where
        T: Sub<T, Output = T> + Zero + Neg<Output = T> + PartialOrd + Copy,
    {
        for (a, b) in self.iter_values().zip(other.iter_values()) {
            let mut diff = *a - *b;
            if diff < T::zero() {
                diff = -diff;
            }

            if diff > atol {
                return Some((*a, *b));
            }
        }

        None
    }

    /// Transpose self
    /// ```
    /// use yaxs::math::linalg::Mat;
    ///
    /// let v = Mat::from_cols([[1, 2, 3], [3, 4, 5]]);
    /// let t = v.transpose();
    /// for row in 0..t.rows() {
    ///     for col in 0..t.cols() {
    ///         assert_eq!(t[(row, col)], v[(col, row)]);
    ///     }
    /// }
    /// ```
    pub fn transpose(&self) -> Mat<T, COLS, ROWS>
    where
        T: Copy,
    {
        // SAFETY: OK because we set every element later
        let mut ret: Mat<_, COLS, ROWS> = unsafe { MaybeUninit::uninit().assume_init() };
        for row in 0..ROWS {
            for col in 0..COLS {
                ret[(col, row)] = self[(row, col)];
            }
        }
        ret
    }

    /// Create Zeroed Matrix
    pub fn zeros() -> Mat<T, ROWS, COLS>
    where
        T: Zero + Copy,
    {
        Mat {
            v: [[T::zero(); COLS]; ROWS],
        }
    }

    /// Perform Matrix Multiplication with diagonal matrix
    ///
    /// * `other`: diagonal matrix entries represented as a column vector
    pub fn matmul_diag(&self, other: &ColVec<T, ROWS>) -> Self
    where
        T: MulAssign<T> + Clone + Copy,
    {
        // a b c d     x 0 0 0
        // e f g h  *  0 y 0 0
        // i j k l     0 0 z 0
        // m n o p     0 0 0 w
        let mut ret = self.clone();
        for row in 0..ROWS {
            for col in 0..COLS {
                ret[(row, col)] *= other[row]
            }
        }
        ret
    }

    /// Perform matrix multiplication
    ///
    /// * `other`:
    pub fn matmul<const RHS_COLS: usize>(
        &self,
        other: &Mat<T, COLS, RHS_COLS>,
    ) -> Mat<T, ROWS, RHS_COLS>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Copy + Zero + AddAssign<T>,
    {
        let mut ret = Mat::<T, ROWS, RHS_COLS>::zeros();
        for row in 0..ROWS {
            for col in 0..RHS_COLS {
                for t in 0..COLS {
                    ret[(row, col)] += self[(row, t)] * other[(t, col)];
                }
            }
        }
        ret
    }

    /// Scale self
    ///
    /// * `s`: scale by value
    pub fn scale(&self, s: T) -> Self
    where
        T: Mul<T, Output = T> + Clone + Copy,
    {
        let mut ret = self.clone();
        for r in ret.iter_values_mut() {
            *r = *r * s;
        }
        ret
    }
}

impl<T> Mat<T, 4, 4> {
    /// Treating self as a homogenous matrix, set the translation component
    ///
    /// * `x_new`: new x translation
    /// * `y_new`: new y translation
    /// * `z_new`: new z translation
    pub fn set_homog_translation(&mut self, x_new: T, y_new: T, z_new: T) {
        #[rustfmt::skip]
        let [
            [_, _, _, x],
            [_, _, _, y],
            [_, _, _, z],
            [_, _, _, _]
        ] = &mut self.v;

        *x = x_new;
        *y = y_new;
        *z = z_new;
    }

    /// Treating self as a homogenous matrix, transform a vector
    ///
    /// * `rhs`: vector to transform
    pub fn homog_mul(&self, rhs: &Vec3<T>) -> Vec3<T>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Copy,
    {
        #[rustfmt::skip]
        let [
            [a, b, c, x],
            [d, e, f, y],
            [g, h, i, z],
            [_, _, _, _]
        ] = self.v;
        Vec3::new(
            a * rhs[0] + b * rhs[1] + c * rhs[2] + x,
            d * rhs[0] + e * rhs[1] + f * rhs[2] + y,
            g * rhs[0] + h * rhs[1] + i * rhs[2] + z,
        )
    }

    /// Treating self as a homogenous matrix, transform a Mat3
    ///
    /// * `rhs`: vector to transform
    pub fn homog_mul_mat(&self, rhs: &Mat3<T>) -> Mat3<T>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Zero + One + Copy + AddAssign<T>,
    {
        let mut m = Mat4::zeros();
        for r in 0..rhs.rows() {
            for c in 0..rhs.cols() {
                m[(r, c)] = rhs[(r, c)];
            }
        }
        m[(3, 3)] = T::one();

        let r_ = self.matmul(&m);

        let mut ret = Mat3::zeros();

        for r in 0..ret.rows() {
            for c in 0..ret.cols() {
                ret[(r, c)] = r_[(r, c)];
            }
        }

        ret
    }
}

impl<T, const N: usize> Mat<T, N, N> {
    #[rustfmt::skip]
    pub fn identity() -> Mat<T, N, N>
    where
        T: Zero + One + Copy,
    {
        let mut ret = Mat::<T, N, N>::zeros();

        for t in 0..N {
            ret[(t, t)] = T::one();
        }

        ret
    }

    /// Compute colesky decomposition using the cholesky-banachiewicz algorithm
    pub fn cholesky_decompose(&self) -> Result<Mat<T, N, N>, String>
    where
        T: Float + AddAssign<T>,
    {
        let mut ret = Mat::zeros();
        for i in 0..N {
            for j in 0..=i {
                let mut sum: T = T::zero();
                for k in 0..j {
                    sum += ret[(i, k)] * ret[(j, k)];
                }

                if i == j {
                    let radicand = self[(i, j)] - sum;
                    if radicand < T::zero() {
                        return Err(format!("Could not compute cholesky decomposition. Matrix is not SPD, led to square root of negative number when computing position ({i}, {j}) in lower triangular matrix."));
                    }
                    ret[(i, j)] = radicand.sqrt();
                } else {
                    ret[(i, j)] = T::one() / ret[(j, j)] * (self[(i, j)] - sum);
                }
            }
        }
        Ok(ret)
    }
}

impl<T, const ROWS: usize, const COLS: usize> std::ops::AddAssign for Mat<T, ROWS, COLS>
where
    T: AddAssign<T> + Clone + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        for (v, r) in self.iter_values_mut().zip(rhs.iter_values()) {
            *v += *r;
        }
    }
}

impl<T, const ROWS: usize, const COLS: usize> std::ops::Add for Mat<T, ROWS, COLS>
where
    T: Add<T, Output = T> + Clone + Copy,
{
    type Output = Mat<T, ROWS, COLS>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = rhs.clone();

        for (r, v) in ret.iter_values_mut().zip(self.iter_values()) {
            *r = *r + *v;
        }

        ret
    }
}

impl<T, const ROWS: usize, const COLS: usize> std::ops::Add for &Mat<T, ROWS, COLS>
where
    T: Add<T, Output = T> + Clone + Copy,
{
    type Output = Mat<T, ROWS, COLS>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = rhs.clone();

        for (r, v) in ret.iter_values_mut().zip(self.iter_values()) {
            *r = *r + *v;
        }

        ret
    }
}

impl<T, const ROWS: usize, const COLS: usize> std::ops::Mul for Mat<T, ROWS, COLS>
where
    T: Mul<T, Output = T> + Clone + Copy,
{
    type Output = Mat<T, ROWS, COLS>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = rhs.clone();

        for (r, v) in ret.iter_values_mut().zip(self.iter_values()) {
            *r = *r * *v;
        }

        ret
    }
}

impl<T, const ROWS: usize, const COLS: usize> std::ops::Mul<T> for Mat<T, ROWS, COLS>
where
    T: Mul<T, Output = T> + Clone + Copy,
{
    type Output = Mat<T, ROWS, COLS>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut ret = self.clone();

        for r in ret.iter_values_mut() {
            *r = *r * rhs
        }

        ret
    }
}

impl<T, const ROWS: usize, const COLS: usize> Div for Mat<T, ROWS, COLS>
where
    T: Div<T, Output = T> + Clone + Copy,
{
    type Output = Mat<T, ROWS, COLS>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();

        for (r, v) in ret.iter_values_mut().zip(rhs.iter_values()) {
            *r = *r / *v
        }

        ret
    }
}

impl<T, const ROWS: usize, const COLS: usize> Div for &Mat<T, ROWS, COLS>
where
    T: Div<T, Output = T> + Clone + Copy,
{
    type Output = Mat<T, ROWS, COLS>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();

        for (r, v) in ret.iter_values_mut().zip(rhs.iter_values()) {
            *r = *r / *v
        }

        ret
    }
}

impl<T, const ROWS: usize, const COLS: usize> Div<T> for Mat<T, ROWS, COLS>
where
    T: Div<T, Output = T> + Clone + Copy,
{
    type Output = Mat<T, ROWS, COLS>;

    fn div(self, rhs: T) -> Self::Output {
        let mut ret = self.clone();

        for r in ret.iter_values_mut() {
            *r = *r / rhs
        }

        ret
    }
}

impl<T, const ROWS: usize, const COLS: usize> std::ops::Index<(usize, usize)>
    for Mat<T, ROWS, COLS>
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < ROWS);
        assert!(col < COLS);

        &self.v[row][col]
    }
}

impl<T> Mat<T, 1, 1> {
    /// reduce matrix to singular item
    pub fn item(self) -> T
    where
        T: Copy,
    {
        self[(0, 0)]
    }
}

impl<T, const ROWS: usize, const COLS: usize> std::ops::IndexMut<(usize, usize)>
    for Mat<T, ROWS, COLS>
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(
            row < ROWS,
            "row index must be smaller than number of rows ({row} < {ROWS})"
        );
        assert!(
            col < COLS,
            "column index must be smaller than number of columns ({col} < {COLS})"
        );

        &mut self.v[row][col]
    }
}

impl<T, V, const N: usize, const M: usize> Neg for Mat<T, M, N>
where
    T: Neg<Output = V> + Copy,
{
    type Output = Mat<V, M, N>;

    fn neg(self) -> Self::Output {
        self.map(|x| -*x)
    }
}

impl<T, V, const N: usize, const M: usize> Sub<Self> for Mat<T, M, N>
where
    for<'a> &'a T: Sub<Output = V>,
{
    type Output = Mat<V, M, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        // SAFETY: this is ok because we set every value
        let mut ret: Mat<_, M, N> = unsafe { MaybeUninit::uninit().assume_init() };

        for ((retv, sv), rhsv) in ret
            .iter_values_mut()
            .zip(self.iter_values())
            .zip(rhs.iter_values())
        {
            *retv = sv - rhsv;
        }

        ret
    }
}

impl<T, V, const N: usize, const M: usize> Sub<Self> for &Mat<T, M, N>
where
    for<'a> &'a T: Sub<Output = V>,
{
    type Output = Mat<V, M, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        // SAFETY: this is ok because we set every value
        let mut ret: Mat<_, M, N> = unsafe { MaybeUninit::uninit().assume_init() };

        for ((retv, sv), rhsv) in ret
            .iter_values_mut()
            .zip(self.iter_values())
            .zip(rhs.iter_values())
        {
            *retv = sv - rhsv;
        }

        ret
    }
}

impl<T, const ROWS: usize, const COLS: usize> Serialize for Mat<T, ROWS, COLS>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(ROWS * COLS))?;
        for v in self.iter_values() {
            seq.serialize_element(v)?;
        }
        seq.end()
    }
}

impl<'de, T, const ROWS: usize, const COLS: usize> Deserialize<'de> for Mat<T, ROWS, COLS>
where
    for<'e> T: Deserialize<'e>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MatVisitor<T, const ROWS: usize, const COLS: usize> {
            _phantom: PhantomData<T>,
        }
        impl<T, const ROWS: usize, const COLS: usize> MatVisitor<T, ROWS, COLS> {
            fn new() -> Self {
                MatVisitor {
                    _phantom: PhantomData,
                }
            }
        }

        impl<'de, T, const ROWS: usize, const COLS: usize> Visitor<'de> for MatVisitor<T, ROWS, COLS>
        where
            for<'a> T: Deserialize<'a>,
        {
            type Value = Mat<T, ROWS, COLS>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "struct Mat<_, {}, {}> values", ROWS, COLS)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                // SAFETY: this is ok because we immediately set all values after or error out
                let mut ret: Mat<T, ROWS, COLS> = unsafe { MaybeUninit::uninit().assume_init() };
                for row in 0..ROWS {
                    for col in 0..COLS {
                        ret[(row, col)] = seq
                            .next_element()?
                            .ok_or_else(|| de::Error::invalid_length(row * COLS + col, &self))?;
                    }
                }
                Ok(ret)
            }
        }

        Ok(deserializer.deserialize_seq(MatVisitor::new())?)
    }
}

#[cfg(test)]
mod test {
    use rand::distr::uniform::{SampleRange, SampleUniform};
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn random_mat<T, const ROWS: usize, const COLS: usize>(
        vrange: impl SampleRange<T> + Clone,
        rng: &mut impl Rng,
    ) -> Mat<T, ROWS, COLS>
    where
        T: Copy + Zero + SampleUniform,
    {
        // SAFETY: This is ok because right below, we set all values
        let mut ret: Mat<_, ROWS, COLS> = unsafe { MaybeUninit::uninit().assume_init() };

        for v in ret.iter_values_mut() {
            *v = rng.random_range(vrange.clone());
        }
        ret
    }

    use super::*;
    #[test]
    fn indexing() {
        #[rustfmt::skip]
        let m = Mat::from_rows([
            [1,  2,  3,  4],
            [5,  6,  7,  8],
        ]);

        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(0, 1)], 2);
        assert_eq!(m[(0, 2)], 3);
        assert_eq!(m[(0, 3)], 4);

        assert_eq!(m[(1, 0)], 5);
        assert_eq!(m[(1, 1)], 6);
        assert_eq!(m[(1, 2)], 7);
        assert_eq!(m[(1, 3)], 8);
    }

    #[test]
    fn transpose_random() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);

        let mat = random_mat::<_, 10, 3>(0.0..=1.0, &mut rng);
        let tp = mat.transpose();
        for row in 0..mat.rows() {
            for col in 0..mat.cols() {
                println!("{}, {}", row, col);
                assert_eq!(mat[(row, col)], tp[(col, row)])
            }
        }
    }

    #[test]
    fn cholesky_decomp() {
        #[rustfmt::skip]
        let l = Mat::from_rows([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [1.0, 3.0, 5.0, 0.0],
            [1.0, 3.0, 8.0, 8.0],
        ]);

        let a = l.matmul(&l.transpose());

        let maybe_l = a.cholesky_decompose().unwrap();
        for row in 0..4 {
            for col in 0..4 {
                print!("{} ", a[(row, col)]);
            }
            print!("\n");
        }

        assert_eq!(maybe_l, l);
    }

    #[test]
    fn inverse_identity3() {
        let m = Mat3::<f64>::identity();
        let inv = m.try_inverse().expect("identity has inverse");
        assert_eq!(inv, m)
    }

    #[test]
    fn cholesky_decomp_random_1000() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
        const N: usize = 5;

        for _ in 0..1000 {
            let mut l = random_mat::<_, N, N>(0.0..10.0, &mut rng);
            for row in 0..N {
                for col in row + 1..N {
                    l[(row, col)] = 0.0;
                }
            }
            let a = l.matmul(&l.transpose());

            let maybe_l = a.cholesky_decompose().unwrap();
            if let Some((a, b)) = maybe_l.isclose(&l, 1e-6) {
                panic!("Expected matrix to be close. Exceeded tolerance for items {a} and {b}");
            }
        }
    }

    #[test]
    fn adjugate() {
        #[rustfmt::skip]
        let m = Mat3::from_rows([
            [-3.0,  2.0, -5.0],
            [-1.0,  0.0, -2.0],
            [ 3.0, -4.0,  1.0]
        ]);
        let expected = Mat3::from_rows([[-8.0, 18.0, -4.0], [-5.0, 12.0, -1.0], [4.0, -6.0, 2.0]]);
        assert_eq!(m.adjugate(), expected)
    }

    #[test]
    fn double_inverse() {
        #[rustfmt::skip]
        let m = Mat3::<f64>::from_rows([
            [-3.0,  2.0, -5.0],
            [-1.0,  0.0, -2.0],
            [ 3.0, -4.0,  1.0]
        ]);
        let inv = m.try_inverse().expect("matrix has inverse");
        let hopefully_m = inv.try_inverse().expect("matrix has inverse");
        if let Some((a, b)) = m.isclose(&hopefully_m, 1e-12f64) {
            panic!("Isclose failed. values {a} and {b} exceeded tolerance");
        }
    }

    #[test]
    fn vec_mat_mul_ident() {
        let v = Vec3::new(1, 2, 3);
        let res = Mat3::identity().matmul(&v);
        assert_eq!(v, res);
    }

    #[test]
    fn vec_mat_mul() {
        let v = Vec3::new(1, 2, 3);
        #[rustfmt::skip]
        let res = Mat3::from_rows([
            [1, 2, 1],
            [2, 3, 1],
            [4, 2, 2],
        ]).matmul(&v);

        let expected = Vec3::new(8, 11, 14);
        assert_eq!(expected, res);
    }

    #[test]
    fn mat_mat_mul() {
        #[rustfmt::skip]
        let m1 = Mat3::from_rows([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);
        #[rustfmt::skip]
        let m2 = Mat3::from_rows([
            [5, 4, 6],
            [8, 7, 9],
            [2, 1, 3],
        ]);

        #[rustfmt::skip]
        let expected = Mat3::from_rows([
            [ 27, 21,  33],
            [ 72, 57,  87],
            [117, 93, 141],
        ]);
        let res = m1.matmul(&m2);
        assert_eq!(res, expected);
    }

    #[test]
    fn transpose() {
        #[rustfmt::skip]
        let mat = Mat3::from_rows([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);

        #[rustfmt::skip]
        let expected = Mat3::from_rows([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        ]);

        let t = mat.transpose();
        assert_eq!(t, expected);
    }

    #[test]
    fn from_cols() {
        let v = Mat::from_cols([[1, 2, 3], [3, 4, 5]]);
        assert_eq!(v[(0, 0)], 1);
        assert_eq!(v[(1, 0)], 2);
        assert_eq!(v[(2, 0)], 3);

        assert_eq!(v[(0, 1)], 3);
        assert_eq!(v[(1, 1)], 4);
        assert_eq!(v[(2, 1)], 5);
    }

    #[test]
    fn dot() {
        let v = Vec3::new(1, 2, 3);
        let dot = v.dot(&v);
        assert_eq!(dot, 1 + 4 + 9);
    }

    #[test]
    fn cross_orthogonal() {
        let v0 = Vec3::new(1, 0, 0);
        let v1 = Vec3::new(0, 1, 0);
        let exp = Vec3::new(0, 0, 1);

        assert_eq!(v0.cross(&v1), exp);
    }

    #[test]
    fn cross_orthogonal_result() {
        let v0 = Vec3::new(1.0, 2.0, 3.0);
        let v1 = Vec3::new(1.0, 2.2, 2.3);
        let c = v0.cross(&v1);

        assert!(c.dot(&v0).abs() < 1e-15);
        assert!(c.dot(&v1).abs() < 1e-15);
    }
}
