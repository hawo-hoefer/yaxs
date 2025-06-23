use std::f32::consts::{PI, TAU};

/// plack's constant in ev * s = ev * hz^-1
pub const H_EV_S: f64 = 4.135_667_696e-15f64;

/// speed of light in m / s
pub const C_M_S: f64 = 299_792_485.0f64;

/// electron mass in kg
pub const ELECTRON_MASS_KG: f64 = 9.10938188e-31;

/// electron mass in kg
pub const EV_TO_JOULE: f64 = 1.60217646e-19;

/// vacuum permeability in N A^-2
pub const MU_0_N_A2: f64 = 1.25663706127e-6;

/// vacuum permittivity in F m^-1
pub const EPS_0_F_M1: f64 = 8.8541788188e-12;

pub fn e_kev_to_lambda_ams(e_kev: f64) -> f64 {
    // e = h * c / lambda
    // lambda = h * c / e
    // m      = ev * s * m / ev
    H_EV_S * C_M_S / e_kev * 1e7
}

/// Calculate Caglioti broadening for a position
/// $FWHM(\theta) = u \tan(\theta)^2 + v \tan(\theta) + w$
///
/// * `u`: parameter u
/// * `v`: parameter v
/// * `w`: parameter w
/// * `theta_rad`: theta in radians
pub fn caglioti(u: f64, v: f64, w: f64, theta: f64) -> f64 {
    u * (theta).tan().powi(2) + v * theta.tan() + w
}

// Scherrer broadening constant
const K: f64 = 0.9;

/// calculate scherrer broadening in angle dispersive XRD
///
/// * `wavelength`: wavelength in nanometers
/// * `theta_rad`: theta in radians
/// * `mean_ds`: mean domain size in nanometers
pub fn scherrer_broadening(wavelength: f64, theta: f64, mean_ds: f64) -> f64 {
    // scherrer
    // tau = k * lambda / (fwhm * cos(theta))
    // fwhm = k * lambda / (tau * cos(theta))
    (K * wavelength / (theta.cos() * mean_ds)).to_degrees()
}

/// calculate scherrer broadening in energy dispersive XRD from
/// Ellmer, K., et al. Measurement Science and Technology 14.3 (2003): 336
/// DOI: <https://doi.org/10.1088/0957-0233/14/3/313>
///
/// * `wavelength`: wavelength in nanometers
/// * `theta_rad`: theta in radians
/// * `mean_ds`: mean domain size in nanometers
pub fn scherrer_broadening_edxrd(d_hkl: f64, e_kev: f64, mean_ds: f64) -> f64 {
    // from the paper above:
    // mean_ds = K d_hkl E / fwhm
    // fwhm = K d_Hkl E / mean_ds
    K * d_hkl * e_kev / mean_ds
}

/// compute the lorentz polarization factor
///
/// * `theta_rad`: position to compute correction for
pub fn lorentz_factor(theta_rad: f64) -> f64 {
    (1.0 + theta_rad.cos().powi(2)) / ((theta_rad / 2.0).sin() * theta_rad.sin())
}

pub fn gauss(dx: f32, sigma: f32) -> f32 {
    (-0.5 * (dx / sigma).powi(2)).exp() / (TAU * sigma.powi(2)).sqrt()
}

pub fn lorentz(dx: f32, gamma: f32) -> f32 {
    1.0 / ((1.0 + (dx / gamma).powi(2)) * PI * gamma)
}

pub fn pseudo_voigt(dx: f32, eta: f32, fwhm: f32) -> f32 {
    // sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * fwhm
    // fwhm = 2 * sqrt(2 ln(2)) sigma
    // sigma = fwhm / (2 sqrt(2 ln 2))
    let two_sqrt_ln_2 = 2.0 * (2.0f32.ln() * 2.0).sqrt();
    let sigma = (1.0 / two_sqrt_ln_2) * fwhm;
    let gamma = fwhm / 2.0;
    eta * lorentz(dx, gamma) + (1.0 - eta) * gauss(dx, sigma)
}

pub fn sample_displacement_delta_two_theta_rad(
    displacement_mu_m: f64,
    goniometer_radius_mm: f64,
    theta_rad: f64,
) -> f64 {
    -2.0 * displacement_mu_m / (goniometer_radius_mm * 1e3) * theta_rad.cos()
}

pub mod acm757 {
    /// Evaluate a chebyshev series using the Clenshaw method with Reinsch modification,
    /// as analyzed in the paper by Oliver. This function is adapted from
    /// Algorithm 757: MISCFUN, a software package to compute uncommon special functions.
    /// DOI: <https://doi.org/10.1145/232826.232846>
    ///
    /// Original Author:
    ///     Dr. Allan J. MacLeod,
    ///     Dept. of Mathematics and Statistics,
    ///     University of Paisley,
    ///     High St.,
    ///     PAISLEY,
    ///     SCOTLAND
    ///
    /// Reference:
    ///     "An error analysis of the modified Clenshaw method for evaluating
    ///      Chebyshev and Fourier series" J. Oliver,
    ///      J.I.M.A., vol. 20, 1977, pp379-391
    ///
    /// * `coef`: chebyshev coefficients
    /// * `t`: value to evaluate function at
    pub fn cheval(coef: &[f64], t: f64) -> f64 {
        let mut u1 = 0.0;
        let mut u2 = 0.0;

        if t.abs() < 0.6 {
            // If ABS ( T )  < 0.6 use the standard Clenshaw method
            let mut u0 = 0.0; // U0 = ZERO
            let tt = 2.0 * t; // TT = T + T
                              // DO 100 I = N , 0 , -1
            for a_i in coef.iter().rev() {
                u2 = u1; // U2 = U1
                u1 = u0; // U1 = U0
                         // U0 = TT * U1 + A( I ) - U2
                u0 = tt * u1 + a_i - u2;
            }
            // CHEVAL =  ( U0 - U2 ) / TWO
            return (u0 - u2) / 2.0;
        }

        // If ABS ( T )  > =  0.6 use the Reinsch modification
        let mut d1 = 0.0;
        let mut d2 = 0.0;
        // IF ( T .GT. ZERO ) THEN
        if t > 0.0 {
            // TT =  ( T - HALF ) - HALF
            // TT = TT + TT
            let tt = 2.0 * (t - 1.0);
            // DO 200 I = N , 0 , -1
            for a_i in coef.iter().rev() {
                d2 = d1; // D2 = D1
                u2 = u1; // U2 = U1
                         // D1 = TT * U2 + A( I ) + D2
                d1 = tt * u2 + a_i + d2;
                // U1 = D1 + U2
                u1 = d1 + u2;
            }
            // CHEVAL =  ( D1 + D2 ) / TWO
            return (d1 + d2) / 2.0;
        }

        // t <= -0.6
        // TT =  ( T + HALF ) + HALF
        // TT = TT + TT
        let tt = 2.0 * (t + 1.0);
        // DO 300 I = N , 0 , -1
        for a_i in coef.iter().rev() {
            d2 = d1; // D2 = D1
            u2 = u1; // U2 = U1
                     // D1 = TT * U2 + A( I ) - D2
            d1 = tt * u2 + a_i - d2;
            // U1 = D1 - U2
            u1 = d1 - u2;
        }

        // CHEVAL =  ( D1 - D2 ) / TWO
        return (d1 - d2) / 2.0;
    }

    /// Calculate the first synchrotron radiation function defined as
    ///    SYNCH1(x) = x * Integral{x to inf} K(5/3)(t) dt,
    /// where K(5/3) is a modified Bessel function of order 5/3.
    /// The code uses Chebyshev expansions, the coefficients of which
    /// are given to 20 decimal places.
    ///
    /// The function is undefined for x < 0, and returns None in that case.
    ///
    /// This function is adapted from
    /// Algorithm 757: MISCFUN, a software package to compute uncommon special functions.
    /// DOI: <https://doi.org/10.1145/232826.232846>
    ///
    /// Original Author:
    ///     Dr. Allan J. MacLeod,
    ///     Dept. of Mathematics and Statistics,
    ///     University of Paisley,
    ///     Paisley,
    ///     SCOTLAND
    ///     PA1 2BE
    ///
    /// * `x`: function input
    pub fn synch_1(x: f64) -> Option<f64> {
        // chebyshev coefficients
        const ASYNC1: [f64; 14] = [
            30.36468_29825_01076_27340e0,
            17.07939_52774_08394_57449e0,
            4.56013_21335_45072_88887e0,
            0.54928_12467_30419_97963e0,
            0.37297_60750_69301_1724e-1,
            0.16136_24302_01041_242e-2,
            0.48191_67721_20370_7e-4,
            0.10512_42528_89384e-5,
            0.17463_85046_697e-7,
            0.22815_48654_4e-9,
            0.24044_3082e-11,
            0.20865_88e-13,
            0.15167e-15,
            0.94e-18,
        ];
        const ASYNC2: [f64; 12] = [
            0.44907_21623_53266_08443e0,
            0.89835_36779_94187_2179e-1,
            0.81044_57377_21512_894e-2,
            0.42617_16991_08916_19e-3,
            0.14760_96312_70746_0e-4,
            0.36286_33615_3998e-6,
            0.66634_80749_84e-8,
            0.94907_71655e-10,
            0.10791_2491e-11,
            0.10022_01e-13,
            0.7745e-16,
            0.51e-18,
        ];
        const ASYNCA: [f64; 25] = [
            2.13293_05161_35500_09848e0,
            0.74135_28649_54200_2401e-1,
            0.86968_09990_99641_978e-2,
            0.11703_82624_87756_921e-2,
            0.16451_05798_61919_15e-3,
            0.24020_10214_20640_3e-4,
            0.35827_75638_93885e-5,
            0.54477_47626_9837e-6,
            0.83880_28561_957e-7,
            0.13069_88268_416e-7,
            0.20530_99071_44e-8,
            0.32518_75368_8e-9,
            0.51791_40412e-10,
            0.83002_9881e-11,
            0.13352_7277e-11,
            0.21591_498e-12,
            0.34996_73e-13,
            0.56994_2e-14,
            0.92906e-15,
            0.15222e-15,
            0.2491e-16,
            0.411e-17,
            0.67e-18,
            0.11e-18,
            0.2e-19,
        ];

        // f64-specific constants
        //
        // let EPSNEG: f64 = 2.0f64.powi(-53);
        // let XLOW: f64 = (8.0 * EPSNEG);
        const XLOW: f64 = 2.9802322387695313e-8;
        // let XHIGH1 = -8.0 * std::f64::MIN_POSITIVE.ln() / 7.0;
        const XHIGH1: f64 = 8.095959068940161e2;
        // let XHIGH2 = std::f64::MIN_POSITIVE.ln();
        const XHIGH2: f64 = -7.083964185322641e2;

        // numeric constants
        const CONLOW: f64 = 2.14952_82415_34478_63671;
        const LNRTP2: f64 = 0.22579_13526_44727_43236;
        const PIBRT3: f64 = 1.81379_93642_34217_85059;

        if x < 0.0 {
            // undefined for x < 0
            return None;
        }

        // IF ( X .LE. FOUR ) THEN
        if x <= 4.0 {
            // Code for 0 <= x <= 4
            // XPOWTH = X ** ( ONE / THREE )
            let xpowth = x.powf(1.0 / 3.0);
            // IF ( X .LT. XLOW ) THEN
            if x < XLOW {
                // SYNCH1 = CONLOW * XPOWTH
                return Some(CONLOW * xpowth);
            }

            // T = ( X * X / EIGHT - HALF ) - HALF
            let mut t = x.powi(2) / 8.0 - 1.0;
            // CHEB1 = CHEVAL(NTERM1,ASYNC1,T)
            // CHEB2 = CHEVAL(NTERM2,ASYNC2,T)
            let cheb1 = cheval(&ASYNC1, t);
            let cheb2 = cheval(&ASYNC2, t);
            // T = XPOWTH * CHEB1 - ( XPOWTH**11 ) * CHEB2
            t = xpowth * cheb1 - xpowth.powi(11) * cheb2;
            // SYNCH1 = T - PIBRT3 * X
            return Some(t - PIBRT3 * x);
        }

        // IF ( X .GT. XHIGH1 ) THEN
        if x > XHIGH1 {
            // SYNCH1 = ZERO
            return Some(0.0);
        }

        // T = ( TWELVE - X ) / ( X + FOUR )
        let mut t = (12.0 - x) / (x + 4.0);
        // CHEB1 = CHEVAL(NTERM3,ASYNCA,T)
        let cheb1 = cheval(&ASYNCA, t);
        // T = LNRTP2 - X + LOG( SQRT(X) * CHEB1 )
        t = LNRTP2 - x + (x.sqrt() * cheb1).ln();

        // IF ( T .LT. XHIGH2 ) THEN
        if t < XHIGH2 {
            // SYNCH1 = ZERO
            return Some(0.0);
        }

        // SYNCH1 = EXP(T)
        return Some(t.exp());
    }

    /// Compute the second synchrotron radiation function defined as
    ///          SYNCH2(x) = x * K(2/3)(x)
    /// where K(2/3) is a modified Bessel function of order 2/3.
    /// The code uses Chebyshev expansions, the coefficients of which
    /// are given to 20 decimal places.
    ///
    /// The function is undefined for x < 0, and returns None in that case.
    ///
    /// This function is adapted from
    /// Algorithm 757: MISCFUN, a software package to compute uncommon special functions.
    /// DOI: <https://doi.org/10.1145/232826.232846>
    ///
    /// Original Author:
    ///     Dr. Allan J. MacLeod,
    ///     Dept. of Mathematics and Statistics,
    ///     University of Paisley,
    ///     Paisley,
    ///     SCOTLAND
    ///     PA1 2BE
    ///
    /// * `x`: synchrotron function argument
    pub fn synch_2(x: f64) -> Option<f64> {
        // chebyshev coefficients
        const ASYN21: [f64; 15] = [
            38.61783_99238_43085_48014e0,
            23.03771_55949_63734_59697e0,
            5.38024_99868_33570_59676e0,
            0.61567_93806_99571_07760e0,
            0.40668_80046_68895_5843e-1,
            0.17296_27455_26484_141e-2,
            0.51061_25883_65769_9e-4,
            0.11045_95950_22012e-5,
            0.18235_53020_649e-7,
            0.23707_69803_4e-9,
            0.24887_2963e-11,
            0.21528_68e-13,
            0.15607e-15,
            0.96e-18,
            0.1e-19,
        ];
        const ASYN22: [f64; 14] = [
            7.90631_48270_66080_42875e0,
            3.13534_63612_85342_56841e0,
            0.48548_79477_45371_45380e0,
            0.39481_66758_27237_2337e-1,
            0.19661_62233_48088_022e-2,
            0.65907_89322_93042_0e-4,
            0.15857_56134_98559e-5,
            0.28686_53011_233e-7,
            0.40412_02359_5e-9,
            0.45568_4443e-11,
            0.42045_90e-13,
            0.32326e-15,
            0.210e-17,
            0.1e-19,
        ];
        const ASYN2A: [f64; 19] = [
            2.02033_70941_70713_60032e0,
            0.10956_23712_18074_0443e-1,
            0.85423_84730_11467_55e-3,
            0.72343_02421_32822_2e-4,
            0.63124_42796_26992e-5,
            0.56481_93141_1744e-6,
            0.51283_24801_375e-7,
            0.47196_53291_45e-8,
            0.43807_44214_3e-9,
            0.41026_81493e-10,
            0.38623_0721e-11,
            0.36613_228e-12,
            0.34802_32e-13,
            0.33301_0e-14,
            0.31856e-15,
            0.3074e-16,
            0.295e-17,
            0.29e-18,
            0.3e-19,
        ];

        // f64-specific constants
        //
        // let EPSNEG: f64 = 2.0f64.powi(-53);
        // let XLOW: f64 = (8.0 * EPSNEG);
        const XLOW: f64 = 2.9802322387695313e-8;
        // let XHIGH1 = -8.0 * std::f64::MIN_POSITIVE.ln() / 7.0;
        const XHIGH1: f64 = 8.095959068940161e2;
        // let XHIGH2 = std::f64::MIN_POSITIVE.ln();
        const XHIGH2: f64 = -7.083964185322641e2;

        // numeric constants
        const CONLOW: f64 = 1.07476_41207_67239_31836;
        const LNRTP2: f64 = 0.22579_13526_44727_43236;

        // IF ( X .LE. FOUR ) THEN
        if x <= 4.0 {
            // Code for 0 <= x <= 4
            // XPOWTH = X ** ( ONE / THREE )
            let xpowth = x.powf(1.0 / 3.0);

            // IF ( X .LT. XLOW ) THEN
            if x < XLOW {
                // SYNCH2 = CONLOW * XPOWTH
                return Some(CONLOW * xpowth);
            } else {
                // T = ( X * X / EIGHT - HALF ) - HALF
                let t = x.powi(2) / 8.0 - 1.0;
                // CHEB1 = CHEVAL(NTERM1,ASYN21,T)
                // CHEB2 = CHEVAL(NTERM2,ASYN22,T)
                let cheb1 = cheval(&ASYN21, t);
                let cheb2 = cheval(&ASYN22, t);

                // SYNCH2 = XPOWTH * CHEB1 - ( XPOWTH**5 ) * CHEB2
                return Some(xpowth * cheb1 - xpowth.powi(5) * cheb2);
            }
        } else {
            // IF ( X .GT. XHIGH1 ) THEN
            if x > XHIGH1 {
                // SYNCH2 = ZERO
                return Some(0.0);
            } else {
                // T = ( TEN - X ) / ( X + TWO )
                let mut t = (10.0 - x) / (x + 2.0);
                // CHEB1 = CHEVAL(NTERM3,ASYN2A,T)
                let cheb1 = cheval(&ASYN2A, t);
                // T = LNRTP2 - X + LOG( SQRT(X) * CHEB1 )
                t = LNRTP2 - x + (x.sqrt() * cheb1).ln();

                // IF ( T .LT. XHIGH2 ) THEN
                if x < XHIGH2 {
                    // SYNCH2 = ZERO
                    return Some(0.0);
                } else {
                    // SYNCH2 = EXP(T)
                    return Some(t.exp());
                }
            }
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;
        // TODO: add proper testing here - not sure how to do that sensibly?

        #[test]
        fn sych_1_range() {
            let n = 1024;
            let (x0, x1) = (0.0, 10.0);
            for x in (0..n).map(|x| {
                let t = (x as f64) / (n - 1) as f64;
                x0 + (x1 - x0) * t
            }) {
                println!("{}, {}", x, synch_1(x).unwrap())
            }
        }

        #[test]
        fn sych_2_range() {
            let n = 1024;
            let (x0, x1) = (0.0, 10.0);
            for x in (0..n).map(|x| {
                let t = (x as f64) / (n - 1) as f64;
                x0 + (x1 - x0) * t
            }) {
                println!("{}, {}", x, synch_2(x).unwrap())
            }
        }
    }
}

pub mod linalg {
    use std::iter::Map;
    use std::ops::{Add, Div, Mul, Neg, Sub};

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

        pub fn cross(&self, rhs: &Self) -> Self {
            todo!()
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

    /// 3x3 matrix in row-major order
    ///
    /// * `v`: matrix content
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub struct Mat3<T> {
        v: [T; 9],
    }

    pub struct Mat3Rows<'a, T> {
        mat: &'a Mat3<T>,
        n: usize
    }

    impl<'a, T> Iterator for Mat3Rows<'a, T> 
        where T : Copy
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
                v: [r0c0,
            r0c1,
            r0c2,
            r1c0,
            r1c1,
            r1c2,
            r2c0,
            r2c1,
            r2c2 ]
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

        pub fn transpose(&self) -> Self {
            todo!()
        }

        pub fn from_columns(cols: &[Vec3<T>; 3]) -> Mat3<T> {
            todo!()
        }

        pub fn try_inverse(&self) -> Option<Mat3<T>> {
            todo!()
        }


        pub fn row_iter(&self) -> Mat3Rows<T> {
            todo!()
        }

        pub fn row(&self, i: usize) -> Vec3<T> 
        where T: Copy
        {
            Vec3::new(
                self[(i, 0)],
                self[(i, 1)],
                self[(i, 2)],
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
        where T: Mul<T, Output = T> + Copy
    {
        type Output = Vec3<T>;

        fn mul(self, rhs: T) -> Self::Output {
            Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
        }
    }

    impl<T> Neg for Vec3<T> 
        where T: Neg<Output=T>
    {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self::new(
                -self.x,
                -self.y,
                -self.z
            )
        }
    }

    impl<T> std::ops::Div<T> for Vec3<T>
        where T: Div<T, Output = T> + Copy
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
        T: Mul<T, Output = T> + Copy
    {
        type Output = Mat3<T>;

        fn mul(self, rhs: T) -> Self::Output 
        {
            Mat3::new(
                self[(0, 0)] * rhs, self[(0, 1)] * rhs, self[(0, 2)] * rhs,
                self[(1, 0)] * rhs, self[(1, 1)] * rhs, self[(1, 2)] * rhs,
                self[(2, 0)] * rhs, self[(2, 1)] * rhs, self[(2, 2)] * rhs,
            )
        }
    }

    impl<T> std::ops::Mul for Mat3<T>
    where
        T: Mul<T, Output = T> + Add<T, Output=T> + Copy
    {
        type Output = Mat3<T>;

        fn mul(self, rhs: Mat3<T>) -> Self::Output 
        {
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


    impl<T> std::ops::Mul<Vec3<T>> for Mat3<T>
    where
        T: Add<T, Output = T> + Mul<T, Output = T> + Copy,
    {
        type Output = Vec3<T>;

        fn mul(self, rhs: Vec3<T>) -> Self::Output {
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
            r0c0,
            r0c1,
            r0c2,
            r0c3,
            r1c0,
            r1c1,
            r1c2,
            r1c3,
            r2c0,
            r2c1,
            r2c2,
            r2c3,
            r3c0,
            r3c1,
            r3c2,
            r3c3,

                ]
            }
        }

        pub fn homog_mul(&self, pos: Vec3<T>) -> Vec3<T> {
            todo!()
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
    }

    impl<'de, T> Deserialize<'de> for Vec3<T> {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            todo!()
        }
    }
}
