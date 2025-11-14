use std::f32::consts::{FRAC_2_SQRT_PI, PI, TAU};

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

pub const SQRT_8_LN_2: f64 = 2.3548200450309493;

pub fn e_kev_to_lambda_ams(e_kev: f64) -> f64 {
    // e = h * c / lambda
    // lambda = h * c / e
    // m      = ev * s * m / ev
    H_EV_S * C_M_S / e_kev * 1e7
}

/// Scherrer broadening constant
//
/// like GSAS-II, we use the volume-weighted domain size, and therefore
/// can set K to 1
const K: f64 = 1.0;

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
    (-0.5 * (dx / sigma).powi(2)).exp() * std::f32::consts::FRAC_2_SQRT_PI / sigma
}

pub fn lorentz(dx: f32, gamma: f32) -> f32 {
    gamma / ((gamma.powi(2) + dx.powi(2)) * PI)
}

pub fn pseudo_voigt(dx: f32, eta: f32, fwhm: f32) -> f32 {
    // sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * fwhm
    // fwhm = 2 * sqrt(2 ln(2)) sigma
    // sigma = fwhm / (2 sqrt(2 ln 2))
    let sigma = fwhm / SQRT_8_LN_2 as f32;
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
        (d1 - d2) / 2.0
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
            3.036_468_298_250_107_6e1,
            1.707_939_527_740_839_5e1,
            4.560_132_133_545_073,
            5.492_812_467_304_2e-1,
            3.729_760_750_693_012e-2,
            1.613_624_302_010_412_5e-3,
            4.819_167_721_203_707e-5,
            0.10512_42528_89384e-5,
            0.17463_85046_697e-7,
            0.22815_48654_4e-9,
            0.24044_3082e-11,
            0.20865_88e-13,
            0.15167e-15,
            0.94e-18,
        ];
        const ASYNC2: [f64; 12] = [
            4.490_721_623_532_661e-1,
            8.983_536_779_941_872e-2,
            8.104_457_377_215_13e-3,
            4.261_716_991_089_162e-4,
            1.476_096_312_707_46e-5,
            0.36286_33615_3998e-6,
            0.66634_80749_84e-8,
            0.94907_71655e-10,
            0.10791_2491e-11,
            0.10022_01e-13,
            0.7745e-16,
            0.51e-18,
        ];
        const ASYNCA: [f64; 25] = [
            2.132_930_516_135_5,
            7.413_528_649_542_002e-2,
            8.696_809_990_996_42e-3,
            1.170_382_624_877_569_2e-3,
            1.645_105_798_619_191_5e-4,
            2.402_010_214_206_403e-5,
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
        const CONLOW: f64 = 2.149_528_241_534_478_7;
        const LNRTP2: f64 = 0.225_791_352_644_727_44;
        const PIBRT3: f64 = 1.813_799_364_234_217_8;

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
        Some(t.exp())
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
            3.861_783_992_384_308_6e1,
            2.303_771_559_496_373_6e1,
            5.380_249_986_833_570_5,
            6.156_793_806_995_711e-1,
            4.066_880_046_688_956e-2,
            1.729_627_455_264_841_3e-3,
            5.106_125_883_657_699e-5,
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
            7.906_314_827_066_081,
            3.135_346_361_285_342_7,
            4.854_879_477_453_714_6e-1,
            3.948_166_758_272_372e-2,
            1.966_162_233_480_88e-3,
            6.590_789_322_930_42e-5,
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
            2.020_337_094_170_713_5,
            1.095_623_712_180_740_5e-2,
            8.542_384_730_114_676e-4,
            7.234_302_421_328_223e-5,
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
        const CONLOW: f64 = 1.074_764_120_767_239_4;
        const LNRTP2: f64 = 0.225_791_352_644_727_44;

        // IF ( X .LE. FOUR ) THEN
        if x <= 4.0 {
            // Code for 0 <= x <= 4
            // XPOWTH = X ** ( ONE / THREE )
            let xpowth = x.powf(1.0 / 3.0);

            // IF ( X .LT. XLOW ) THEN
            if x < XLOW {
                // SYNCH2 = CONLOW * XPOWTH
                Some(CONLOW * xpowth)
            } else {
                // T = ( X * X / EIGHT - HALF ) - HALF
                let t = x.powi(2) / 8.0 - 1.0;
                // CHEB1 = CHEVAL(NTERM1,ASYN21,T)
                // CHEB2 = CHEVAL(NTERM2,ASYN22,T)
                let cheb1 = cheval(&ASYN21, t);
                let cheb2 = cheval(&ASYN22, t);

                // SYNCH2 = XPOWTH * CHEB1 - ( XPOWTH**5 ) * CHEB2
                Some(xpowth * cheb1 - xpowth.powi(5) * cheb2)
            }
        } else {
            // IF ( X .GT. XHIGH1 ) THEN
            if x > XHIGH1 {
                // SYNCH2 = ZERO
                Some(0.0)
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
                    Some(0.0)
                } else {
                    // SYNCH2 = EXP(T)
                    Some(t.exp())
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
