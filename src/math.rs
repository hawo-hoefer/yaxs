use std::f32::consts::{PI, TAU};

/// plack's constant in ev * s = ev * hz^-1
pub const H_EV_S: f64 = 4.135_667_696e-15f64;

/// speed of light in m / s
pub const C_M_S: f64 = 299_792_485.0f64;

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
/// https://doi.org/10.1088/0957-0233/14/3/313
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
