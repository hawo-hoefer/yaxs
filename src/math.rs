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

/// calculate scherrer broadening
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

pub fn gauss(x: f64, mu: f64, sigma: f64) -> f64 {
    (-0.5 * ((x - mu) / sigma).powi(2)).exp() / (std::f64::consts::TAU * sigma.powi(2))
}

pub fn lorentz(x: f64, mu: f64, gamma: f64) -> f64 {
    1.0 / ((1.0 + ((x - mu) / gamma).powi(2)) * std::f64::consts::PI * gamma)
}

pub fn pseudo_voigt(x: f64, eta: f64, pos: f64, fwhm: f64) -> f64 {
    // sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * fwhm
    // fwhm = 2 * sqrt(2 ln(2)) sigma
    // sigma = fwhm / (2 sqrt(2 ln 2))
    let two_sqrt_ln_2 = 2.0 * (2.0f64.ln() * 2.0).sqrt();
    let sigma = (1.0 / two_sqrt_ln_2) * fwhm;
    let gamma = fwhm / 2.0;
    eta * lorentz(x, pos, gamma) + (1.0 - eta) * gauss(x, pos, sigma)
}
