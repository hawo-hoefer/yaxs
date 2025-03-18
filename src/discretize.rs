use crate::math::{caglioti, pseudo_voigt, scherrer_broadening};

pub struct Peak {
    // position in degrees two-theta
    pub pos: f64,
    pub intensity: f64,
}

impl Peak {
    /// Render the peak into an XRD pattern
    ///
    /// * `pat`: target pattern
    /// * `two_thetas`: two theta values of pattern's intensities in degrees
    /// * `wavelength`: wavelength of the x-rays in nanometers
    /// * `weight`: weight of the emission line's wavelength
    /// * `mean_ds`: mean domain size used for scherrer broadening
    /// * `u`: caglioti parameter u
    /// * `v`: caglioti parameter v
    /// * `w`: caglioti parameter w
    pub fn into_pattern(
        self,
        pat: &mut [f64],
        two_thetas: &[f64],
        wavelength: f64,
        weight: f64,
        mean_ds: f64,
        eta: f64,
        u: f64,
        v: f64,
        w: f64,
    ) {
        let fwhm = caglioti(u, v, w, self.pos / 2.0)
            + scherrer_broadening(wavelength, self.pos.to_radians() / 2.0, mean_ds);
        for (intensity, two_theta) in pat.iter_mut().zip(two_thetas) {
            *intensity += weight * self.intensity * pseudo_voigt(*two_theta, eta, self.pos, fwhm);
        }
    }
}
