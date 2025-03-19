use crate::background::Background;
use crate::math::{caglioti, pseudo_voigt, scherrer_broadening};
use crate::structure::Structure;

pub struct EmissionLine {
    pub wavelength: f64,
    pub weight: f64,
}

impl EmissionLine {
    /// create a new emission line from wavelength and weight
    ///
    /// * `wavelength`: wavelength in amstrong
    /// * `weight`: intensity of the emission line relative to other emission lines in the spectrum
    pub fn new(wavelength: f64, weight: f64) -> Self {
        Self { wavelength, weight }
    }
}

pub struct Component {
    pub structure: Structure,
    pub volume_fraction: f64,
}

pub struct SimulationJob<'a> {
    pub structures: &'a [Structure],
    pub vol_fractions: Box<[f64]>,
    pub emission_lines: &'a [EmissionLine],
    pub n_steps: u32,
    pub two_theta_range: (f64, f64),
    pub eta: f64,
    pub mean_ds: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub background: Background,
    pub normalize: bool,
}

impl<'a> SimulationJob<'a> {
    pub fn run(&self, two_thetas: &[f64], pat: &mut [f64]) {
        for (s, vf) in self.structures.iter().zip(&self.vol_fractions) {
            for EmissionLine { wavelength, weight } in self.emission_lines.iter() {
                let peaks = s.get_pattern(*wavelength, &self.two_theta_range);
                let wavelength_nm = wavelength / 10.0;
                for peak in peaks {
                    // * `pat`: target pattern
                    // * `two_thetas`: two theta values of pattern's intensities in degrees
                    // * `wavelength`: wavelength of the x-rays in nanometers
                    // * `mean_ds`: mean domain size used for scherrer broadening
                    // * `u`: caglioti parameter u
                    // * `v`: caglioti parameter v
                    // * `w`: caglioti parameter w
                    peak.render(
                        pat,
                        &two_thetas,
                        wavelength_nm,
                        *weight * vf,
                        self.mean_ds,
                        self.eta,
                        self.u,
                        self.v,
                        self.w,
                    )
                }
            }
        }
        if self.normalize {
            // TODO: check for NaNs and normalization
            let f = *pat.first().unwrap();
            let vmin = pat.iter().fold(f, |a, b| f64::min(a, *b));
            let vmax = pat.iter().fold(f, |a, b| f64::max(a, *b));
            pat.iter_mut().for_each(|x| {
                *x = (*x - vmin) / (vmax - vmin);
            });
        }
    }
}

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
    pub fn render(
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
        // TODO: make position in radians
        let theta_pos_rad = self.pos.to_radians() / 2.0;
        let fwhm = caglioti(u, v, w, theta_pos_rad)
            + scherrer_broadening(wavelength, theta_pos_rad, mean_ds);
        let peak_weight = weight * self.intensity;
        for (intensity, two_theta) in pat.iter_mut().zip(two_thetas) {
            let dx = *two_theta - self.pos;
            *intensity += peak_weight * pseudo_voigt(dx, eta, fwhm);
        }
    }
}
