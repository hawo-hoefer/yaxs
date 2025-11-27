use std::sync::Arc;

use cfg_if::cfg_if;
use itertools::Itertools;
use log::{info, warn};
use ndarray::{Array2, Array4};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::background::Background;
use crate::io::PatternMeta;
use crate::math::linalg::Vec3;
use crate::math::{
    e_kev_to_lambda_ams, pseudo_voigt, sample_displacement_delta_theta_rad, scherrer_broadening,
    scherrer_broadening_edxrd, C_M_S, H_EV_S, SQRT_8_LN_2,
};
use crate::noise::Noise;
use crate::structure::Structure;

use self::adxrd::InstrumentParameters;
pub use self::adxrd::{ADXRDMeta, DiscretizeAngleDispersive};
use self::edxrd::Beamline;

use crate::cfg::{CompactSimResults, TextureMeasurement, VolumeFraction};

pub mod adxrd;
pub mod edxrd;

// TODO: Somehow encode that all samples have the same of measurement in
// the type system
pub enum DiscretizeSample<T> {
    Standard(T),
    TextureMeasurement(Vec<T>),
}

impl<T> DiscretizeSample<T> {
    pub fn n_patterns(&self) -> usize {
        match self {
            DiscretizeSample::Standard(_) => 1,
            DiscretizeSample::TextureMeasurement(items) => items.len(),
        }
    }
}

pub trait DiscretizeJobGenerator {
    type Item;

    fn next(&mut self) -> Option<DiscretizeSample<Self::Item>>;
    fn remaining(&self) -> usize;
    fn xs(&self) -> &[f32];
    fn get_job_params(&self) -> JobParams;
}

/// rendering resources for a single phase. these are essentially indices into
/// long vectors of all simulated data
///
/// * `sim_res`:
/// * `indices`:
/// * `impurity_peaks`:
/// * `random_seed`:
/// * `noise`:
#[derive(Clone)]
pub struct RenderCommon {
    // all simulated peaks for all phases in order [structure, structure permutations, (texture_measurement_idx)]
    pub sim_res: Arc<CompactSimResults>,
    // indices to select from simulated peaks, length is number of structures
    pub indices: Box<[usize]>,
    pub impurity_peaks: Box<[ImpurityPeak]>,
    pub random_seed: u64,
    pub noise: Option<Noise>,
}

impl RenderCommon {
    pub fn idx(&self, phase_id: usize) -> usize {
        let perm_id = self.indices[phase_id];
        self.sim_res.idx(phase_id, perm_id)
    }

    pub fn n_phases(&self) -> usize {
        self.indices.len()
    }
}

pub struct VFGenerator {
    pub fraction_sum: f64,
    pub n_free: usize,
    pub fractions: Vec<Option<VolumeFraction>>,
    pub max_subset_dim: Option<ConcentrationSubset>,
}

pub fn get_weight_fractions(
    volume_fractions: &[f64],
    structures: &[Structure],
) -> Option<Box<[f64]>> {
    for s in structures.iter() {
        if s.density.is_none() {
            return None;
        }
    }

    // phi_i = V_i / V_tot
    // V_tot = sum_i V_i
    // V_i = rho_i * m_i
    // V_tot = sum_i rho_i * m_i
    // m_i = rho_i * V_i
    // w_i = m_i / m_ges
    //
    // m_ges = sum_i m_i
    // m_ges = sum_i rho_i V_i
    //       = sum_i rho_i phi_i V_tot
    //       = V_tot * (sum_i rho_i phi_i)
    //
    // w_i = rho_i * V_i / m_ges
    //     = rho_i * V_i / (V_tot * (sum_i rho_i phi_i))
    //     = rho_i / (sum_i rho_i phi_i) * V_i / V_tot
    //     = rho_i / (sum_i rho_i phi_i) * phi_i
    let mut sum_rho_i_phi_i = 0.0;
    let mut mass_fractions = volume_fractions
        .iter()
        .zip(structures)
        .map(|(phi_i, s)| {
            let rho_i = s.density.expect("we have all densities");
            let rpi = rho_i * phi_i;
            sum_rho_i_phi_i += rpi;
            rpi
        })
        .collect_vec();

    // normalize again
    for rpi in mass_fractions.iter_mut() {
        *rpi /= sum_rho_i_phi_i;
    }

    Some(mass_fractions.into())
}

/// sample integers uniformly without replacement from the interval [0, max_val)
///
/// from here https://stackoverflow.com/questions/311703/algorithm-for-sampling-without-replacement
///
/// * `n`: number of samples
/// * `max_val`: upper bound of the
/// * `rng`: random number generator
pub fn uniform_sample_no_replacement_knuth(
    n: usize,
    max_val: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    // TODO: does this belong in stats?
    let mut samples = Vec::with_capacity(n);
    let mut t = 0;
    while samples.len() < n {
        let u = rng.random_range(0.0..=1.0);
        if (max_val - t) as f64 * u >= (n - samples.len()) as f64 {
            t += 1;
        } else {
            samples.push(t);
            t += 1;
        }
    }
    samples
}

#[derive(PartialEq, Clone, Debug, Serialize)]
pub enum ConcentrationSubset {
    MaxDim(usize),
    Probabilities(Vec<f64>),
}

impl ConcentrationSubset {
    pub fn roll(&self, rng: &mut impl Rng) -> usize {
        use ConcentrationSubset::*;
        match self {
            MaxDim(maxdim) => {
                if *maxdim == 0 {
                    // should this be an error?
                    0
                } else {
                    rng.random_range(1..=*maxdim)
                }
            }
            Probabilities(probs) => {
                let t = rng.random_range(0.0..=1.0);
                let mut acc = 0.0;
                for (i, p) in probs.iter().enumerate() {
                    acc += p;
                    if t <= acc {
                        return i + 1;
                    }
                }

                unreachable!("t is between 0 and 1")
            }
        }
    }

    pub fn max_subset_size(&self) -> usize {
        match self {
            ConcentrationSubset::MaxDim(n) => *n,
            ConcentrationSubset::Probabilities(items) => items.len(),
        }
    }
}

impl<'de> Deserialize<'de> for ConcentrationSubset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum CSProxy {
            MaxDim(usize),
            Weights(Vec<f64>),
        }

        let p = CSProxy::deserialize(deserializer)?;

        match p {
            CSProxy::MaxDim(d) => return Ok(ConcentrationSubset::MaxDim(d)),
            CSProxy::Weights(mut items) => {
                for i in items.iter() {
                    if *i < 0.0 {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Float(*i),
                            &"subset size weights needs to be larger than 0.",
                        ));
                    }
                }
                let sum = items.iter().sum::<f64>();

                if sum == 0.0 {
                    return Err(serde::de::Error::invalid_value(
                        serde::de::Unexpected::Float(sum),
                        &"Concentration subset weights need to sum to more than 0.",
                    ));
                }

                for i in items.iter_mut() {
                    *i /= sum;
                }

                return Ok(ConcentrationSubset::Probabilities(items));
            }
        }
    }
}

impl VFGenerator {
    pub fn try_new(
        mut fractions: Vec<Option<VolumeFraction>>,
        max_subset_dim: Option<ConcentrationSubset>,
    ) -> Result<Self, String> {
        if fractions.len() == 0 {
            // no structures is ok. in that case, no volume fractions will be generated
            return Ok(Self {
                n_free: 0,
                fraction_sum: 0.0,
                fractions,
                max_subset_dim: None, // no subsets if no fractions are specified
            });
        }
        let fraction_sum = fractions.iter().filter_map(|x| x.map(|x| x.0)).sum::<f64>();
        let n_free = fractions.iter().filter(|x| x.is_none()).count();
        const ATOL: f64 = 1e-5;

        if fraction_sum > 1.0 + ATOL {
            return Err(format!(
                "Specified fractions must to sum to less than or equal to 1.0. Got sum: {}",
                fraction_sum
            ));
        }
        if fraction_sum > 1.0 {
            // ignore the extra volume fraction < ATOL
            warn!("Fraction sum is larger than 1.0 but inside the tolerance: {fraction_sum}. Reducing each set phase equally");
            let delta = fraction_sum - 1.0;
            let n_set = fractions.len() - n_free;
            for fraction in fractions.iter_mut().filter_map(|x| match x {
                Some(v) => Some(v),
                None => None,
            }) {
                fraction.0 -= delta / n_set as f64;
            }
        }

        if fraction_sum > 0.99 && n_free > 0 {
            warn!("Fraction sum ({fraction_sum:.3}) is close to 1. There are {n_free} non-fixed volume fractions which are strongly constrained because of this.");
        }

        if n_free == 0 && fraction_sum < 1.0 - ATOL {
            // no free parameters and fraction sum smaller than 1
            return Err(format!("All structures volume fractions are fixed, but the sum of their fractions is smaller than one (delta: {d:.2e}). Make sure that the volume fractions add up to 1, or remove the specification for one fraction if you want to compute it automatically.", d = 1.0 - fraction_sum));
        }

        if let Some(max_subset_dim) = &max_subset_dim {
            let max_subset_size = max_subset_dim.max_subset_size();
            if max_subset_size > n_free {
                return Err(format!("max_subset_dim can be at most equal to the number of free phases. Expected < {n_free}, got {max_subset_size}"));
            }
        }

        Ok(Self {
            n_free,
            fraction_sum,
            fractions,
            max_subset_dim,
        })
    }

    pub fn generate(&self, rng: &mut impl Rng) -> Box<[f64]> {
        // Generate n_free random numbers summing to self.fraction_sum in the
        // first n_free slots of the concentration buffer.
        //
        // This is done by initializing the first slot to 0.0, the next
        // n_free - 2 to a random number in [0, self.fraction_sum], and
        // the one at n_free to self.fraction_sum.
        //
        // Then, the first few elements are sorted, and the a difference is taken
        // between adjacent elements to produce the random numbers summing to one.

        if self.fractions.len() == 0 {
            // handle no structures case explicitly.
            // theoretically, the code below should 'just work'
            // but it's much clearer to do it this way.
            return Vec::new().into();
        }

        let mut concentration_buf = Vec::with_capacity(self.fractions.len() + 1);

        let roll_n = if let Some(max_subset_dim) = &self.max_subset_dim {
            max_subset_dim.roll(rng)
        } else {
            self.n_free
        };

        if roll_n > 0 {
            concentration_buf.push(0.0);
            let free_mass = 1.0 - self.fraction_sum;
            concentration_buf.extend((0..roll_n - 1).map(|_| rng.random_range(0.0..=free_mass)));
            concentration_buf.push(free_mass);

            concentration_buf[1..roll_n]
                .sort_unstable_by(|a, b| a.partial_cmp(b).expect("not nan"));

            // compute the difference
            for i in 0..concentration_buf.len() - 1 {
                concentration_buf[i] = concentration_buf[i + 1] - concentration_buf[i];
            }
        }
        concentration_buf.resize(concentration_buf.capacity(), 0.0);

        // compute zeroed entries because of allow_subsets
        // sample n_zeroed indices smaller than self.n_free without replacement
        let n_zeroed = self.n_free - roll_n;
        let mut zeroed_indices = uniform_sample_no_replacement_knuth(n_zeroed, self.n_free, rng);
        zeroed_indices.sort();

        // shift the zeroed indices such that they won't lie on fixed value positions
        for (pos, f) in self.fractions.iter().enumerate() {
            if f.is_some() {
                for s in zeroed_indices.iter_mut() {
                    if *s >= pos {
                        *s += 1;
                    }
                }
            }
        }

        // place the fixed numbers at the correct positions
        if roll_n < self.fractions.len() {
            let mut free_idx = self.n_free - n_zeroed;
            for (idx, fraction) in self.fractions.iter().enumerate() {
                if let Some(fraction) = fraction {
                    // before:
                    //          idx       free_idx
                    //           |           |
                    // +---+---+---+---+---+---+---+---+
                    // | ? | ? | A | ? | ? | ? | ? | ? |
                    // +---+---+---+---+---+---+---+---+
                    //           |           ^
                    //           |   move    |
                    //           +-----------+
                    //
                    // move from idx to free_idx
                    // write the fixed value (V)
                    // increment free_idx
                    //
                    // state after:
                    //              idx       free_idx
                    //               |           |
                    // +---+---+---+---+---+---+---+---+
                    // | ? | ? | V | ? | ? | A | ? | ? |
                    // +---+---+---+---+---+---+---+---+
                    concentration_buf[free_idx] = concentration_buf[idx];
                    concentration_buf[idx] = fraction.0;
                    free_idx += 1;
                }

                if zeroed_indices.binary_search(&idx).is_ok() {
                    concentration_buf[free_idx] = concentration_buf[idx];
                    concentration_buf[idx] = 0.0;
                    free_idx += 1;
                }
            }
        }

        concentration_buf.truncate(self.fractions.len());
        concentration_buf.into_boxed_slice()
    }
}

pub fn lorentz_polarization_factor(theta_rad: f64) -> f64 {
    // TODO: revisit lorentz polarization. currently, we assume no polarization due to
    // monochromator. is that correct?
    // let lorentz_factor = 1.0 / (4.0 * theta_rad.sin().powi(2) * theta_rad.cos());
    // let polarization_factor = 0.5 * (1.0 + (2.0 * theta_rad).cos().powi(2));

    (1.0 + (2.0 * theta_rad).cos().powi(2)) / (theta_rad.sin().powi(2) * theta_rad.cos())
}

fn edxrd_polarization_factor_horizontal_plane(theta_rad: f64) -> f64 {
    (theta_rad * 2.0).cos().powi(2)
}

pub struct JobParams {
    pub abstol: f32,
    pub n_phases: usize,
    pub has_weight_fracs: bool,
    pub textured_phases: Option<usize>,
    pub texture_measurement: Option<TextureMeasurement>,
}

pub struct PeakRenderParams {
    pub pos: f32,
    pub intensity: f32,
    pub fwhm: f32,
    pub eta: f32,
}

#[derive(Debug, Clone)]
pub struct ImpurityPeak {
    pub peak: Peak,
    pub eta: f64,
    pub mean_ds_nm: f64,
}

pub trait Discretizer {
    fn peak_info_iterator(&self) -> impl Iterator<Item = PeakRenderParams>;
    fn n_peaks_tot(&self) -> usize;
    fn bkg(&self) -> &Background;
    fn seed(&self) -> u64;
    fn normalize(&self) -> bool;
    fn noise(&self) -> &Option<Noise> {
        &None
    }

    fn write_meta_data(&self, key: &mut PatternMeta, pat_id: usize);
    fn init_meta_data(n_samples: usize, p: &JobParams) -> Vec<PatternMeta>;

    fn discretize_into(&self, intensities: &mut [f32], positions: &[f32], abstol: f32) {
        // rendering the backgrounds first allows for background scaling without fancy
        // math or extra memory allocation
        self.bkg().render(intensities, positions);

        for p in self.peak_info_iterator() {
            p.render(positions, intensities, abstol)
        }

        if let Some(noise) = self.noise() {
            noise.apply(intensities, self.seed());
        }

        if self.normalize() {
            let f = *intensities.first().unwrap();
            let vmin = intensities.iter().fold(f, |a, b| f32::min(a, *b));
            let vmax = intensities.iter().fold(f, |a, b| f32::max(a, *b));
            intensities.iter_mut().for_each(|x| {
                *x = (*x - vmin) / (vmax - vmin);
            });
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
/// A diffraction peak with d_hkl in amstrong and i_hkl in arbitrary units
///
/// * `d_hkl`: crystal lattice distance in amstrong
/// * `i_hkl`: intensity in arbitrary units
/// * `hkls`: miller indices corresponding to peak
pub struct Peak {
    pub d_hkl: f64,
    pub i_hkl: f64,
    pub hkls: Vec<Vec3<i16>>,
}
pub type Peaks = Box<[Peak]>;

impl PeakRenderParams {
    pub fn render(self, xs: &[f32], ys: &mut [f32], abstol: f32) {
        let n = xs.len();
        let midpoint = ((self.pos - xs[0]) / (xs[n - 1] - xs[0]) * n as f32) as usize;

        let mut i = midpoint;
        if i > n - 1 {
            i = n - 1
        }

        // left half
        loop {
            let dx = xs[i] - self.pos;
            let di = self.intensity * pseudo_voigt(dx, self.eta, self.fwhm);
            if di < abstol {
                break;
            }
            ys[i] += di;
            if i == 0 {
                break;
            }
            i -= 1;
        }

        // right half
        i = midpoint + 1;
        while i < n {
            let dx = xs[i] - self.pos;
            let di = self.intensity * pseudo_voigt(dx, self.eta, self.fwhm);
            if di < abstol {
                break;
            }
            ys[i] += di;
            i += 1;
        }
    }
}

/// Compute Pseudo-Voigt eta from gaussian and pseudo-voigt fwhms
///
/// use the approximation given in equation 2 of
///
/// Thompson, P., D. E. Cox, and J. B. Hastings.
/// "Rietveld refinement of Debye–Scherrer synchrotron X-ray data from Al2O3."
/// Applied Crystallography 20.2 (1987): 79-83.
///
/// https://doi.org/10.1107/s0021889887087090
///
/// * `pv_fwhm`: pseudo-voigt fwhm
/// * `g_fwhm`: gaussian fwhm
fn compute_pv_eta(pv_fwhm: f64, l_fwhm: f64) -> f64 {
    let mut eta = 0.0;

    let frac = l_fwhm / pv_fwhm;
    let mut pf = 1.0;
    for coef in [1.36603, -0.47719, 0.11116] {
        pf *= frac;
        eta += coef * pf;
    }

    return eta;
}

/// Compute the pseudo-voigt fwhm from fwhms of the gaussian and
/// lorentzian components
///
/// uses the approximation in equation 3 of
///
/// Thompson, P., D. E. Cox, and J. B. Hastings.
/// "Rietveld refinement of Debye–Scherrer synchrotron X-ray data from Al2O3."
/// Applied Crystallography 20.2 (1987): 79-83.
///
/// https://doi.org/10.1107/s0021889887087090
///
///
/// * `g_fwhm`: gaussian fwhm
/// * `l_fwhm`: lorentzian fwhm
fn compute_pv_fwhm(g_fwhm: f64, l_fwhm: f64) -> f64 {
    let mut x = g_fwhm.powi(5);
    let mut sum = 0.0;
    for coef in [1.0, 2.69269, 2.42843, 4.47163, 0.07842, 1.0] {
        sum += coef * x;
        x = x / g_fwhm * l_fwhm;
    }
    sum.powf(0.2)
}

fn compute_pv_params_from_fwhms(g_fwhm: f64, l_fwhm: f64) -> (f64, f64) {
    let pv_fwhm = compute_pv_fwhm(g_fwhm, l_fwhm);
    (compute_pv_eta(pv_fwhm, l_fwhm), pv_fwhm)
}

impl Peak {
    /// Get ADXRD Peak location, intensity and fwhm
    ///
    /// * `wavelength_nm`: X-ray wavelength
    /// * `caglioti`: Caglioti instrument parameters for gaussian line broadening
    /// * `eta_size_broadening`: gaussian-lorentzian mixing parameter for size broadening
    /// * `mean_ds_nm`: mean domain size in nanometers
    /// * `weight`: weight of the peak (usually something like volume fraction multiplied by the
    ///   emission line's relative intensity)
    #[allow(clippy::too_many_arguments)]
    pub fn get_adxrd_render_params(
        &self,
        wavelength_nm: f64,
        instrument_parameters: &InstrumentParameters,
        mean_ds_nm: f64,
        ds_eta: f64,
        mustrain: f64,
        mustrain_eta: f64,
        weight: f64,
        sample_displacement_mu_m: f64,
        goniometer_radius_mm: f64,
    ) -> PeakRenderParams {
        // bragg condition
        // lambda = 2 d sin(theta)
        // theta = asin(lambda / 2d)
        let wavelength_ams = wavelength_nm * 10.0;
        let theta_hkl_rad = {
            let theta_hkl_rad = (wavelength_ams / (2.0 * self.d_hkl)).asin();

            let sd_delta_theta_rad = sample_displacement_delta_theta_rad(
                sample_displacement_mu_m,
                goniometer_radius_mm,
                theta_hkl_rad,
            );

            theta_hkl_rad + sd_delta_theta_rad
        };

        let f_lorentz = lorentz_polarization_factor(theta_hkl_rad);

        // use names from GSAS for now
        // size and microstrain broadening fwhms
        let sgam = scherrer_broadening(wavelength_nm, theta_hkl_rad, mean_ds_nm);
        let mgam = (mustrain * theta_hkl_rad.tan()).to_degrees();

        // FWHM = sqrt(8 ln 2) sigma
        // sigma^2 = FWHM^2 / 8 ln 2
        #[rustfmt::skip]
        let mut sample_g_fwhm_sq = (sgam * (1.0 -       ds_eta)).powi(2)
                                 + (mgam * (1.0 - mustrain_eta)).powi(2);
        sample_g_fwhm_sq /= 8.0 * std::f64::consts::LN_2;
        let sample_l_fwhm = sgam * ds_eta + mgam * mustrain_eta;

        // clip sigma^2 at 0.001 centidegrees^2 = 0.0000001 degrees^2 (like GSAS)
        let g_fwhm_sq = (instrument_parameters.gauss_broadening(theta_hkl_rad) + sample_g_fwhm_sq)
            .max(0.0000001);

        // clip gamma at 0.001 centidegrees = 0.00001 degrees (like GSAS)
        // multiply by 2 to get gamma, and by 2 another time to get gamma in 2-theta instead
        // of theta
        let l_fwhm = instrument_parameters.lorentz_broadening(theta_hkl_rad) + sample_l_fwhm;
        let l_fwhm = l_fwhm.max(0.00001);

        let (eta, fwhm) = compute_pv_params_from_fwhms(g_fwhm_sq.sqrt() * SQRT_8_LN_2, l_fwhm);

        let peak_weight = (self.i_hkl * f_lorentz * wavelength_ams.powi(3) * weight) as f32;

        PeakRenderParams {
            pos: (theta_hkl_rad.to_degrees() * 2.0) as f32,
            intensity: peak_weight,
            fwhm: fwhm as f32,
            eta: eta as f32,
        }
    }

    /// Get EDXRD Peak location, intensity and fwhm
    ///
    /// * `wavelength_nm`: X-ray wavelength
    /// * `mean_ds_nm`: mean domain size in nanometers
    /// * `weight`: weight of the peak (usually something like volume fraction multiplied by the
    ///    emission line's relative intensity)
    pub fn get_edxrd_render_params(
        &self,
        theta_rad: f64,
        f_lorentz: f64,
        mean_ds_nm: f64,
        ds_eta: f64,
        weight: f64,
        beamline: &Beamline,
    ) -> PeakRenderParams
where {
        // here, we apply intensity corrections to each peak, and
        // convert positions from d_hkl in Amstrong to energy in keV
        let hc = H_EV_S * C_M_S * 1e7;

        // bragg condition
        // d = h c / (2 E sin (theta))
        // 2 E sin(theta) = h c / d
        // E = h c / (2 d sin(theta))
        //
        // hc in eV m
        // eV * m * (m^-10)
        // ev * e-10
        // g_hkl in ams = m^-10
        let e_kev = hc / (2.0 * self.d_hkl * theta_rad.sin());

        let beamline_intensity = beamline.get_intensity(e_kev);
        let polarization_correction = edxrd_polarization_factor_horizontal_plane(theta_rad);
        let peak_weight = self.i_hkl
            * f_lorentz
            * polarization_correction
            * e_kev_to_lambda_ams(e_kev).powi(3)
            * beamline_intensity
            * weight;

        let fwhm = scherrer_broadening_edxrd(theta_rad, mean_ds_nm);

        let g_fwhm = (1.0 - ds_eta) * fwhm;
        let l_fwhm = (1.0 - ds_eta) * fwhm;

        let (eta, fwhm) = compute_pv_params_from_fwhms(g_fwhm, l_fwhm);

        PeakRenderParams {
            pos: e_kev as f32,
            intensity: peak_weight as f32,
            fwhm: fwhm as f32,
            eta: eta as f32,
        }
    }
}

pub enum Intensities {
    /// One sample is just a single XRD measurement. the dimensions are
    /// therefore [n_samples, pattern_steps]
    Standard(Array2<f32>),
    /// For Texture Measurements, one sample is represented by n * m
    /// Xrd patterns, where n and m are the resolution in phi and chi, respectively
    /// therefore, the resolution must be [n_samples, phi_steps, chi_steps, pattern_steps]
    TextureMeasurement(Array4<f32>),
}

pub fn render_jobs<T>(
    jobs: Vec<DiscretizeSample<T>>,
    xs: &[f32],
    p: &JobParams,
) -> Result<(Intensities, Vec<PatternMeta>), String>
where
    T: Discretizer + Send + Sync + 'static,
{
    let n_samples = jobs.len();
    let mut metadata = T::init_meta_data(n_samples, p);
    info!("Initialized metadata for {n_samples} sample(s).");

    // TODO: incorporate bkg_coefs into JobParams
    // NOTE: currently, we are only able to render one kind of background per simulation
    // should this ever change, we need to adapt this implementation
    // let n_bkg_params = j.bkg().bkg_coefs();
    // let mut metadata = T::init_meta_data(jobs.len(), n_phases, with_weight_fractions, n_bkg_params);
    for (i, job) in jobs.iter().enumerate() {
        for m in metadata.iter_mut() {
            let job = match job {
                DiscretizeSample::Standard(job) => job,
                DiscretizeSample::TextureMeasurement(items) => items
                    .first()
                    .expect("at least one pattern in texture measurement"),
            };
            job.write_meta_data(m, i)
        }
    }

    let n_peak_sets = jobs.iter().map(|x| x.n_patterns()).sum();

    // actual rendering of the patterns
    cfg_if! {
        if #[cfg(feature = "use-gpu")] {
            use crate::discretize_cuda::discretize_peaks_cuda;
            let intensities = discretize_peaks_cuda(jobs, xs)?;
            let intensities = ndarray::Array2::from_shape_vec((n_peak_sets, xs.len()), intensities)
                .expect("sizes must match");
        } else {
        let mut intensities = Array2::<f32>::zeros((n_peak_sets, xs.len()));
            let mut peak_set = 0;
            for job in jobs {
                // TODO: somehow encode that all samples have the same simulation type
                // in the type system
                match job {
                    DiscretizeSample::Standard(job) => {
                        job.discretize_into(intensities.row_mut(peak_set).as_slice_mut().unwrap(), &xs, p.abstol);
                        peak_set += 1;
                    },
                    DiscretizeSample::TextureMeasurement(items) => {
                        for job in items.iter() {
                            job.discretize_into(intensities.row_mut(peak_set).as_slice_mut().unwrap(), &xs, p.abstol);
                            peak_set += 1;
                        }
                    },
                }
            }
        }
    };

    let intensities = if let Some(t) = p.texture_measurement {
        Intensities::TextureMeasurement(
            intensities
                .into_shape_with_order((n_samples, t.chi.steps, t.phi.steps, xs.len()))
                .expect("shapes match"),
        )
    } else {
        Intensities::Standard(intensities)
    };

    Ok((intensities, metadata))
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn vf_generation_basic() {
        let n = 5;
        let vfs = (0..n).map(|_| None).collect_vec();
        let gen = VFGenerator::try_new(vfs, None).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let generated = gen.generate(&mut rng);
        assert_eq!(generated.len(), n);
        assert_eq!(generated.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn vf_generation_single_fixed() {
        let vfs = vec![None, None, Some(VolumeFraction(0.3))];
        let n = vfs.len();
        let gen = VFGenerator::try_new(vfs, None).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let generated = gen.generate(&mut rng);
        assert_eq!(generated[2], 0.3);
        assert_eq!(generated.len(), n);
        assert_eq!(generated.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn vf_generation_multiple_fixed() {
        let vfs = vec![
            Some(VolumeFraction(0.2)),
            None,
            None,
            Some(VolumeFraction(0.3)),
        ];
        let n = vfs.len();
        let gen = VFGenerator::try_new(vfs, None).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let generated = gen.generate(&mut rng);
        assert_eq!(generated[0], 0.2);
        assert_eq!(generated[3], 0.3);
        assert_eq!(generated.len(), n);
        assert_eq!(generated.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn vf_generation_multiple_inbetween_fixed() {
        let vfs = vec![
            Some(VolumeFraction(0.2)),
            None,
            Some(VolumeFraction(0.1)),
            None,
            Some(VolumeFraction(0.3)),
        ];

        let n = vfs.len();
        let gen = VFGenerator::try_new(vfs, None).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let generated = gen.generate(&mut rng);
        assert_eq!(generated[0], 0.2);
        assert_eq!(generated[2], 0.1);
        assert_eq!(generated[4], 0.3);
        assert_eq!(generated.len(), n);
        assert_eq!(generated.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn vf_generation_no_degrees_of_freedom() {
        let vfs = vec![Some(VolumeFraction(0.2)), Some(VolumeFraction(0.1)), None];

        let n = vfs.len();
        let gen = VFGenerator::try_new(vfs, None).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let generated = gen.generate(&mut rng);
        assert_eq!(generated[0], 0.2);
        assert_eq!(generated[1], 0.1);
        assert_eq!(generated[2], 0.7);
        assert_eq!(generated.len(), n);
        assert_eq!(generated.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn vf_generation_no_degrees_of_freedom_two_phases() {
        let vfs = vec![Some(VolumeFraction(0.6)), Some(VolumeFraction(0.4))];

        let n = vfs.len();
        let gen = VFGenerator::try_new(vfs, None).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let generated = gen.generate(&mut rng);
        assert_eq!(generated[0], 0.6);
        assert_eq!(generated[1], 0.4);
        assert_eq!(generated.len(), n);
        assert_eq!(generated.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn vf_generation_allow_subsets() {
        let vfs = vec![
            None,
            Some(VolumeFraction(0.3)),
            None,
            None,
            Some(VolumeFraction(0.3)),
        ];

        let n = vfs.len();
        let gen = VFGenerator::try_new(vfs, Some(ConcentrationSubset::MaxDim(2))).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let mut n_zeroed = 0;
        for _ in 0..1000 {
            let generated = gen.generate(&mut rng);
            assert_eq!(generated[1], 0.3);
            assert_eq!(generated[4], 0.3);
            assert_eq!(generated.len(), n);
            assert!((generated.iter().sum::<f64>() - 1.0).abs() < 1e-10);
            for v in generated.iter() {
                if *v == 0.0 {
                    n_zeroed += 1;
                }
            }
        }
        assert!(n_zeroed > 0)
    }
}
