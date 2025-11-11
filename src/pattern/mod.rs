use std::sync::Arc;

use cfg_if::cfg_if;
use itertools::Itertools;
use log::warn;
use ndarray::Array2;
use rand::Rng;
use serde::de::Visitor;
use serde::{Deserialize, Serialize};

use crate::background::Background;
use crate::io::PatternMeta;
use crate::math::linalg::Vec3;
use crate::math::{
    e_kev_to_lambda_ams, pseudo_voigt, sample_displacement_delta_two_theta_rad,
    scherrer_broadening, scherrer_broadening_edxrd, C_M_S, H_EV_S,
};
use crate::noise::Noise;
use crate::structure::Structure;

use self::adxrd::Caglioti;
pub use self::adxrd::{ADXRDMeta, DiscretizeAngleDispersive};
use self::edxrd::Beamline;

use crate::cfg::{CompactSimResults, VolumeFraction};

pub mod adxrd;
pub mod edxrd;

pub trait DiscretizeJobGenerator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;
    fn remaining(&self) -> usize;
    fn xs(&self) -> &[f32];
    fn n_phases(&self) -> usize;
    fn abstol(&self) -> f32;
    fn with_weight_fractions(&self) -> bool;
}

pub struct RenderCommon {
    // all simulated peaks for all phases in order [structure, structure permutations]
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
    (1.0 + (2.0 * theta_rad).cos().powi(2)) / (theta_rad.sin().powi(2) * theta_rad.cos())
}

fn edxrd_polarization_factor_horizontal_plane(theta_rad: f64) -> f64 {
    (theta_rad * 2.0).cos().powi(2)
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
    fn init_meta_data(
        n_patterns: usize,
        n_phases: usize,
        with_weight_fractions: bool,
        bkg_params: Option<usize>,
    ) -> Vec<PatternMeta>;

    fn discretize_into(&self, intensities: &mut [f32], positions: &[f32], abstol: f32) {
        // rendering the backgrounds first allows for background scaling without fancy
        // math or extra memory allocation
        self.bkg().render(intensities, positions);

        for PeakRenderParams {
            pos,
            intensity,
            fwhm,
            eta,
        } in self.peak_info_iterator()
        {
            render_peak(pos, intensity, fwhm, eta, abstol, positions, intensities)
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

pub fn render_peak(
    pos: f32,
    weight: f32,
    fwhm: f32,
    eta: f32,
    abstol: f32,
    xs: &[f32],
    ys: &mut [f32],
) {
    let n = xs.len();
    let midpoint = ((pos - xs[0]) / (xs[n - 1] - xs[0]) * n as f32) as usize;

    let mut i = midpoint;
    if i > n - 1 {
        i = n - 1
    }

    // left half
    loop {
        let dx = xs[i] - pos;
        let di = weight * pseudo_voigt(dx, eta, fwhm);
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
        let dx = xs[i] - pos;
        let di = weight * pseudo_voigt(dx, eta, fwhm);
        if di < abstol {
            break;
        }
        ys[i] += di;
        i += 1;
    }
}

impl Peak {
    /// Get ADXRD Peak location, intensity and fwhm
    ///
    /// * `wavelength_nm`: X-ray wavelength
    /// * `u`: caglioti u parameter
    /// * `v`: caglioti v parameter
    /// * `w`: caglioti w parameter
    /// * `mean_ds_nm`: mean domain size in nanometers
    /// * `weight`: weight of the peak (usually something like volume fraction multiplied by the
    ///   emission line's relative intensity)
    #[allow(clippy::too_many_arguments)]
    pub fn get_adxrd_render_params(
        &self,
        wavelength_nm: f64,
        caglioti: &Caglioti,
        mean_ds_nm: f64,
        weight: f64,
        sample_displacement_mu_m: f64,
        goniometer_radius_mm: f64,
    ) -> (f32, f32, f32) {
        // bragg condition
        // lambda = 2 d sin(theta)
        // theta = asin(lambda / 2d)
        let wavelength_ams = wavelength_nm * 10.0;
        let theta_hkl_rad = (wavelength_ams / (2.0 * self.d_hkl)).asin();
        let f_lorentz = lorentz_polarization_factor(theta_hkl_rad);
        let fwhm = caglioti.broadening(theta_hkl_rad)
            + scherrer_broadening(wavelength_nm, theta_hkl_rad, mean_ds_nm);
        let peak_weight = (self.i_hkl * f_lorentz * wavelength_ams.powi(3) * weight) as f32;

        let sd_delta_two_theta_rad = sample_displacement_delta_two_theta_rad(
            sample_displacement_mu_m,
            goniometer_radius_mm,
            theta_hkl_rad,
        );

        let two_theta_hkl_deg = (2.0 * theta_hkl_rad + sd_delta_two_theta_rad).to_degrees() as f32;
        (two_theta_hkl_deg, peak_weight, fwhm as f32)
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
        weight: f64,
        beamline: &Beamline,
    ) -> (f32, f32, f32)
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

        let fwhm = scherrer_broadening_edxrd(self.d_hkl, e_kev, mean_ds_nm);
        (e_kev as f32, peak_weight as f32, fwhm as f32)
    }
}

pub fn render_jobs<T>(
    jobs: Vec<T>,
    two_thetas: &[f32],
    #[allow(unused)] atol: f32,
    n_phases: usize,
    with_weight_fractions: bool,
) -> Result<(Array2<f32>, Vec<PatternMeta>), String>
where
    T: Discretizer + Send + Sync + 'static,
{
    let n = jobs.len();
    let j = jobs.first().expect("at least one job");

    // NOTE: currently, we are only able to render one kind of background per simulation
    // should this ever change, we need to adapt this implementation
    let n_bkg_params = j.bkg().bkg_coefs();
    let mut metadata = T::init_meta_data(jobs.len(), n_phases, with_weight_fractions, n_bkg_params);
    for (i, job) in jobs.iter().enumerate() {
        for m in metadata.iter_mut() {
            job.write_meta_data(m, i)
        }
    }

    // actual rendering of the patterns
    cfg_if! {
        if #[cfg(feature = "cpu-only")] {
            let mut intensities = Array2::<f32>::zeros((n, two_thetas.len()));
            for (mut pattern, job) in intensities.outer_iter_mut().zip(jobs) {
                job.discretize_into(pattern.as_slice_mut().unwrap(), &two_thetas, atol);
            }
        } else {
            use crate::discretize_cuda::discretize_peaks_cuda;
            let intensities = discretize_peaks_cuda(jobs, two_thetas)?;
            let intensities = ndarray::Array2::from_shape_vec((n, two_thetas.len()), intensities)
                .expect("sizes must match");
        }
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
