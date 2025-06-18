use cfg_if::cfg_if;
use log::{error, warn};
use nalgebra::Vector3;
use ndarray::Array2;
use rand::Rng;

use crate::background::Background;
use crate::io::PatternMeta;
use crate::math::{
    caglioti, e_kev_to_lambda_ams, pseudo_voigt, sample_displacement_delta_two_theta_rad,
    scherrer_broadening, scherrer_broadening_edxrd, C_M_S, H_EV_S,
};
use crate::preferred_orientation::MarchDollase;
use crate::structure::Strain;

pub use self::adxrd::{ADXRDMeta, DiscretizeAngleDisperse};
use self::edxrd::Beamline;

use crate::cfg::VolumeFraction;

pub mod adxrd;
pub mod edxrd;

pub struct RenderCommon<'a> {
    // all simulated peaks for all phases in order [structure, structure permutations]
    pub all_simulated_peaks: &'a Box<[Box<[Peaks]>]>,
    pub all_preferred_orientations: &'a Box<[Box<[Option<MarchDollase>]>]>,
    pub all_strains: &'a Box<[Box<[Strain]>]>,
    // indices to select from simulated peaks, length is number of structures
    pub indices: Box<[usize]>,
    pub impurity_peaks: Box<[ImpurityPeak]>,
}

pub struct VFGenerator<'a> {
    pub fraction_sum: f64,
    pub n_free: usize,
    pub fractions: &'a Vec<Option<VolumeFraction>>,
}

impl<'a> VFGenerator<'a> {
    pub fn try_new(fractions: &'a Vec<Option<VolumeFraction>>) -> Result<Self, ()> {
        let fraction_sum = fractions.iter().filter_map(|x| x.map(|x| x.0)).sum::<f64>();
        let n_free = fractions.iter().filter(|x| x.is_none()).count();

        if fraction_sum > 1.0 {
            error!("Could not create volume fraction generator. Specified fractions need to sum to less than or equal to 1.0. Got sum: {}", fraction_sum);
            return Err(());
        }

        if fraction_sum > 0.99 && n_free > 0 {
            warn!("Fraction sum ({fraction_sum:.3}) is close to 1. There are {n_free} non-fixed volume fractions which are strongly constrained because of this.");
        }

        if n_free == 0 && fraction_sum < 1.0 - 1e-5 {
            // no free parameters and fraction sum smaller than 1
            error!("All structures volume fractions are fixed, but the sum of their fractions is smaller than one (delta: {d:.2e}). Make sure that the volume fractions add up to 1, or remove the specification for one fraction if you want to compute it automatically.", d = 1.0 - fraction_sum);
            return Err(());
        }

        Ok(Self {
            n_free,
            fraction_sum,
            fractions,
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
        let mut concentration_buf = Vec::with_capacity(self.fractions.len() + 1);
        if self.n_free > 0 {
            concentration_buf.push(0.0);
            concentration_buf.extend(
                (0..self.n_free - 1).map(|_| rng.random_range(0.0..=1.0 - self.fraction_sum)),
            );
            concentration_buf.push(1.0 - self.fraction_sum);

            concentration_buf[1..self.n_free]
                .sort_unstable_by(|a, b| a.partial_cmp(b).expect("not nan"));

            // compute the difference
            for i in 0..concentration_buf.len() - 1 {
                concentration_buf[i] = concentration_buf[i + 1] - concentration_buf[i];
            }
        }
        concentration_buf.resize(concentration_buf.capacity(), 0.0);

        // place the fixed numbers at the correct positions
        if self.n_free < self.fractions.len() {
            let mut free_idx = self.n_free;
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
            }
        }

        concentration_buf.truncate(self.fractions.len());
        concentration_buf.into_boxed_slice()
    }
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
    fn normalize(&self) -> bool;

    fn write_meta_data(&self, key: &mut PatternMeta, pat_id: usize);
    fn init_meta_data(n_patterns: usize, n_phases: usize) -> Vec<PatternMeta>;

    fn discretize_into(&self, intensities: &mut [f32], positions: &[f32], abstol: f32) {
        for PeakRenderParams {
            pos,
            intensity,
            fwhm,
            eta,
        } in self.peak_info_iterator()
        {
            render_peak(pos, intensity, fwhm, eta, abstol, positions, intensities)
        }

        self.bkg().render(intensities, positions);

        if self.normalize() {
            // TODO: check for NaNs and normalization
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
    pub hkls: Vec<Vector3<i16>>,
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
    /// emission line's relative intensity)
    pub fn get_adxrd_render_params(
        &self,
        wavelength_nm: f64,
        u: f64,
        v: f64,
        w: f64,
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
        let f_lorentz = lorentz_factor(theta_hkl_rad);
        let fwhm = caglioti(u, v, w, theta_hkl_rad)
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
    /// emission line's relative intensity)
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
        let peak_weight = self.i_hkl
            * f_lorentz
            * e_kev_to_lambda_ams(e_kev).powi(3)
            * beamline_intensity
            * weight;

        let fwhm = scherrer_broadening_edxrd(self.d_hkl, e_kev, mean_ds_nm);
        (e_kev as f32, peak_weight as f32, fwhm as f32)
    }
}

fn lorentz_factor(theta_rad: f64) -> f64 {
    (1.0 + (2.0 * theta_rad).cos().powi(2)) / (theta_rad.sin().powi(2) * theta_rad.cos())
}

pub fn render_jobs<T>(
    jobs: &[T],
    two_thetas: &[f32],
    #[allow(unused)] atol: f32,
    n_phases: usize,
) -> (Array2<f32>, Vec<PatternMeta>)
where
    T: Discretizer,
{
    let n = jobs.len();
    // actual rendering of the patterns
    cfg_if! {
        if #[cfg(feature = "cpu-only")] {
            let mut intensities = Array2::<f32>::zeros((n, two_thetas.len()));
            for (mut pattern, job) in intensities.outer_iter_mut().zip(jobs) {
                job.discretize_into(pattern.as_slice_mut().unwrap(), &two_thetas, atol);
            }
        } else {
            use crate::discretize_cuda::discretize_peaks_cuda;
            let intensities = discretize_peaks_cuda(jobs, &two_thetas);
            let intensities = ndarray::Array2::from_shape_vec((n, two_thetas.len()), intensities)
                .expect("sizes must match");
        }
    };

    let mut metadata = T::init_meta_data(jobs.len(), n_phases);

    for (i, job) in jobs.iter().enumerate() {
        for m in metadata.iter_mut() {
            job.write_meta_data(m, i)
        }
    }

    (intensities, metadata)
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
        let gen = VFGenerator::try_new(&vfs).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let generated = gen.generate(&mut rng);
        assert_eq!(generated.len(), n);
        assert_eq!(generated.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn vf_generation_single_fixed() {
        let vfs = vec![None, None, Some(VolumeFraction(0.3))];
        let n = vfs.len();
        let gen = VFGenerator::try_new(&vfs).unwrap();
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
        let gen = VFGenerator::try_new(&vfs).unwrap();
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
        let gen = VFGenerator::try_new(&vfs).unwrap();
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
        let gen = VFGenerator::try_new(&vfs).unwrap();
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
        let gen = VFGenerator::try_new(&vfs).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let generated = gen.generate(&mut rng);
        assert_eq!(generated[0], 0.6);
        assert_eq!(generated[1], 0.4);
        assert_eq!(generated.len(), n);
        assert_eq!(generated.iter().sum::<f64>(), 1.0);
    }
}
