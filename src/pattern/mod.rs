use nalgebra::Vector3;
use ndarray::{Array1, Array2, Array3};

use crate::discretize_cuda::discretize_peaks_cuda;
use crate::io::PatternMetaData;
use crate::math::{
    caglioti, e_kev_to_lambda_ams, pseudo_voigt, scherrer_broadening, scherrer_broadening_edxrd, C_M_S, H_EV_S
};

pub use self::adxrd::{ADXRDMeta, DiscretizeAngleDisperse};

pub mod adxrd;
pub mod edxrd;

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

        let two_theta_hkl_deg = 2.0 * theta_hkl_rad.to_degrees() as f32;
        (two_theta_hkl_deg, peak_weight, fwhm as f32)
    }

    /// Get EDXRD Peak location, intensity and fwhm
    ///
    /// * `wavelength_nm`: X-ray wavelength
    /// * `mean_ds_nm`: mean domain size in nanometers
    /// * `weight`: weight of the peak (usually something like volume fraction multiplied by the
    /// emission line's relative intensity)
    pub fn get_edxrd_render_params<F>(
        &self,
        theta_rad: f64,
        f_lorentz: f64,
        mean_ds_nm: f64,
        weight: f64,
        beamline_intensity: F,
    ) -> (f32, f32, f32)
    where
        F: Fn(f64) -> f64,
    {
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
        let peak_weight = self.i_hkl
            * f_lorentz
            * e_kev_to_lambda_ams(e_kev).powi(3)
            * beamline_intensity(e_kev)
            * weight;

        let fwhm = scherrer_broadening_edxrd(self.d_hkl, e_kev, mean_ds_nm);
        (e_kev as f32, peak_weight as f32, fwhm as f32)
    }
}

fn lorentz_factor(theta_rad: f64) -> f64 {
    (1.0 + (2.0 * theta_rad).cos().powi(2)) / (theta_rad.sin().powi(2) * theta_rad.cos())
}

pub fn render_jobs(
    jobs: &[DiscretizeAngleDisperse],
    two_thetas: &[f32],
    atol: f32,
    n_phases: usize,
) -> (Array2<f32>, PatternMetaData) {
    // actual rendering of the patterns
    let intensities = if cfg!(feature = "cpu-only") {
        let mut intensities = Array2::<f32>::zeros((jobs.len(), two_thetas.len()));
        for (mut pattern, job) in intensities.outer_iter_mut().zip(jobs) {
            job.discretize_into(pattern.as_slice_mut().unwrap(), &two_thetas, atol);
        }
        intensities
    } else {
        let intensities = discretize_peaks_cuda(&jobs, &two_thetas);
        ndarray::Array2::from_shape_vec((jobs.len(), two_thetas.len()), intensities)
            .expect("sizes must match")
    };

    // collect the pattern metadata
    let mut strains = Array3::<f32>::zeros((jobs.len(), n_phases, 6));
    let mut etas = Array1::<f32>::zeros(jobs.len());
    let mut caglioti_params = Array2::<f32>::zeros((jobs.len(), 3));
    let mut mean_ds_nm = Array2::<f32>::zeros((jobs.len(), n_phases));
    let mut volume_fractions = Array2::<f32>::zeros((jobs.len(), n_phases));

    for (i, (job, mut strain_loc)) in jobs.iter().zip(strains.outer_iter_mut()).enumerate() {
        etas[i] = job.meta.eta as f32;
        caglioti_params[(i, 0)] = job.meta.u as f32;
        caglioti_params[(i, 1)] = job.meta.v as f32;
        caglioti_params[(i, 2)] = job.meta.w as f32;

        for (j, cs) in job.meta.mean_ds_nm.iter().enumerate() {
            mean_ds_nm[(i, j)] = *cs as f32;
        }

        for (j, vf) in job.meta.vol_fractions.iter().enumerate() {
            volume_fractions[(i, j)] = *vf as f32;
        }

        for (phase_idx, (permutation_idx, strain_permutations_for_phase)) in
            job.indices.iter().zip(job.all_strains).enumerate()
        {
            for j in 0..6 {
                strain_loc[(phase_idx, j)] =
                    strain_permutations_for_phase[*permutation_idx].0[j] as f32;
            }
        }
    }

    (
        intensities,
        PatternMetaData {
            volume_fractions,
            strains,
            etas,
            mean_ds_nm,
            caglioti_params,
        },
    )
}
