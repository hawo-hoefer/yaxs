use std::sync::Arc;

use log::error;
use nalgebra::Vector3;
use ndarray::{Array1, Array2, Array3};

use crate::background::Background;
use crate::discretize_cuda::discretize_peaks_cuda;
use crate::io::{render_angle_disperse_and_queue_write_in_thread, PatternMetaData, WriteJob};
use crate::math::{
    caglioti, e_kev_to_lambda_ams, pseudo_voigt, scherrer_broadening, scherrer_broadening_edxrd,
    C_M_S, H_EV_S,
};
use crate::structure::Strain;

#[derive(Clone, Debug, PartialEq)]
pub struct PatternMeta {
    pub vol_fractions: Box<[f64]>,
    pub mean_ds_nm: Box<[f64]>,
    pub eta: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub background: Background,
}

#[derive(serde::Deserialize, serde::Serialize, PartialEq, Debug, Clone)]
#[repr(C)]
pub struct EmissionLine {
    // wavelength in amstrong
    pub wavelength_ams: f64,
    // wavelength relative weight
    pub weight: f64,
}

impl EmissionLine {
    /// create a new emission line from wavelength and weight
    ///
    /// * `wavelength`: wavelength in amstrong
    /// * `weight`: intensity of the emission line relative to other emission lines in the spectrum
    pub fn new(wavelength: f64, weight: f64) -> Self {
        Self {
            wavelength_ams: wavelength,
            weight,
        }
    }
}

pub type Peaks = Box<[Peak]>;

pub struct DiscretizeAngleDisperse<'a> {
    // all simulated peaks for all phases in order [structure, structure permutations]
    pub all_simulated_peaks: &'a Vec<Vec<Peaks>>,
    pub all_strains: &'a Vec<Vec<Strain>>,
    // indices to select from simulated peaks, length is number of structures
    pub indices: Vec<usize>,
    pub emission_lines: &'a [EmissionLine],
    pub normalize: bool,
    pub meta: PatternMeta,
}

impl<'a> DiscretizeAngleDisperse<'a> {
    pub fn discretize_into(&self, pat: &mut [f32], two_thetas: &[f32], abstol: f32) {
        let PatternMeta {
            vol_fractions,
            eta,
            mean_ds_nm,
            u,
            v,
            w,
            background,
        } = &self.meta;

        for (((phase_peaks, idx), vf), phase_mean_ds_nm) in self
            .all_simulated_peaks
            .iter()
            .zip(&self.indices)
            .zip(vol_fractions)
            .zip(mean_ds_nm)
        {
            let peaks = &phase_peaks[*idx];
            // * `pat`: target pattern
            // * `two_thetas`: two theta values of pattern's intensities in degrees
            // * `wavelength`: wavelength of the x-rays in nanometers
            // * `mean_ds`: mean domain size used for scherrer broadening
            // * `u`: caglioti parameter u
            // * `v`: caglioti parameter v
            // * `w`: caglioti parameter w
            for emission_line in self.emission_lines {
                let wavelength_nm = emission_line.wavelength_ams / 10.0;
                for peak in peaks.iter() {
                    let (two_theta_hkl_deg, peak_weight, fwhm) = peak.get_adxrd_render_params(
                        wavelength_nm,
                        *u,
                        *v,
                        *w,
                        *phase_mean_ds_nm,
                        emission_line.weight * vf,
                    );
                    render_peak(
                        two_theta_hkl_deg,
                        peak_weight,
                        fwhm,
                        *eta as f32,
                        abstol,
                        two_thetas,
                        pat,
                    )
                }
            }
        }
        background.render(pat, two_thetas);

        if self.normalize {
            // TODO: check for NaNs and normalization
            let f = *pat.first().unwrap();
            let vmin = pat.iter().fold(f, |a, b| f32::min(a, *b));
            let vmax = pat.iter().fold(f, |a, b| f32::max(a, *b));
            pat.iter_mut().for_each(|x| {
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
        _mean_ds_nm: f64,
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

        let fwhm = scherrer_broadening_edxrd(self.d_hkl, e_kev, 100.0);
        (e_kev as f32, peak_weight as f32, fwhm as f32)
    }
}

fn lorentz_factor(theta_rad: f64) -> f64 {
    (1.0 + (2.0 * theta_rad).cos().powi(2)) / (theta_rad.sin().powi(2) * theta_rad.cos())
}

pub fn render_write_chunked(
    jobs: &[DiscretizeAngleDisperse],
    two_thetas: &[f32],
    abstol: f32,
    n_phases: usize,
    io_opts: &crate::io::Opts,
) -> Vec<String> {
    let (tx, rx) = std::sync::mpsc::channel::<Arc<WriteJob<_>>>();
    let compress = io_opts.compress;
    let io_thread_handle = std::thread::spawn(move || loop {
        match rx.recv() {
            Ok(v) => match *v {
                WriteJob::Done => return,
                WriteJob::Write {
                    ref intensities,
                    ref path,
                    ref meta,
                } => match crate::io::write_to_npz(path, intensities, meta, compress) {
                    Err(()) => {
                        error!("IO thread quitting...");
                        return;
                    }
                    Ok(()) => (),
                },
            },
            Err(err) => {
                error!("IO thread: Error receiving from channel: {err}. Quitting...");
                return;
            }
        }
    });

    let mut i = 0;
    let l = jobs.len();
    let chunk_size = io_opts.chunk_size.unwrap_or(l);
    let n_chunks = l / chunk_size + (l % chunk_size > 0) as usize;
    let pad_width = if n_chunks > 1 {
        1 + (n_chunks - 1).ilog10()
    } else {
        1
    };

    let mut datafiles = Vec::new();
    while i < l {
        let chunk_file_name = format!(
            "data_{:0width$}.npz",
            i / chunk_size,
            width = pad_width as usize
        );
        let chunk: &[DiscretizeAngleDisperse] = &jobs[i..(i + chunk_size).min(l)];

        let mut chunk_path = std::path::PathBuf::new();
        chunk_path.push(&io_opts.output_name);
        chunk_path.push(&chunk_file_name);
        datafiles.push(chunk_file_name);

        let _ = render_angle_disperse_and_queue_write_in_thread(
            &chunk,
            &two_thetas,
            chunk_path,
            tx.clone(),
            abstol,
            n_phases,
        )
        .map_err(|_| {
            std::process::exit(1);
        });
        i += chunk_size;
    }
    let _ = tx.send(Arc::new(WriteJob::Done));
    io_thread_handle.join().unwrap_or_else(|err| {
        error!("Could not join io thread: {err:?}");
        std::process::exit(1)
    });
    datafiles
}

pub fn render_jobs(
    jobs: &[DiscretizeAngleDisperse],
    two_thetas: &[f32],
    atol: f32,
    n_phases: usize,
) -> (Array2<f32>, PatternMetaData) {
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
