use std::sync::Arc;

use ndarray::Array2;

use crate::background::Background;
use crate::cfg::Config;
use crate::discretize_cuda::discretize_peaks_cuda;
use crate::io::{render_jobs_to_npz, WriteJob};
use crate::math::{caglioti, pseudo_voigt, scherrer_broadening};

#[derive(Clone, Debug, PartialEq)]
pub struct PatternMeta {
    pub vol_fractions: Box<[f64]>,
    pub eta: f64,
    pub mean_ds_nm: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub background: Background,
}

#[derive(serde::Deserialize, serde::Serialize, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct Peaks {
    pub peaks: Box<[Peak]>,
    pub wavelength_nm: f64,
}

pub struct DiscretizationJob<'a> {
    // all simulated peaks for all phases in order [structure, structure permutations]
    pub all_simulated_peaks: &'a Vec<Vec<Peaks>>,
    // indices to select from simulated peaks, length is number of structures
    pub indices: Vec<usize>,
    pub emission_lines: &'a [EmissionLine],
    pub normalize: bool,
    pub meta: PatternMeta,
}

impl<'a> DiscretizationJob<'a> {
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

        for ((phase_peaks, idx), vf) in self
            .all_simulated_peaks
            .iter()
            .zip(&self.indices)
            .zip(vol_fractions)
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
                for peak in &peaks.peaks {
                    let cpeak =
                        peak.convert(peaks.wavelength_nm, emission_line.wavelength_ams / 10.0);
                    cpeak.render(
                        pat,
                        two_thetas,
                        emission_line.wavelength_ams / 10.0,
                        emission_line.weight * vf,
                        *mean_ds_nm,
                        *eta,
                        *u,
                        *v,
                        *w,
                        abstol,
                    );
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

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Peak {
    // position in degrees two-theta
    pub pos: f64,
    pub intensity: f64,
}

impl Peak {
    /// Render the peak into an XRD pattern
    ///
    /// * `pat`: target pattern
    /// * `two_thetas`: two theta values of Pattern's intensities in degrees
    /// * `wavelength`: wavelength of the x-rays in nanometers
    /// * `weight`: weight of the emission line's wavelength
    /// * `mean_ds_nm`: mean domain size used for scherrer broadening (in nm)
    /// * `u`: caglioti parameter u
    /// * `v`: caglioti parameter v
    /// * `w`: caglioti parameter w
    pub fn render(
        &self,
        pat: &mut [f32],
        two_thetas: &[f32],
        wavelength: f64,
        weight: f64,
        mean_ds_nm: f64,
        eta: f64,
        u: f64,
        v: f64,
        w: f64,
        abstol: f32,
    ) {
        let pos = self.pos as f32;
        // TODO: make position in radians
        let theta_pos_rad = self.pos.to_radians() / 2.0;
        let fwhm = caglioti(u, v, w, theta_pos_rad)
            + scherrer_broadening(wavelength, theta_pos_rad, mean_ds_nm);
        let peak_weight = (weight * self.intensity) as f32;
        let midpoint = ((pos as f32 - two_thetas[0])
            / ((two_thetas[two_thetas.len() - 1] - two_thetas[0]) * two_thetas.len() as f32))
            as usize;

        let mut i = midpoint;
        if i > two_thetas.len() - 1 {
            i = two_thetas.len() - 1
        }

        // left half
        loop {
            let two_theta = two_thetas[i];
            let dx = two_theta - pos as f32;
            let di = peak_weight * pseudo_voigt(dx, eta as f32, fwhm as f32);
            if di < abstol {
                break;
            }
            pat[i] += di;
            if i == 0 {
                break;
            }
            i -= 1;
        }

        // right half
        i = midpoint + 1;
        while i < two_thetas.len() {
            let two_theta = two_thetas[i];
            let dx = two_theta - pos;
            let di = peak_weight * pseudo_voigt(dx, eta as f32, fwhm as f32);
            if di < abstol {
                break;
            }
            pat[i] += di;
            i += 1;
        }

        // for (intensity, two_theta) in pat.iter_mut().zip(two_thetas) {
        //     let dx = *two_theta - self.pos;
        //     *intensity += peak_weight * pseudo_voigt(dx, eta, fwhm);
        // }
    }

    pub fn convert(&self, from_wavelength_nm: f64, to_wavelength_nm: f64) -> Peak {
        let new_pos = 2.0
            * ((self.pos / 2.0).to_radians().sin() * from_wavelength_nm / to_wavelength_nm)
                .asin()
                .to_degrees();
        let old_wav_scaling = from_wavelength_nm.powi(3);
        let new_wav_scaling = to_wavelength_nm.powi(3);

        let old_lorentz = lorentz_factor((self.pos / 2.0).to_radians());
        let new_lorentz = lorentz_factor((new_pos / 2.0).to_radians());

        let wav_correction = new_wav_scaling / old_wav_scaling;
        let lorentz_correction = new_lorentz / old_lorentz;

        return Peak {
            pos: new_pos,
            intensity: self.intensity * wav_correction as f64 * lorentz_correction as f64,
        };
    }
}

fn lorentz_factor(theta_rad: f64) -> f64 {
    (1.0 + (2.0 * theta_rad).cos().powi(2)) / (theta_rad.sin().powi(2) * theta_rad.cos())
}

pub fn process_chunked(
    jobs: &[DiscretizationJob],
    two_thetas: &[f32],
    cfg: &Config,
    io_opts: &crate::io::Opts,
) {
    let (tx, rx) = std::sync::mpsc::channel::<Arc<WriteJob<_>>>();
    let compress = io_opts.compress;
    let io_thread_handle = std::thread::spawn(move || loop {
        match rx.recv() {
            Ok(v) => match *v {
                WriteJob::Done => return,
                WriteJob::Write {
                    ref intensities,
                    ref path,
                } => match crate::io::write_to_npz(path, intensities, compress) {
                    Err(()) => {
                        eprintln!("IO thread quitting...");
                        return;
                    }
                    Ok(()) => (),
                },
            },
            Err(err) => {
                eprintln!("IO thread: Error receiving from channel: {err}. Quitting...");
                return;
            }
        }
    });

    // prepare output directory
    match std::fs::DirBuilder::new().create(&io_opts.output_name) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && io_opts.overwrite => {
            eprintln!(
                "Removing '{out_dir}' according to user input...",
                out_dir = &io_opts.output_name
            );
            std::fs::remove_dir_all(&io_opts.output_name).unwrap_or_else(|err| {
                eprintln!(
                    "Could not remove output directory '{out_dir}': {err}",
                    out_dir = &io_opts.output_name
                );
                std::process::exit(1);
            });
            std::fs::create_dir(&io_opts.output_name).unwrap_or_else(|err| {
                eprintln!(
                    "Could not (re)create output directory '{out_dir}': {err}",
                    out_dir = io_opts.output_name
                );
                std::process::exit(1);
            });
        }
        Err(e) => {
            eprintln!(
                "Error creating output directory {out_dir}: {e:?}",
                out_dir = &io_opts.output_name
            );
            std::process::exit(1);
        }
        Ok(_) => {} // all good,
    }

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
        let chunk: &[DiscretizationJob] = &jobs[i..(i + chunk_size).min(l)];

        let mut chunk_path = std::path::PathBuf::new();
        chunk_path.push(&io_opts.output_name);
        chunk_path.push(&chunk_file_name);
        datafiles.push(chunk_file_name);

        let _ =
            render_jobs_to_npz(&chunk, &two_thetas, chunk_path, tx.clone(), cfg).map_err(|_| {
                std::process::exit(1);
            });
        i += chunk_size;
    }
    let _ = tx.send(Arc::new(WriteJob::Done));
    io_thread_handle.join().unwrap_or_else(|err| {
        eprintln!("Error joining io thread: {err:?}");
        std::process::exit(1)
    });
}

pub fn render_jobs(jobs: &[DiscretizationJob], two_thetas: &[f32], atol: f32) -> Array2<f32> {
    if cfg!(feature = "cpu-only") {
        let mut intensities = Array2::<f32>::zeros((jobs.len(), two_thetas.len()));
        for (mut pattern, job) in intensities.outer_iter_mut().zip(jobs) {
            job.discretize_into(pattern.as_slice_mut().unwrap(), &two_thetas, atol);
        }
        intensities
    } else {
        let intensities = discretize_peaks_cuda(&jobs, &two_thetas);
        ndarray::Array2::from_shape_vec((jobs.len(), two_thetas.len()), intensities)
            .expect("sizes must match")
    }
}
