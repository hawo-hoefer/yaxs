use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::Arc;

use itertools::Itertools;
use log::{debug, info};
use ordered_float::NotNan;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::cfg::{
    apply_strain_cfg, CompactSimResults, POGenerator, SampleParameters, StrainCfg,
    TextureMeasurement, ToDiscretize,
};
use crate::lattice::Lattice;
use crate::math::linalg::Vec3;
use crate::pattern::Peaks;
use crate::preferred_orientation::{BinghamParams, KDEBinghamODF};
use crate::scatter::Scatter;
use crate::species::Atom;
use crate::strain::Strain;
use crate::structure::Structure;
use crate::uninit_vec;

enum PossiblyTextureMeasurementPeaks {
    NoTexture(Peaks),
    Texture(Vec<Peaks>),
}

pub enum Alignment<'a> {
    Raw {
        po: &'a KDEBinghamODF,
        phi: f64,
        chi: f64,
    },
    Precomputed {
        po: &'a KDEBinghamODF,
    },
}

impl<'a> Alignment<'a> {
    pub fn weight(&self, pos: &Vec3<f64>) -> NotNan<f64> {
        match self {
            Alignment::Raw { po, phi, chi } => {
                NotNan::new(po.weight(&pos, *chi, *phi)).expect("weight is not nan")
            }
            Alignment::Precomputed { po } => {
                NotNan::new(po.weight_aligned(&pos)).expect("weight is not NaN")
            }
        }
    }
}

struct PeakSimResult {
    strain: Strain,
    po: Option<KDEBinghamODF>,
    peaks: PossiblyTextureMeasurementPeaks,
    struct_id: usize,
    permutation_id: usize,
}

struct WriteCtx {
    inner: UnsafeCell<Inner>,
}

struct Inner {
    strain: Vec<Strain>,
    pos: Vec<Option<BinghamParams>>,
    peaks: Vec<MaybeUninit<Peaks>>,
    ok: Vec<bool>,
    n_measurements: usize,
    n_permutations: usize,
}

unsafe impl Sync for WriteCtx {}

impl WriteCtx {
    pub fn new(n_structs: usize, n_measurements: usize, n_permutations: usize) -> Self {
        let n_peak_sets = n_structs * n_permutations * n_measurements;
        let n_simulations = n_structs * n_permutations;

        Self {
            inner: UnsafeCell::new(Inner {
                peaks: unsafe { uninit_vec(n_peak_sets) },
                strain: unsafe { uninit_vec(n_simulations) },
                pos: unsafe { uninit_vec(n_simulations) },
                ok: unsafe { uninit_vec(n_simulations) },
                n_permutations,
                n_measurements,
            }),
        }
    }

    pub unsafe fn add(&self, p: PeakSimResult) {
        let Inner {
            strain,
            pos,
            peaks,
            ok,
            n_permutations,
            n_measurements,
        } = unsafe { &mut *(self.inner.get()) };

        let sample_idx = p.struct_id * *n_permutations + p.permutation_id;
        match p.peaks {
            PossiblyTextureMeasurementPeaks::NoTexture(res) => {
                assert_eq!(*n_measurements, 1, "n_measurements needs to be 1 if no texture measurement is done. this is likely a bug in yaxs.");

                pos[sample_idx] = p.po.map(|x| x.params);
                strain[sample_idx] = p.strain;
                ok[sample_idx] = true;

                peaks[sample_idx] = MaybeUninit::new(res);
            }
            PossiblyTextureMeasurementPeaks::Texture(mut texture_measurement_peaks) => {
                assert_eq!(texture_measurement_peaks.len(), *n_measurements, "number of peak sets must match number of simulated peaks. this is likely a bug in yaxs.");
                let sample_idx = p.struct_id * *n_permutations + p.permutation_id;
                // index is [structure_id, permutation_id, texture_measurment_id]
                for (measurement_id, res) in texture_measurement_peaks.drain(..).enumerate() {
                    let idx = sample_idx * *n_measurements + measurement_id;

                    peaks[idx] = MaybeUninit::new(res);
                }

                pos[sample_idx] = p.po.as_ref().map(|x| x.params.clone());
                strain[sample_idx] = p.strain.clone();
                ok[sample_idx] = true;
            }
        }
    }

    pub fn make_to_discretize(
        self,
        structures: Arc<[Structure]>,
        sample_parameters: SampleParameters,
        texture_measurement: Option<TextureMeasurement>,
    ) -> ToDiscretize {
        let Inner {
            strain,
            pos,
            mut peaks,
            ok,
            n_permutations,
            n_measurements: _,
        } = self.inner.into_inner();

        // TODO: make this return a result instead so we can error gracefully
        assert!(ok.iter().all(|x| *x));

        ToDiscretize {
            structures,
            sample_parameters,
            sim_res: Arc::new(CompactSimResults {
                all_simulated_peaks: peaks
                    .drain(..)
                    .map(|x| unsafe { x.assume_init() })
                    .collect(),
                all_strains: strain.into(),
                all_preferred_orientations: pos.into(),
                n_permutations,
                texture_measurement,
            }),
        }
    }
}

/// Generate Peaks for the input structures and their physical parameters.
///
/// * `sample_params`: physical parameter ranges for the structures
/// * `structures`: structures to simulate peaks for
/// * `rng`: random number generator to use
///
/// * `sample_params`: the user specified sample parameters
/// * `structures`: the structures to simulate
/// * `structure_po_configs`: preferred orientation configuration for all structures
/// * `structure_strain_configs`: strain configurations for each of the structures
/// * `structure_files`: file paths to the structure's cifs
/// * `rng`: rng to use
pub fn simulate_peaks(
    (min_r, max_r): (f64, f64),
    sample_parameters: SampleParameters,
    structures: Box<[Structure]>,
    structure_po_configs: Box<[Option<POGenerator>]>,
    structure_strain_configs: Box<[Option<StrainCfg>]>,
    structure_files: Box<[String]>,
    texture_measurement: Option<TextureMeasurement>,
    rng: &mut impl Rng,
) -> Result<ToDiscretize, String> {
    struct PeakSim {
        structure: usize,
        permutation: usize,
        seed: u64,
        min_r: f64,
        max_r: f64,
        t: Option<TextureMeasurement>,
    }

    #[derive(Clone)]
    struct RunCtx {
        structs: Arc<[Structure]>,
        po_gens: Box<[Option<POGenerator>]>,
        strain_cfgs: Box<[Option<StrainCfg>]>,
        structure_files: Box<[String]>,
    }

    impl RunCtx {
        fn run(
            &mut self,
            job: PeakSim,
            scattering_parameters: &HashMap<Atom, Scatter>,
        ) -> Result<PeakSimResult, String> {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(job.seed);
            let Some((perm_s, strain)) = apply_strain_cfg(
                &self.strain_cfgs[job.structure],
                &self.structs[job.structure],
                &mut rng,
            ) else {
                return Err(format!("Could not apply strain to structure '{file}'. Strain matrix is not invertible. Please check the strain configuration.", file=self.structure_files[job.structure]));
            };
            let po = self.po_gens[job.structure]
                .as_mut()
                .map(|x| x.sample(&mut rng));

            let peaks = if let Some(t) = job.t {
                let hkls_intensities_spacings = perm_s
                    .get_hkl_intensities_spacings(job.min_r, job.max_r, scattering_parameters)
                    .into_boxed_slice();
                let mut peaks = Vec::new();
                for (_, (chi, phi)) in t
                    .chi
                    .into_iter()
                    .cartesian_product(t.phi.into_iter())
                    .enumerate()
                {
                    let transformed_po = po.as_ref().map(|x| x.with_orientation(chi, phi));
                    let p = perm_s.apply_alignment_to_hkls_intensities(
                        &hkls_intensities_spacings,
                        transformed_po
                            .as_ref()
                            .map(|x| Alignment::Precomputed { po: x }),
                    );
                    peaks.push(p.into_boxed_slice());
                }
                PossiblyTextureMeasurementPeaks::Texture(peaks)
            } else {
                let peaks = perm_s
                    .get_d_spacings_intensities(job.min_r, job.max_r, None, scattering_parameters)
                    .into_boxed_slice();
                PossiblyTextureMeasurementPeaks::NoTexture(peaks)
            };

            Ok(PeakSimResult {
                strain,
                po: po.clone(),
                peaks,
                struct_id: job.structure,
                permutation_id: job.permutation,
            })
        }
    }

    let n_structs = structures.len();
    let n_permutations = sample_parameters.structure_permutations;
    let n_texture_measurements = texture_measurement.map(|t| t.stride()).unwrap_or(1);

    enum Task {
        Job(PeakSim),
        Stop,
    }

    let mut n_threads: usize = std::thread::available_parallelism()
        .map(|x| x.into())
        .unwrap_or(1);

    if n_structs * n_permutations < 50 {
        info!("Small number of simulations. Using single-threaded mode.");
        n_threads = 1;
    }

    let mut scattering_parameters = HashMap::new();
    for s in structures.iter() {
        s.gather_scattering_params(&mut scattering_parameters);
    }

    let mut ctx = RunCtx {
        structs: structures.into(),
        po_gens: structure_po_configs,
        strain_cfgs: structure_strain_configs,
        structure_files,
    };

    let results = Arc::new(WriteCtx::new(
        n_structs,
        n_texture_measurements,
        n_permutations,
    ));

    if n_threads == 1 {
        info!("Running single-threaded peak simulation");

        for (struct_id, permutation_id) in (0..n_structs).cartesian_product(0..n_permutations) {
            let job = PeakSim {
                structure: struct_id,
                permutation: permutation_id,
                seed: rng.random(),
                min_r,
                max_r,
                t: texture_measurement,
            };
            let p = ctx.run(job, &scattering_parameters)?;
            unsafe { results.add(p) };
        }
    } else {
        let (job_sender, job_receiver) = crossbeam_channel::unbounded();
        info!("Launching {n_threads} threads for peak simulation");

        let handles = (0..n_threads)
            .map(|i| {
                let results = Arc::clone(&results);
                let job_receiver = job_receiver.clone();

                let mut ctx = ctx.clone();
                let scattering_parameters = scattering_parameters.clone();

                for po_gen in ctx.po_gens.iter_mut().filter_map(|x| x.as_mut()) {
                    // just to pull n_thinning samples out of the po_gen internal Hit and Run sampler
                    // this is hopefully enough to make the samplers of every thread independent
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng.random());
                    _ = po_gen.sample(&mut rng);
                }

                std::thread::spawn(move || -> Result<(), String> {
                    loop {
                        let job: PeakSim = match job_receiver.recv() {
                            Ok(Task::Stop) => break,
                            Ok(Task::Job(v)) => v,
                            Err(_) => break,
                        };

                        let p = ctx
                            .run(job, &scattering_parameters)
                            .map_err(|err| format!("Peak simulation thread {i}: {err}"))?;
                        unsafe {
                            results.add(p);
                        }
                    }
                    debug!("Peak simulation thread {i} finished.");
                    Ok(())
                })
            })
            .collect_vec();

        for (struct_id, permutation_id) in
            (0..n_structs).cartesian_product(0..sample_parameters.structure_permutations)
        {
            let job = PeakSim {
                structure: struct_id,
                permutation: permutation_id,
                seed: rng.random(),
                min_r,
                max_r,
                t: texture_measurement,
            };
            let _ = job_sender.send(Task::Job(job));
        }

        debug!("Sending stop signal to peak simulation threads");
        for _ in 0..n_threads {
            job_sender
                .send(Task::Stop)
                .map_err(|err| format!("Could not send stop signal for peak simulation: '{err}'"))?
        }

        for handle in handles {
            handle.join().map_err(|err| {
                format!("Could not join peak simulation thread: '{err:?}'. Exiting...")
            })??;
        }

        debug!("All simulation threads joined")
    }

    let results = Arc::into_inner(results).expect("no more references to write context");
    Ok(results.make_to_discretize(ctx.structs, sample_parameters, texture_measurement))
}
