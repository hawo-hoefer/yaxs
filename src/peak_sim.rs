use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::Arc;

use itertools::Itertools;
use log::{debug, info};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::cfg::{apply_strain_cfg, CompactSimResults, MarchDollaseCfg, SampleParameters, StrainCfg, ToDiscretize};
use crate::pattern::Peaks;
use crate::preferred_orientation::MarchDollase;
use crate::scatter::Scatter;
use crate::species::Atom;
use crate::strain::Strain;
use crate::structure::Structure;
use crate::uninit_vec;

struct PeakSimResult {
    strain: Strain,
    po: Option<MarchDollase>,
    peaks: Peaks,
    struct_id: usize,
    permutation_id: usize,
}

struct WriteCtx {
    inner: UnsafeCell<Inner>,
}

struct Inner {
    strain: Vec<Strain>,
    pos: Vec<Option<MarchDollase>>,
    peaks: Vec<MaybeUninit<Peaks>>,
    ok: Vec<bool>,
    n_permutations: usize,
}

unsafe impl Sync for WriteCtx {}

impl WriteCtx {
    pub fn new(n_structs: usize, n_permutations: usize) -> Self {
        Self {
            inner: UnsafeCell::new(Inner {
                strain: unsafe { uninit_vec(n_structs * n_permutations) },
                pos: unsafe { uninit_vec(n_structs * n_permutations) },
                peaks: unsafe { uninit_vec(n_structs * n_permutations) },
                ok: unsafe { uninit_vec(n_structs * n_permutations) },
                n_permutations,
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
        } = unsafe { &mut *(self.inner.get()) };

        let idx = p.struct_id * (*n_permutations) + p.permutation_id;

        strain[idx] = p.strain;
        pos[idx] = p.po;
        peaks[idx] = MaybeUninit::new(p.peaks);
        ok[idx] = true
    }

    pub fn make_to_discretize(
        self,
        structures: Arc<[Structure]>,
        sample_parameters: SampleParameters,
    ) -> ToDiscretize {
        let Inner {
            strain,
            pos,
            mut peaks,
            ok,
            n_permutations,
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
    structure_po_configs: Box<[Option<MarchDollaseCfg>]>,
    structure_strain_configs: Box<[Option<StrainCfg>]>,
    structure_files: Box<[String]>,
    rng: &mut impl Rng,
) -> Result<ToDiscretize, String> {
    struct PeakSim {
        structure: usize,
        permutation: usize,
        seed: u64,
        min_r: f64,
        max_r: f64,
    }

    struct RunCtx {
        structs: Arc<[Structure]>,
        po_cfgs: Box<[Option<MarchDollaseCfg>]>,
        strain_cfgs: Box<[Option<StrainCfg>]>,
        structure_files: Box<[String]>,
    }

    impl RunCtx {
        fn run(
            &self,
            job: PeakSim,
            scattering_param_cache: &HashMap<Atom, Scatter>,
        ) -> Result<PeakSimResult, String> {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(job.seed);
            let Some((perm_s, strain)) = apply_strain_cfg(
                &self.strain_cfgs[job.structure],
                &self.structs[job.structure],
                &mut rng,
            ) else {
                return Err(format!("Could not apply strain to structure '{file}'. Strain matrix is not invertible. Please check the strain configuration.", file=self.structure_files[job.structure]));
            };
            let po_cfg = &self.po_cfgs[job.structure];
            let po = po_cfg.as_ref().map(|cfg| cfg.generate(&mut rng));

            let peaks = perm_s
                .get_d_spacings_intensities(
                    job.min_r,
                    job.max_r,
                    po.as_ref(),
                    scattering_param_cache,
                )
                .into_boxed_slice();

            Ok(PeakSimResult {
                strain,
                po,
                peaks,
                struct_id: job.structure,
                permutation_id: job.permutation,
            })
        }
    }

    let n_structs = structures.len();
    let n_permutations = sample_parameters.structure_permutations;

    let mut scattering_parameters = HashMap::<Atom, Scatter>::new();
    for struct_id in 0..n_structs {
        structures[struct_id].gather_scattering_params(&mut scattering_parameters);
    }

    enum Task {
        Job(PeakSim),
        Stop,
    }

    let ctx = Arc::new(RunCtx {
        structs: structures.into(),
        po_cfgs: structure_po_configs,
        strain_cfgs: structure_strain_configs,
        structure_files,
    });

    let results = Arc::new(WriteCtx::new(n_structs, n_permutations));

    let mut n_threads: usize = std::thread::available_parallelism()
        .map(|x| x.into())
        .unwrap_or(1);

    if n_structs * n_permutations < 50 {
        info!("Small number of simulations. Using single-threaded mode.");
        n_threads = 1;
    }

    if n_threads == 1 {
        info!("Running single-threaded peak simulation");
        for (struct_id, permutation_id) in (0..n_structs).cartesian_product(0..n_permutations) {
            let job = PeakSim {
                structure: struct_id,
                permutation: permutation_id,
                seed: rng.random(),
                min_r,
                max_r,
            };
            let p = ctx.run(job, &scattering_parameters)?;
            unsafe { results.add(p) };
        }
    } else {
        let (job_sender, job_receiver) = crossbeam_channel::unbounded();
        info!("Launching {n_threads} threads for peak simulation");

        let handles = (0..n_threads)
            .map(|i| {
                let ctx = Arc::clone(&ctx);
                let results = Arc::clone(&results);
                let job_receiver = job_receiver.clone();
                let scattering_parameters = scattering_parameters.clone();
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
    let ctx = Arc::into_inner(ctx).expect("no more references to simulation ctx");
    Ok(results.make_to_discretize(ctx.structs, sample_parameters))
}

