use chrono::Utc;
use clap::Parser;
use itertools::Itertools;
use ordered_float::NotNan;
use rand::SeedableRng;
use sha2::Digest;
use std::path::PathBuf;
use std::time::{Instant, SystemTime};
use yaxs::absorption::MACGenerator;
use yaxs::domain_size::DomainSize;
use yaxs::math::pseudo_voigt;
use yaxs::pattern::adxrd::{InstrumentParameters, PrecomputedLACs};
use yaxs::pattern::{adxrd, edxrd, lorentz_polarization_factor_edxrd};
use yaxs::structure::Peak;

use log::{debug, error, info};

use yaxs::cfg::{prepare_peak_simulation, Config, SimulationKind, StructureDef, ToDiscretize};
use yaxs::io::{self, prepare_output_directory, HKLDisplayMode};
use yaxs::pattern::CompositionGenerator;

const ARTWORK: &'static str = r#"Running YAXS (YAXS: an Accelerated XRD Simulator)
          
                                        _,,,_
                   ----------       .-'`  (  '.
               -----------       .-'    ,_  ;  \___      _,
       --------              __.'    )   \'.__.'(:;'.__.'/
                     __..--""       (     '.__{':');}__.'
 ----------        .'         (    ;    (   .-|` '  |-.
     --------     /    (       )     )      '-p     q-'
                 (    ;     ;          ;    ; |.---.|
      ------     ) (              (      ;    \ o  o)
                 |  )     ;       |    )    ) /'.__/
       ----      )    ;  )    ;   | ;       //
        ------   ( )             _,\    ;  //
                 ; ( ,_,,-~""~`""   \ (   //
    -------       \_.'\\_            '.  /<_
                   \\_)--\             \ \--\
               jgs )--\""`             )--\"`
                   `""`                `""`

#### Yak Art by Joan Stark (Spunk) ####"#;

#[derive(Parser)]
#[command(
    version = env!("YAXS_VERSION"),
    about = "Simulate a dataset of XRD patterns from YAML config.",
    long_about = if cfg!(feature="use-gpu") {
        "YaXS simulates XRD patterns from CIF
GPU-accelerated build: CUDA-based peak rendering."
    } else {
        "YaXS simulates XRD patterns from CIF
CPU-only build: This may be much slower than the GPU-accelerated build."
    }
)]
struct Cli {
    #[arg(value_name = "FILE", help = "Configuration yaml file.")]
    cfg: PathBuf,

    #[command(flatten)]
    io: io::Opts,
}

fn display_hkls(
    to_discretize: ToDiscretize,
    cfg: Config,
    structure_paths: &[String],
    mode: HKLDisplayMode,
) -> Result<(), String> {
    info!("Displaying HKLs");

    for i in 0..to_discretize.structures.len() {
        let idx = to_discretize.sim_res.idx(i, 0, Some(0));
        let s = &to_discretize.sample_parameters.structures[i];
        let mean_ds_nm = s.domain_size.mean();
        let ds_eta = s.ds_eta.mean();
        let mustrain = s
            .mustrain
            .as_ref()
            .map(|x| x.amplitude.mean())
            .unwrap_or(0.0);
        let mustrain_eta = s.mustrain.as_ref().map(|x| x.eta.mean()).unwrap_or(0.0);

        info!("======= Structure {} =======", structure_paths[i]);
        let intensities_positions = match &cfg.kind {
            SimulationKind::AngleDispersive(ad) => {
                let structures = std::sync::Arc::as_ref(&to_discretize.structures);
                let wavelength_ams = ad.emission_lines[0].wavelength_ams;
                let abs = PrecomputedLACs::try_new(
                    std::iter::once(wavelength_ams),
                    structures,
                    structure_paths,
                )?;

                let instrument_parameters = ad
                    .instrument_parameters
                    .as_ref()
                    .map(|c| c.mean())
                    .unwrap_or(InstrumentParameters::zero());

                let sd = ad.sample_displacement_mu_m.map(|x| x.mean()).unwrap_or(0.0);
                let sid = to_discretize.sim_res.all_simulated_peaks[idx].struct_idx;
                to_discretize.sim_res.all_simulated_peaks[idx]
                    .iter_peaks()
                    .map(move |p: &Peak| {
                        let rp = p.get_adxrd_render_params(
                            wavelength_ams / 10.0,
                            wavelength_ams,
                            &instrument_parameters,
                            abs.0[0][sid],
                            &DomainSize::Isotropic(mean_ds_nm),
                            ds_eta,
                            mustrain,
                            mustrain_eta,
                            1.0,
                            sd,
                            ad.goniometer_radius_mm,
                            ad.monochromator_angle,
                        );

                        let intens = pseudo_voigt(0.0, rp.eta, rp.fwhm) * rp.intensity;
                        if matches!(mode, io::HKLDisplayMode::Structure { .. }) {
                            return (rp.pos, *p.i_hkl as f32);
                        }
                        (rp.pos, intens)
                    })
                    .collect_vec()
            }
            SimulationKind::EnergyDispersive(ed) => {
                let theta_rad = ed.theta_deg.to_radians();
                let f_lorentz = lorentz_polarization_factor_edxrd(theta_rad);
                let structures = std::sync::Arc::as_ref(&to_discretize.structures);
                let mac_generator = MACGenerator::from_structures_energy(
                    &structures[idx..idx + 1],
                    ed.energy_range_kev,
                )
                .unwrap_or_else(|err| {
                    error!("Could not create MAC Generator: {err}");
                    std::process::exit(1);
                });
                let mac_data = mac_generator
                    .get_mixture(std::iter::once((&structures[idx].wt_composition, 1.0f64)));
                let peak_sets = &to_discretize.sim_res.all_simulated_peaks[idx];
                assert_eq!(peak_sets.struct_idx, idx);
                peak_sets
                    .iter_peaks()
                    .map(|p: &Peak| {
                        let rp = p.get_edxrd_render_params(
                            theta_rad,
                            f_lorentz,
                            &DomainSize::Isotropic(mean_ds_nm),
                            ds_eta,
                            0.0,
                            0.0,
                            1.0,
                            &ed.beamline,
                            &mac_data,
                        );

                        let intens = pseudo_voigt(0.0, rp.eta, rp.fwhm) * rp.intensity;

                        if matches!(mode, io::HKLDisplayMode::Structure { .. }) {
                            return (rp.pos, *p.i_hkl as f32);
                        }
                        (rp.pos, intens)
                    })
                    .collect_vec()
            }
        };

        let scale = match mode {
            io::HKLDisplayMode::Standard { normalized: true }
            | io::HKLDisplayMode::Structure { normalized: true } => {
                intensities_positions
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .expect("at least one peak")
                    .1
            }
            _ => 1.0,
        };

        for (p, (pos, i)) in to_discretize.sim_res.all_simulated_peaks[idx]
            .iter_peaks()
            .zip(intensities_positions)
        {
            use std::fmt::Write;
            let mut hkls = String::new();
            write!(
                &mut hkls,
                "({h:2} {k:2} {l:2}) ",
                h = p.hkl[0],
                k = p.hkl[1],
                l = p.hkl[2]
            )
            .expect("enough memory");
            let intensity = i / scale;
            info!(
                "i_hkl: {intensity:.4} d_hkl: {d_hkl:.4} pos: {pos:.4} | {hkls}",
                intensity = intensity,
                d_hkl = p.d_hkl,
                pos = pos
            );
        }
    }
    Ok(())
}

fn load_cfg_and_check_sim_neeeded(args: &Cli) -> (Config, String) {
    let cfg_str = std::fs::read_to_string(&args.cfg).unwrap_or_else(|err| {
        error!(
            "Could not open File '{}': {}",
            args.cfg.to_str().unwrap(),
            err
        );
        std::process::exit(1);
    });

    let cfg_hash = format!("{:x}", sha2::Sha256::digest(&cfg_str));
    debug!(
        "SHA256 of config file {f}: {cfg_hash}",
        f = args.cfg.to_str().expect("config is utf8")
    );

    let cfg = serde_yaml::from_str::<Config>(&cfg_str).unwrap_or_else(|err| {
        error!(
            "Could not parse config: '{x}': {err}",
            x = args.cfg.to_str().unwrap()
        );
        std::process::exit(1);
    });

    let how_continue = prepare_output_directory(&args.io, &cfg_hash).unwrap_or_else(|err| {
        error!("Could not prepare output directory: {err}");
        std::process::exit(1);
    });
    match how_continue {
        io::CheckedOutput::ResimulationNotNeeded => {
            info!("Resimulation is not needed. Config Hash and YaXS version match. Skipping Simulation. In case you want to re-simulate anyway, use the command line flag '--re-simulate'.");
            std::process::exit(0);
        }
        io::CheckedOutput::ContinueNormally => (),
    }

    (cfg, cfg_hash)
}

fn main() {
    io::init_logging();

    let args = Cli::parse();

    if !args.io.quiet {
        info!("{}", ARTWORK)
    }

    let (mut cfg, cfg_hash) = load_cfg_and_check_sim_neeeded(&args);

    yaxs::init_gpu_if_applicable();

    let root_path = args
        .cfg
        .parent()
        .expect("cfg is file, must have parent dir")
        .to_owned();

    let timestamp_started: chrono::DateTime<Utc> = SystemTime::now().into();

    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(
        cfg.simulation_parameters.seed.unwrap_or(0),
    );

    let mut psd = prepare_peak_simulation(&mut cfg, &root_path, &mut rng).unwrap_or_else(|err| {
        error!("Could not simulate peaks: {err}");
        std::process::exit(1);
    });
    let structures = psd.structures.clone();
    let structure_paths = psd.structure_paths.clone();

    let vf_generator = CompositionGenerator::try_new(
        &mut psd.composition_constraints,
        cfg.sample_parameters.concentration_subset.clone(),
    )
    .unwrap_or_else(|err| {
        error!("Error: Could not generate volume fractions: '{err}'");
        std::process::exit(1);
    });

    let begin = Instant::now();

    let mut to_discretize = cfg
        .kind
        .simulate_peaks(
            psd,
            cfg.sample_parameters.clone(),
            cfg.simulation_parameters.texture_measurement,
            &mut rng,
        )
        .unwrap_or_else(|err| {
            error!("Could not simulate peaks: {err}");
            std::process::exit(1);
        });

    let elapsed = begin.elapsed().as_secs_f64();
    info!("Simulating Peak Positions took {elapsed:.2}s");

    if let Some(mode) = args.io.display_hkls {
        display_hkls(to_discretize, cfg, &structure_paths, mode).unwrap_or_else(|err| {
            error!("Could not display hkls: {err}");
            std::process::exit(1);
        });
        std::process::exit(0);
    }

    if let Some(ref rand_scale) = cfg.simulation_parameters.randomly_scale_peaks {
        let v = std::sync::Arc::get_mut(&mut to_discretize.sim_res)
            .expect("no other references to sim_res should exist at this point");
        for phase_peaks in v.all_simulated_peaks.iter_mut() {
            for peak in phase_peaks.iter_peaks_mut() {
                peak.i_hkl = NotNan::try_from(rand_scale.scale_peak(*peak.i_hkl, &mut rng))
                    .expect("scale is not Nan");
            }
        }
    }

    let cfg_file_name = args
        .cfg
        .file_name()
        .expect("configuration file was not moved until since starting simulation.");
    let mut copied_cfg_path = args.io.output_path.clone();
    copied_cfg_path.push(cfg_file_name);
    let _ = std::fs::copy(&args.cfg, copied_cfg_path).unwrap_or_else(|err| {
        error!(
            "Could not copy configuration file '{infile}' to output directory '{outdir}': {err}",
            infile = args.cfg.display(),
            outdir = args.io.output_path.display()
        );
        std::process::exit(1);
    });

    let params = cfg.simulation_parameters;
    let extra = io::Extra {
        n_patterns: params.n_patterns,
        max_phases: cfg.sample_parameters.structures.len(),
        texture: params.texture_measurement,
        encoding: cfg
            .sample_parameters
            .structures
            .iter()
            .map(|StructureDef { path, .. }| path.to_string())
            .collect_vec(),
        cfg: cfg.kind.clone(),
    };

    match cfg.kind.clone() {
        SimulationKind::AngleDispersive(angle_dispersive) => {
            let absorption_factors = PrecomputedLACs::try_new(
                angle_dispersive
                    .emission_lines
                    .iter()
                    .map(|line| line.wavelength_ams),
                &structures,
                &structure_paths,
            )
            .unwrap_or_else(|err| {
                error!("Could not precompute absorption factors: {err}.");
                std::process::exit(1);
            });
            let gen = adxrd::JobGen::new(
                angle_dispersive,
                to_discretize,
                params,
                vf_generator,
                absorption_factors,
                rng,
            );
            yaxs::render_and_write_jobs(gen, args.io, timestamp_started, extra, cfg_hash)
        }
        SimulationKind::EnergyDispersive(energy_dispersive) => {
            let mac_generator = MACGenerator::from_structures_energy(
                &structures,
                energy_dispersive.energy_range_kev,
            )
            .unwrap_or_else(|err| {
                error!("Could not create MAC Generator: {err}");
                std::process::exit(1);
            });
            let gen = edxrd::JobGen::new(
                energy_dispersive,
                to_discretize,
                params,
                vf_generator,
                mac_generator,
                rng,
            );
            yaxs::render_and_write_jobs(gen, args.io, timestamp_started, extra, cfg_hash)
        }
    }
    .unwrap_or_else(|err| {
        error!("Could not render peak shapes: {err}");
        std::process::exit(1);
    })
}
