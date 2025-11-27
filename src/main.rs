use chrono::{Datelike, Timelike, Utc};
use clap::Parser;
use colog::format::CologStyle;
use colored::Colorize;
use itertools::Itertools;
use rand::SeedableRng;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime};
use yaxs::cif::CifParser;
use yaxs::math::pseudo_voigt;
use yaxs::pattern::adxrd::InstrumentParameters;
use yaxs::pattern::{adxrd, edxrd, lorentz_polarization_factor, PeakRenderParams};
use yaxs::structure::Structure;

use log::{error, info, warn};

use yaxs::cfg::{Config, SimulationKind, StructureDef};
use yaxs::io::{
    self, prepare_output_directory, render_write_chunked, write_to_npz, OutputNames,
    SimulationMetadata,
};
use yaxs::pattern::{render_jobs, DiscretizeJobGenerator, Discretizer, VFGenerator};

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

struct CustomPrefix;

impl CologStyle for CustomPrefix {
    fn prefix_token(&self, level: &log::Level) -> String {
        let datetime: chrono::DateTime<Utc> = SystemTime::now().into();
        let datetime = datetime.naive_local();
        let date_str = format!(
            "{y}-{m:02}-{d:02}",
            y = datetime.year(),
            m = datetime.month(),
            d = datetime.day(),
        );
        let time_str = format!(
            "{h:02}:{m:02}:{s:02}",
            h = datetime.hour(),
            m = datetime.minute(),
            s = datetime.second(),
        );

        format!(
            "{pref}{level} {date} {time}{post}",
            pref = "[".blue().bold(),
            post = "]".blue().bold(),
            level = self.level_color(level, self.level_token(level)),
            date = self.level_color(level, &date_str),
            time = self.level_color(level, &time_str),
        )
    }

    #[rustfmt::skip]
    fn level_token(&self, level: &log::Level) -> &str {
        match level {
            log::Level::Error => "ERROR",
            log::Level::Warn  => "WARN ",
            log::Level::Info  => "INFO ",
            log::Level::Debug => "DEBUG",
            log::Level::Trace => "TRACE",
        }
    }

    fn level_color(&self, level: &log::Level, msg: &str) -> String {
        match level {
            log::Level::Error => msg.red().bold().to_string(),
            log::Level::Warn => msg.yellow().bold().to_string(),
            log::Level::Info => msg.green().to_string(),
            log::Level::Debug => msg.bright_purple().to_string(),
            log::Level::Trace => msg.white().to_string(),
        }
    }
}

fn main() {
    colog::basic_builder()
        .filter_level(log::LevelFilter::Info)
        .format(colog::formatter(CustomPrefix))
        .parse_env("LOG_LEVEL")
        .init();

    let args = Cli::parse();
    let f = match std::fs::File::open(&args.cfg) {
        Ok(f) => f,
        Err(e) => {
            error!(
                "Could not open File '{}': {}",
                args.cfg.to_str().unwrap(),
                e
            );
            std::process::exit(1);
        }
    };

    let mut cfg = match serde_yaml::from_reader::<_, Config>(BufReader::new(f)) {
        Ok(cfg) => {
            if cfg.simulation_parameters.n_patterns == 0 {
                error!("Number n_patterns needs to be larger than 0",);
                std::process::exit(1);
            }
            cfg
        }
        Err(e) => {
            error!(
                "Could not parse config: '{x}': {e}",
                x = args.cfg.to_str().unwrap()
            );
            std::process::exit(1);
        }
    };

    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(
        cfg.simulation_parameters.seed.unwrap_or(0),
    );
    let timestamp_started: chrono::DateTime<Utc> = SystemTime::now().into();

    if let Some(ref mut imp) = cfg.sample_parameters.impurities {
        // get upper and lower bound for d_hkl
        let (lb, ub) = {
            let (r_min, r_max) = cfg.kind.get_r_range();
            (1.0 / r_max, 1.0 / r_min)
        };
        for spec in imp.iter_mut() {
            spec.validate_d_hkl_or_adjust(lb, ub);
        }
    }

    let mut structures = Vec::new();
    let mut pref_o = Vec::new();
    let mut strain_cfgs = Vec::new();
    let mut structure_paths = Vec::new();
    let mut vf_constraints = Vec::new();

    for StructureDef {
        path,
        preferred_orientation: po,
        strain,
        volume_fraction,
        mean_ds_nm,
        ds_eta: _,
        mustrain: _,
    } in cfg.sample_parameters.structures.iter()
    {
        let mut struct_path = args
            .cfg
            .parent()
            .expect("cfg is file, must have parent dir")
            .to_owned();
        struct_path.push(path.clone());
        let mut reader = BufReader::new(
            std::fs::File::open(&struct_path)
                .map_err(|err| {
                    error!("Could not load cif at '{path}': {err}");
                    std::process::exit(1);
                })
                .expect("we exit if error"),
        );

        let mut cif = String::new();
        let _ = reader.read_to_string(&mut cif).unwrap();
        let mut p = CifParser::new(&cif).with_file(struct_path.display().to_string());

        let structure = Structure::try_from(&p.parse().unwrap_or_else(|err| {
            error!("Invalid CIF Syntax for '{path}': {err}");
            std::process::exit(1)
        }))
        .unwrap_or_else(|err| {
            error!("Invalid contents for CIF '{path}': {err}");
            std::process::exit(1);
        });
        if structure.density.is_none() {
            warn!(
                "Cannot output weight fractions because density is missing in cif '{p}.",
                p = struct_path.display()
            );
        }

        if mean_ds_nm.upper_bound() > 200.0 {
            error!("Specified a mean domain size with an upper bound of {hi} nm. The scherrer Formula is only valid up until 200 nm. Larger domain sizes are not supported for now. Quitting...", hi=mean_ds_nm.upper_bound());
            std::process::exit(1);
        }
        structure_paths.push(struct_path.to_str().expect("valid path").to_owned());
        vf_constraints.push(*volume_fraction);
        strain_cfgs.push(strain.clone());
        let po_gen = po.as_ref().map(|x| {
            x.try_into_generator(&mut rng).unwrap_or_else(|x| {
                error!(
                    "Could not get preferred orientation generator for {p}: {x}",
                    p = struct_path.display()
                );
                std::process::exit(1);
            })
        });
        pref_o.push(po_gen);
        structures.push(structure);
    }

    let vf_generator = VFGenerator::try_new(
        vf_constraints,
        cfg.sample_parameters.concentration_subset.clone(),
    )
    .map_err(|err| {
        error!("Error: Could not generate volume fractions: '{err}'");
        std::process::exit(1);
    })
    .expect("error is handled inside");

    let begin = Instant::now();

    let mut to_discretize = cfg
        .kind
        .simulate_peaks(
            structures.into(),
            pref_o.into(),
            strain_cfgs.into(),
            structure_paths.clone().into(),
            cfg.sample_parameters.clone(),
            cfg.simulation_parameters.texture_measurement,
            &mut rng,
        )
        .unwrap_or_else(|err| {
            error!("Could not simulate peaks: {err}");
            std::process::exit(1);
        });

    let elapsed = begin.elapsed().as_secs_f64();

    if let Some(mode) = args.io.display_hkls {
        info!("Displaying HKLs");

        for i in 0..to_discretize.structures.len() {
            let idx = to_discretize.sim_res.idx(i, 0);
            let s = &to_discretize.sample_parameters.structures[i];
            let mean_ds_nm = s.mean_ds_nm.mean();
            let ds_eta = s.ds_eta.mean();
            let mustrain = s
                .mustrain
                .as_ref()
                .map(|x| x.amplitude.mean())
                .unwrap_or(0.0);
            let mustrain_eta = s.mustrain.as_ref().map(|x| x.eta.mean()).unwrap_or(0.0);

            info!("======= Structure {} =======", structure_paths[i]);
            let intensities_positions = to_discretize.sim_res.all_simulated_peaks[idx]
                .iter()
                .map(|p| {
                    let (pos, intens) = match &cfg.kind {
                        SimulationKind::AngleDispersive(ad) => {
                            let wavelength_ams = ad.emission_lines[0].wavelength_ams;
                            let caglioti = ad
                                .instrument_parameters
                                .as_ref()
                                .map(|c| c.mean())
                                .unwrap_or(InstrumentParameters::zero());

                            let sd = ad.sample_displacement_mu_m.map(|x| x.mean()).unwrap_or(0.0);
                            let rp = p.get_adxrd_render_params(
                                wavelength_ams / 10.0,
                                &caglioti,
                                mean_ds_nm,
                                ds_eta,
                                mustrain,
                                mustrain_eta,
                                1.0,
                                sd,
                                ad.goniometer_radius_mm,
                            );

                            let intens = pseudo_voigt(0.0, rp.eta, rp.fwhm) * rp.intensity;
                            (rp.pos, intens)
                        }
                        SimulationKind::EnergyDispersive(energy_dispersive) => {
                            let theta_rad = energy_dispersive.theta_deg.to_radians();
                            let f_lorentz = lorentz_polarization_factor(theta_rad);
                            let rp = p.get_edxrd_render_params(
                                theta_rad,
                                f_lorentz,
                                mean_ds_nm,
                                ds_eta,
                                1.0,
                                &energy_dispersive.beamline,
                            );

                            let intens = pseudo_voigt(0.0, rp.eta, rp.fwhm) * rp.intensity;

                            (rp.pos, intens)
                        }
                    };

                    if matches!(mode, io::HKLDisplayMode::Structure { .. }) {
                        return (pos, p.i_hkl as f32);
                    }

                    (pos, intens)
                })
                .collect_vec();

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
                .iter()
                .zip(intensities_positions)
            {
                use std::fmt::Write;
                let mut hkls = String::new();
                for hkl in p.hkls.iter() {
                    write!(
                        &mut hkls,
                        "({h:2} {k:2} {l:2}) ",
                        h = hkl[0],
                        k = hkl[1],
                        l = hkl[2]
                    )
                    .expect("enough memory");
                }
                let intensity = i / scale;
                info!(
                    "i_hkl: {intensity:.4} d_hkl: {d_hkl:.4} pos: {pos:.4} | {hkls}",
                    intensity = intensity,
                    d_hkl = p.d_hkl,
                    pos = pos
                );
            }
        }
        std::process::exit(0);
    }

    info!("Simulating Peak Positions took {elapsed:.2}s");

    if let Some(ref rand_scale) = cfg.simulation_parameters.randomly_scale_peaks {
        let v = std::sync::Arc::get_mut(&mut to_discretize.sim_res)
            .expect("no other references to sim_res should exist at this point");
        for phase_peaks in v.all_simulated_peaks.iter_mut() {
            for peak in phase_peaks.iter_mut() {
                peak.i_hkl = rand_scale.scale_peak(peak.i_hkl, &mut rng);
            }
        }
    }

    // IOption<RandomlyScalePeaks>,

    prepare_output_directory(&args.io)
        .map_err(|err| {
            error!("Could not prepare output directory: {err}");
            std::process::exit(1);
        })
        .expect("error is handled inside");

    let cfg_file_name = args
        .cfg
        .file_name()
        .expect("configuration file was hopefully not moved until now.");
    let mut copied_cfg_path = args.io.output_path.clone();
    copied_cfg_path.push(cfg_file_name);
    let _ = std::fs::copy(&args.cfg, copied_cfg_path)
        .map_err(|err| {
            error!(
                "Could not copy configuration file '{infile}' to output directory '{outdir}': {err}",
                infile = args.cfg.display(),
                outdir = args.io.output_path.display()
            );
            std::process::exit(1);
        })
        .expect("we deal with the error inside");

    let params = cfg.simulation_parameters;
    let extra = io::Extra {
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
            let gen =
                adxrd::JobGen::new(angle_dispersive, to_discretize, params, vf_generator, rng);
            render_and_write_jobs(gen, args, timestamp_started, extra)
        }
        SimulationKind::EnergyDispersive(energy_dispersive) => {
            let gen =
                edxrd::JobGen::new(energy_dispersive, to_discretize, params, vf_generator, rng);
            render_and_write_jobs(gen, args, timestamp_started, extra)
        }
    }
    .unwrap_or_else(|err| {
        error!("Could not render peak shapes: {err}");
        std::process::exit(1);
    })
}

fn render_and_write_jobs<T, G>(
    mut gen: G,
    args: Cli,
    timestamp_started: chrono::DateTime<Utc>,
    extra: io::Extra,
) -> Result<(), String>
where
    T: Discretizer + Send + Sync + 'static,
    G: DiscretizeJobGenerator<Item = T>,
{
    let begin_render = Instant::now();
    let output_names = if args.io.chunk_size.is_some() {
        render_write_chunked(gen, &args.io)
    } else {
        // write as single chunk
        let mut jobs = Vec::with_capacity(gen.remaining());
        while let Some(job) = gen.next() {
            jobs.push(job);
        }
        let xs = gen.xs();
        let (intensities, pattern_metadata) = render_jobs(jobs, xs, &gen.get_job_params())?;
        let mut data_path = std::path::PathBuf::new();
        data_path.push(&args.io.output_path);
        data_path.push("data.npz");
        let (data_slot_names, metadata_slot_names) = write_to_npz(
            data_path,
            &intensities,
            &pattern_metadata,
            args.io.compress,
            1,
            1,
        )
        .unwrap_or_else(|err| {
            error!("Error writing data to disk: {err}");
            std::process::exit(1)
        });
        Ok(OutputNames {
            chunk_names: None,
            data_slot_names,
            metadata_slot_names,
        })
    };

    let output_names = output_names.unwrap_or_else(|err| {
        error!("could not write data to disk: {err}");
        std::process::exit(1)
    });

    let elapsed = begin_render.elapsed().as_secs_f64();

    let timestamp_finished: chrono::DateTime<Utc> = SystemTime::now().into();

    let meta = serde_json::to_string(&SimulationMetadata {
        timestamp_started,
        timestamp_finished,
        yaxs_version: env!("YAXS_VERSION").to_string(),
        chunked: args.io.chunk_size.is_some(),
        datafiles: output_names.chunk_names,
        input_names: &output_names.data_slot_names,
        target_names: &output_names.metadata_slot_names,
        extra,
    })
    .expect("SimulationMetadata is serializable");

    let mut path = std::path::PathBuf::new();
    path.push(args.io.output_path);
    path.push("meta.json");
    info!("Writing {}", path.display());
    let f = std::fs::File::create_new(&path).unwrap_or_else(|err| {
        if err.kind() == ErrorKind::AlreadyExists {
            // TODO: time of check / time of use issue?
            error!("Could not write meta.json. Since check at start of simulation, a file was written at '{}'. Printing contents to stderr just to be sure.", path.display());
            error!("{}", meta);
            std::process::exit(1);
        } else {
            // TODO: time of check / time of use issue?
            error!("Could not create meta.json (at '{}'): {err}. Printing contents to stderr just to be sure.", path.display());
            error!("{}", meta);
            std::process::exit(1);
        }
    });
    BufWriter::new(f).write_all(meta.as_bytes()).unwrap_or_else(|err| {
        // TODO: time of check / time of use issue?
        error!("Could not write meta.json (at '{}'): {err}. Printing contents to stderr just to be sure.", path.display());
        error!("{}", meta);
        std::process::exit(1);
    });

    info!("Done rendering patterns. Took {elapsed:.2}s");
    Ok(())
}
