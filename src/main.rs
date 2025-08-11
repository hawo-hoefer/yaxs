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
use yaxs::pattern::{adxrd, edxrd};
use yaxs::structure::Structure;

use log::{error, info};

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
    long_about = if cfg!(feature="cpu-only") {
        "YaXS simulates XRD patterns from CIF
CPU-only build: This may be much slower than GPU with support."
    } else {
        "YaXS simulates XRD patterns from CIF
GPU-accelerated build: CUDA-based peak rendering."
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
        mean_ds_nm: _,
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
        let mut p = CifParser::new(&cif);

        structures.push(
            Structure::try_from(&p.parse().unwrap_or_else(|err| {
                error!("Invalid CIF Syntax for '{path}': {err}");
                std::process::exit(1)
            }))
            .unwrap_or_else(|err| {
                error!("Invalid contents for CIF '{path}': {err}");
                std::process::exit(1);
            }),
        );

        structure_paths.push(struct_path.to_str().expect("valid path").to_owned());

        vf_constraints.push(*volume_fraction);
        strain_cfgs.push(strain.clone());
        pref_o.push(po.clone());
    }

    let vf_generator = VFGenerator::try_new(vf_constraints)
        .map_err(|err| {
            error!("Error: Could not generate volume fractions: '{err}'");
            std::process::exit(1);
        })
        .expect("error is handled inside");

    let begin = Instant::now();

    let to_discretize = cfg
        .kind
        .simulate_peaks(
            structures.into(),
            pref_o.into(),
            strain_cfgs.into(),
            structure_paths.clone().into(),
            cfg.sample_parameters.clone(),
            &mut rng,
        )
        .unwrap_or_else(|err| {
            error!("Could not simulate peaks: {err}");
            std::process::exit(1);
        });

    let elapsed = begin.elapsed().as_secs_f64();

    info!("Simulating Peak Positions took {elapsed:.2}s");

    let params = cfg.simulation_parameters;

    let extra = io::Extra {
        max_phases: cfg.sample_parameters.structures.len(),
        encoding: cfg
            .sample_parameters
            .structures
            .iter()
            .map(|StructureDef { path, .. }| path.to_string())
            .collect_vec(),
        preferred_orientation_hkl: cfg
            .sample_parameters
            .structures
            .iter()
            .map(|x| x.preferred_orientation.as_ref().map(|po| po.hkl))
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
        let (intensities, pattern_metadata) = render_jobs(jobs, xs, gen.abstol(), gen.n_phases())?;
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
