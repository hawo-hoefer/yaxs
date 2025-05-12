use chrono::Utc;
use clap::Parser;
use ordered_float::NotNan;
use rand::SeedableRng;
use std::io::{BufReader, BufWriter, ErrorKind, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime};
use yaxs::structure::simulate_peaks;

use yaxs::cfg::{Config, MetaGenerator};
use yaxs::io::{self, prepare_output_directory, write_to_npz, SimulationMetadata};
use yaxs::pattern::{render_jobs, render_write_chunked};

const H_EV_S: f64 = 4.135_667_696e-15f64;
const C_M_S: f64 = 299_792_485.0f64;

fn output_exists(path: &str) -> bool {
    std::fs::exists(&path).unwrap_or_else(|err| {
        eprintln!("Could not check whether output file/directory {path} exists: {err}");
        std::process::exit(1);
    })
}

pub fn e_kev_to_lambda_ams(e_kev: f64) -> f64 {
    // e = h * c / lambda
    // lambda = h * c / e
    // m      = ev * s * m / ev
    H_EV_S * C_M_S / e_kev * 1e7
}

#[derive(Parser)]
#[command(
    version,
    about = "Simulate a dataset of XRD patterns from YAML config.",
    long_about = if cfg!(feature="cpu-only") { 
        "YaXS simulates XRD patterns from CIF
CPU-only build: This may be much slower than GPU with support."
    } else {
        "GPU-accelerated build: CUDA-based peak rendering."
    }
)]
struct Cli {
    #[arg(value_name = "FILE", help = "Configuration yaml file.")]
    cfg: PathBuf,

    #[command(flatten)]
    io: io::Opts,
}

fn main() {
    let args = Cli::parse();

    let f = match std::fs::File::open(&args.cfg) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "Error: Could not open File '{}': {}",
                args.cfg.to_str().unwrap(),
                e
            );
            std::process::exit(1);
        }
    };

    let (gen, mut rng) = {
        let cfg: Config = match serde_yaml::from_reader(BufReader::new(f)) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!(
                    "Could not parse config: '{x}': {e}",
                    x = args.cfg.to_str().unwrap()
                );
                std::process::exit(1);
            }
        };
        eprintln!("struct_cifs: {:?}", cfg.struct_cifs);
        let rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.unwrap_or(0));
        (MetaGenerator::from(cfg), rng)
    };

    let output_path_exists = output_exists(&args.io.output_name);
    if output_path_exists {
        if args.io.overwrite {
            std::fs::remove_dir_all(&args.io.output_name).unwrap_or_else(|err| {
                eprintln!(
                    "Could not delete output directory '{}': {}",
                    &args.io.output_name, err
                );
                std::process::exit(1);
            });
        } else {
            eprintln!("Output path '{}' already exists.", args.io.output_name);
            std::process::exit(1);
        }
    }

    let timestamp_started: chrono::DateTime<Utc> = SystemTime::now().into();

    let begin = Instant::now();
    let (all_simulated_peaks, all_strains) = simulate_peaks(&gen, &mut rng);
    let elapsed = begin.elapsed().as_secs_f64();

    eprintln!("Simulating Peak Positions took {elapsed:.2}s");

    let begin_render = Instant::now();

    // Prepare rendering / generation (two_thetas buffer, concentrations)
    let mut two_thetas = Vec::with_capacity(gen.cfg.n_steps);
    two_thetas.resize(two_thetas.capacity(), 0.0f32);
    for (i, t) in two_thetas.iter_mut().enumerate() {
        let r = gen.cfg.two_theta_range;
        *t = (r.0 + (r.1 - r.0) * (i as f64 / (gen.cfg.n_steps as f64 - 1.0))) as f32;
    }

    let mut concentration_buf = Vec::with_capacity(gen.cfg.struct_cifs.len());
    concentration_buf.resize(
        concentration_buf.capacity(),
        NotNan::new(0.0).expect("0.0 is not NaN"),
    );

    // create rendering jobs
    let mut jobs = Vec::with_capacity(gen.cfg.n_patterns);
    for _ in 0..gen.cfg.n_patterns {
        let job = gen.generate_job(
            &all_simulated_peaks,
            &all_strains,
            &mut concentration_buf,
            &mut rng,
        );
        jobs.push(job);
    }

    prepare_output_directory(&args.io);

    // simulate and write to file(s)
    let datafiles = if let Some(_) = args.io.chunk_size {
        Some(render_write_chunked(&jobs, &two_thetas, &gen.cfg, &args.io))
    } else {
        let (intensities, pattern_metadata) = render_jobs(
            &jobs,
            &two_thetas,
            gen.cfg.abstol,
            gen.cfg.struct_cifs.len(),
        );
        let mut data_path = std::path::PathBuf::new();
        data_path.push(&args.io.output_name);
        data_path.push("data.npz");
        let _ = write_to_npz(data_path, &intensities, &pattern_metadata, args.io.compress)
            .unwrap_or_else(|_| std::process::exit(1));
        None
    };

    let elapsed = begin_render.elapsed().as_secs_f64();

    let timestamp_finished: chrono::DateTime<Utc> = SystemTime::now().into();

    let meta = serde_json::to_string(&SimulationMetadata {
        timestamp_started,
        timestamp_finished,
        chunked: datafiles.is_some(),
        datafiles,
        input_names: &io::INPUT_NAMES,
        target_names: &io::TARGET_NAMES,
        extra: io::Extra {
            encoding: gen.cfg.struct_cifs.clone().to_vec(),
            max_phases: gen.cfg.struct_cifs.len(),
            cfg: gen.cfg,
        },
    })
    .expect("SimulationMetadata is serializable");

    let mut path = std::path::PathBuf::new();
    path.push(args.io.output_name);
    path.push("meta.json");
    eprintln!("Writing {}", path.display());
    let f = std::fs::File::create_new(&path).unwrap_or_else(|err| {
        if err.kind() == ErrorKind::AlreadyExists {
            // TODO: time of check / time of use issue?
            eprintln!("Could not write meta.json. Since check at start of simulation, a file was written at '{}'. Printing contents to stderr just to be sure.", path.display());
            eprintln!("{}", meta);
            std::process::exit(1);
        } else {
            // TODO: time of check / time of use issue?
            eprintln!("Could not create meta.json (at '{}'): {err}. Printing contents to stderr just to be sure.", path.display());
            eprintln!("{}", meta);
            std::process::exit(1);
        }
    });
    BufWriter::new(f).write_all(meta.as_bytes()).unwrap_or_else(|err| {
        // TODO: time of check / time of use issue?
        eprintln!("Could not write meta.json (at '{}'): {err}. Printing contents to stderr just to be sure.", path.display());
        eprintln!("{}", meta);
        std::process::exit(1);
    });

    eprintln!("Done rendering patterns. Took {elapsed:.2}s");
}
