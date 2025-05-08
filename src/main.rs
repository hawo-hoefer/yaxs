use clap::Parser;
use ordered_float::NotNan;
use rand::SeedableRng;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use yaxs::cfg::{Config, MetaGenerator};
use yaxs::io::{self, write_to_npz};
use yaxs::pattern::{process_chunked, render_jobs, Peaks};

const H_EV_S: f64 = 4.135_667_696e-15f64;
const C_M_S: f64 = 299_792_485.0f64;

fn output_exists(path: &str, chunked: bool) -> (bool, String) {
    let path = if chunked {
        path.to_string()
    } else {
        format!("{path}.npz")
    };

    match std::fs::exists(&path) {
        Ok(exists) => (exists, path),
        Err(e) => {
            eprintln!("Could not check whether output file/directory {path} exists: {e}");
            std::process::exit(1)
        }
    }
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
    about = "Simulate a dataset of XRD patterns.",
    long_about = if cfg!(feature="cpu-only") { "CPU-only build. This may be much slower than GPU with support." } else {"Renders peak positions to patterns using GPU"}
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

    let (output_path_exists, chunk_dependent_output_path) =
        output_exists(&args.io.output_name, args.io.chunk_size.is_some());
    if output_path_exists && !args.io.overwrite {
        eprintln!("Output path '{chunk_dependent_output_path}' already exists.");
        std::process::exit(1);
    }

    let begin = Instant::now();
    let mut two_thetas = Vec::with_capacity(gen.cfg.n_steps);
    two_thetas.resize(two_thetas.capacity(), 0.0f32);
    for (i, t) in two_thetas.iter_mut().enumerate() {
        let r = gen.cfg.two_theta_range;
        *t = (r.0 + (r.1 - r.0) * (i as f64 / (gen.cfg.n_steps as f64 - 1.0))) as f32;
    }

    let min_line = &gen
        .cfg
        .emission_lines
        .iter()
        .min_by(|a, b| {
            a.wavelength_ams
                .partial_cmp(&b.wavelength_ams)
                .expect("no NaNs in wavelengths")
        })
        .expect("at least one emission line");

    let mut all_simulated_peaks = Vec::with_capacity(gen.structures.len());
    for s in &gen.structures {
        let mut permuted_phase_peaks = Vec::with_capacity(gen.cfg.structure_permutations);
        for _ in 0..gen.cfg.structure_permutations {
            let peaks = Peaks {
                peaks: s
                    .permute(gen.cfg.max_strain, &mut rng)
                    .get_pattern(min_line.wavelength_ams, &gen.cfg.two_theta_range)
                    .into(),
                wavelength_nm: min_line.wavelength_ams / 10.0,
            };
            permuted_phase_peaks.push(peaks);
        }
        all_simulated_peaks.push(permuted_phase_peaks);
    }
    let elapsed = begin.elapsed().as_secs_f64();
    eprintln!("Simulating Peak Positions took {elapsed:.2}s");

    let begin = Instant::now();

    let mut concentration_buf = Vec::with_capacity(gen.cfg.struct_cifs.len());
    concentration_buf.resize(
        concentration_buf.capacity(),
        NotNan::new(0.0).expect("0.0 is not NaN"),
    );

    let mut jobs = Vec::with_capacity(gen.cfg.n_patterns);
    for _ in 0..gen.cfg.n_patterns {
        let job = gen.generate_job(&all_simulated_peaks, &mut concentration_buf, &mut rng);
        jobs.push(job);
    }

    if let Some(_) = args.io.chunk_size {
        process_chunked(&jobs, &two_thetas, &gen.cfg, &args.io);
    } else {
        let intensities = render_jobs(&jobs, &two_thetas, gen.cfg.abstol);
        if output_path_exists {
            std::fs::remove_file(&chunk_dependent_output_path).unwrap_or_else(|e| {
                eprintln!("Error removing output path '{chunk_dependent_output_path}': {e}");
                std::process::exit(1);
            });
        }
        let _ = write_to_npz(chunk_dependent_output_path, &intensities, args.io.compress)
            .unwrap_or_else(|_| std::process::exit(1));
    }

    let elapsed = begin.elapsed().as_secs_f64();
    eprintln!("Done rendering patterns. Took {elapsed:.2}s");
}
