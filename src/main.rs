use itertools::Itertools;
use ndarray::{arr2, IntoNdProducer};
use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;
use yaxs::cfg::{BackgroundSpec, Config, MetaGenerator};
use yaxs::pattern::{EmissionLine, Peaks};

use clap::Parser;

const H_EV_S: f64 = 4.135_667_696e-15f64;
const C_M_S: f64 = 299_792_485.0f64;

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
    long_about = "XRD datasets are simulated for input sets in the cif"
)]
struct Args {
    #[arg(value_name = "FILE")]
    cfg: PathBuf,
}

fn main() {
    let args = Args::parse();

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

    let mut gen = {
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
        println!("struct_cifs: {:?}", cfg.struct_cifs);
        MetaGenerator::from(cfg)
    };

    let begin = Instant::now();
    let mut two_thetas = Vec::with_capacity(gen.cfg.n_steps);
    two_thetas.resize(two_thetas.capacity(), 0.0);
    for (i, t) in two_thetas.iter_mut().enumerate() {
        let r = gen.cfg.two_theta_range;
        *t = r.0 + (r.1 - r.0) * (i as f64 / (gen.cfg.n_steps as f64 - 1.0));
    }

    let min_line = &gen
        .cfg
        .emission_lines
        .iter()
        .min_by(|a, b| {
            a.wavelength
                .partial_cmp(&b.wavelength)
                .expect("no NaNs in wavelengths")
        })
        .expect("at least one emission line");

    let mut all_simulated_peaks = Vec::with_capacity(gen.structures.len());
    for s in &gen.structures {
        let mut permuted_phase_peaks = Vec::with_capacity(gen.cfg.structure_permutations);
        for _ in 0..gen.cfg.structure_permutations {
            let peaks = Peaks {
                peaks: s
                    .permute(gen.cfg.max_strain, &mut gen.rng)
                    .get_pattern(min_line.wavelength, &gen.cfg.two_theta_range)
                    .into(),
                wavelength_nm: min_line.wavelength,
            };
            permuted_phase_peaks.push(peaks);
        }
        all_simulated_peaks.push(permuted_phase_peaks);
    }

    let mut data = ndarray::Array2::<f64>::zeros((gen.cfg.n_patterns, gen.cfg.n_steps));
    for (i, mut pattern) in data.outer_iter_mut().enumerate() {
        if i % 100 == 0 {
            println!("Processing Job {i}");
        }
        let job = gen.generate_job(&all_simulated_peaks);
        job.discretize_into(pattern.as_slice_mut().unwrap(), &two_thetas);
    }

    let elapsed = begin.elapsed().as_secs_f64();
    eprintln!("Rendering patterns took {elapsed:.2}s");

    let out = hdf5_metno::File::create("out.h5").unwrap();
    let group = out.create_group("dataset").unwrap();
    let builder = group.new_dataset_builder();
    builder.clone().with_data(&data).create("patterns").unwrap();
    builder
        .clone()
        .with_data(&two_thetas)
        .create("two_thetas_deg")
        .unwrap();
}
