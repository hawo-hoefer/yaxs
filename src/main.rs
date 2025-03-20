use itertools::Itertools;
use ndarray::{arr2, IntoNdProducer};
use serde::Serialize;
use std::time::Instant;
use yaxs::cfg::{BackgroundSpec, Config, MetaGenerator};
use yaxs::pattern::EmissionLine;

const H_EV_S: f64 = 4.135_667_696e-15f64;
const C_M_S: f64 = 299_792_485.0f64;

pub fn e_kev_to_lambda_ams(e_kev: f64) -> f64 {
    // e = h * c / lambda
    // lambda = h * c / e
    // m      = ev * s * m / ev
    H_EV_S * C_M_S / e_kev * 1e7
}

fn main() {
    let mut args = std::env::args();
    let program = args.next().expect("Program Name");
    let files = args.collect_vec().into_boxed_slice();
    if files.is_empty() {
        println!("Usage: {program} </path/to/cif>");
        println!("   will simulate an XRD pattern with wavelength 1.54209 Amstrong and 2-theta range of (0, 90) for the cif");
        std::process::exit(1);
    };

    let emission_lines = [
        EmissionLine::new(1.5406, 1.0),
        EmissionLine::new(1.5445, 0.5206),
        EmissionLine::new(1.3923, 0.4121),
    ];

    let cfg = Config {
        struct_cifs: files,
        emission_lines: emission_lines.into(),
        n_steps: 2048,
        two_theta_range: (10.0, 70.0),
        eta_range: (0.1, 0.9),
        normalize: false,
        seed: Some(1234),
        n_simulations: 1000,
        ..Default::default()
    };

    println!("struct_cifs: {:?}", cfg.struct_cifs);

    let mut gen = MetaGenerator::from(cfg);
    let size = gen.cfg.n_steps * gen.cfg.n_simulations;
    let steps = gen.cfg.n_steps;

    let begin = Instant::now();
    let mut two_thetas = Vec::with_capacity(steps);
    two_thetas.resize(steps, 0.0);
    for (i, t) in two_thetas.iter_mut().enumerate() {
        let r = gen.cfg.two_theta_range;
        *t = r.0 + (r.1 - r.0) * (i as f64 / (gen.cfg.n_steps as f64 - 1.0));
    }
    let mut data = ndarray::Array2::<f64>::zeros((gen.cfg.n_simulations, gen.cfg.n_steps));

    for (i, mut chunk) in data.outer_iter_mut().enumerate() {
        if i % 100 == 0 {
            println!("Processing Job {i}");
        }
        let job = gen.generate_job();
        job.run(&two_thetas, chunk.as_slice_mut().unwrap());
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
