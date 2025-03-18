use std::fs::File;
use std::io::{BufReader, Read};

use itertools::Itertools;
use std::time::Instant;
use yaxs::cif::CifParser;
use yaxs::structure::Structure;

const H_EV_S: f64 = 4.135_667_696e-15f64;
const C_M_S: f64 = 299_792_485.0f64;

pub fn e_kev_to_lambda_ams(e_kev: f64) -> f64 {
    // e = h * c / lambda
    // lambda = h * c / e
    // m      = ev * s * m / ev
    H_EV_S * C_M_S / e_kev * 1e7
}

struct EmissionLine {
    wavelength: f64,
    weight: f64,
}

impl EmissionLine {
    /// create a new emission line from wavelength and weight
    ///
    /// * `wavelength`: wavelength in amstrong
    /// * `weight`: intensity of the emission line relative to other emission lines in the spectrum
    pub fn new(wavelength: f64, weight: f64) -> Self {
        Self { wavelength, weight }
    }
}

pub struct PatternMeta<'a> {
    pub structures: &'a [Structure],
    pub emission_lines: &'a [EmissionLine],
    pub n_steps: u32,
    pub two_theta_range: (f64, f64),
    pub eta: f64,
    pub mean_ds: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
}

impl<'a> PatternMeta<'a> {
    pub fn render_pattern(&self, two_thetas: &[f64], pat: &mut [f64]) {
        for s in self.structures {
            for EmissionLine { wavelength, weight } in self.emission_lines.iter() {
                let peaks = s.get_pattern(*wavelength, &self.two_theta_range);
                let wavelength_nm = wavelength / 10.0;
                for peak in peaks {
                    // * `pat`: target pattern
                    // * `two_thetas`: two theta values of pattern's intensities in degrees
                    // * `wavelength`: wavelength of the x-rays in nanometers
                    // * `mean_ds`: mean domain size used for scherrer broadening
                    // * `u`: caglioti parameter u
                    // * `v`: caglioti parameter v
                    // * `w`: caglioti parameter w
                    peak.into_pattern(
                        pat,
                        &two_thetas,
                        wavelength_nm,
                        *weight,
                        self.mean_ds,
                        self.eta,
                        self.u,
                        self.v,
                        self.w,
                    )
                }
            }
        }
    }
}

fn main() {
    let mut args = std::env::args();
    let program = args.next().expect("Program Name");
    let Some(file) = args.next() else {
        println!("Usage: {program} </path/to/cif>");
        println!("   will simulate an XRD pattern with wavelength 1.54209 Amstrong and 2-theta range of (0, 90) for the cif");
        std::process::exit(1);
    };
    let file = File::open(file).unwrap();
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    let _ = reader.read_to_string(&mut contents).unwrap();
    let contents = CifParser::new(&contents).parse();
    let s = Structure::from(&contents);

    let n_steps = 2048;
    let two_theta_range = (10.0, 70.0);
    let two_thetas = (0..n_steps)
        .map(|x| {
            two_theta_range.0
                + (two_theta_range.1 - two_theta_range.0) * x as f64 / (n_steps - 1) as f64
        })
        .collect_vec();
    let mut pat = (0..n_steps).map(|_| 0.0).collect_vec();

    let begin = Instant::now();

    let emission_lines = [
        EmissionLine::new(1.5406, 1.0),
        EmissionLine::new(1.5445, 0.5206),
        EmissionLine::new(1.3923, 0.4121),
    ];
    let meta = PatternMeta {
        structures: &[s],
        emission_lines: &emission_lines,
        n_steps: 2048,
        two_theta_range,
        eta: 0.5,
        mean_ds: 25.0,
        u: 0.0,
        v: 0.0,
        w: 0.0,
    };

    meta.render_pattern(&two_thetas, &mut pat);
    let elapsed = begin.elapsed().as_secs_f64();

    for (two_theta, intensity) in two_thetas.iter().zip(&pat) {
        println!("{two_theta},{intensity}")
    }
    eprintln!("Rendering pattern took {elapsed:.2}s")
}
