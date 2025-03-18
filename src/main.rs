use std::fs::File;
use std::io::{BufReader, Read};

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

    let wavelength_ams = 1.54209;
    let peaks = s.get_pattern(wavelength_ams, (0.0, 90.0));
    if peaks.len() > 1000 {
        println!("got {} peaks", peaks.len())
    } else {
        for (two_theta, i) in peaks.iter() {
            println!("{two_theta:12.4}, {i:12.4}")
        }
    }
}
