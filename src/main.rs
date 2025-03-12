use itertools::Itertools;
use nalgebra::{Complex, ComplexField, Matrix3, Vector3};
use ordered_float::NotNan;
use std::collections::HashMap;
use std::f64::consts::PI;

use sim_edxrd::element::atomic_scattering_params;
use sim_edxrd::species::Species;
use sim_edxrd::structure::{Structure, Site, Lattice};

const H_EV_S: f64 = 4.135_667_696e-15f64;
const C_M_S: f64 = 299_792_485.0f64;

pub fn e_kev_to_lambda_ams(e_kev: f64) -> f64 {
    // e = h * c / lambda
    // lambda = h * c / e
    // m      = ev * s * m / ev
    H_EV_S * C_M_S / e_kev * 1e7
}

fn main() {
    // let wavelength_ams = 1.5405;
    // let wavelength_ams = e_kev_to_lambda_ams(8.04);
    // let wavelength_ams = e_kev_to_lambda_ams(8.04);
    let sites = vec![
        Site {
            coords: Vector3::new(0.0, 0.5, 0.5),
            species: "Cu2+".parse().unwrap(),
            occu: 1
        },
        Site {
            coords: Vector3::new(0.5, 0.0, 0.0),
            species: "Cu2+".parse().unwrap(),
            occu: 1
        },
        Site {
            coords: Vector3::new(0.5816, 0.4184, 0.25),
            species: "O2-".parse().unwrap(),
            occu: 1
        },
        Site {
            coords: Vector3::new(0.4184, 0.5816, 0.75),
            species: "O2-".parse().unwrap(),
            occu: 1
        },
    ];

    let lattice = Lattice {
        #[rustfmt::skip]
        mat: Matrix3::new(
             2.16500590e-01,  2.16500590e-01, -3.27679720e-02,
             2.92175539e-01, -2.92175539e-01, 1.58116914e-17,
            -0.00000000e+00,  -0.00000000e+00, -1.94977383e-01
        ),
    };
    let s = Structure {
        lat: lattice,
        sites,
    };

    let wavelength_ams = 0.69;
    let peaks = s.get_pattern(wavelength_ams, (0.0, 15.0));
    for (two_theta, i) in peaks {
        println!("{two_theta:10.2}, {i:10.2}")
    }

    // 0, 0, 0  | 0.0
    // 0, 0, -1 | 0.19771170861976084
    // 0, 0, 1  | 0.19771170861976084
    // 0, -1, 0 | 0.36364687664399453
    // 0, 1, 0  | 0.36364687664399453
    // -1, 0, 0 | 0.3636468766439945
    // 1, 0, 0  | 0.3636468766439945
    //
    // frac_coords = lattice.get_fractional_coords(center_coords)
    // nmin_temp = np.floor(np.min(frac_coords, axis=0)) - maxr
    // nmax_temp = np.ceil(np.max(frac_coords, axis=0)) + maxr
}
