use itertools::Itertools;
use nalgebra::{Complex, ComplexField, Matrix3, Vector3};
use std::collections::HashMap;
use std::f64::consts::PI;

use sim_edxrd::element::atomic_scattering_params;
use sim_edxrd::species::Species;

#[derive(Debug)]
struct Lattice {
    mat: Matrix3<f64>,
}
impl Lattice {
    fn recip_lattice(&self) -> Lattice {
        Self {
            mat: self.mat.try_inverse().unwrap().transpose() * 2.0 * PI,
        }
    }

    fn abc(&self) -> Vector3<f64> {
        let mut values = [0.0; 3];
        for (i, v) in self
            .mat
            .row_iter()
            .into_iter()
            .map(|x| x.iter().map(|a| a.powi(2)).sum::<f64>().sqrt())
            .enumerate()
        {
            values[i] = v;
        }
        Vector3::from([values[0], values[1], values[2]])
    }
}

impl std::fmt::Display for Lattice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lattice(\n")?;
        for row in self.mat.row_iter() {
            write!(f, "  {:5.2}, {:5.2}, {:5.2}\n", row[0], row[1], row[2])?;
        }
        write!(f, ")\n")
    }
}

const H_EV_S: f64 = 4.135_667_696e-15f64;
const C_M_S: f64 = 299_792_485.0f64;

pub fn e_kev_to_lambda_ams(e_kev: f64) -> f64 {
    // e = h * c / lambda
    // lambda = h * c / e
    // m      = ev * s * m / ev
    H_EV_S * C_M_S / e_kev * 1e7
}

// struct Structure {
//     lat: Lattice,
//     sites: Vec<Site>,
// }

struct Site {
    coords: Vector3<f64>,
    species: Species,
    occu: u16,
}

fn main() {
    let two_theta_range: (f64, f64) = (0.0, 15.0);
    // let wavelength_ams = 1.5405;
    // let wavelength_ams = e_kev_to_lambda_ams(8.04);
    // let wavelength_ams = e_kev_to_lambda_ams(8.04);
    let wavelength_ams = 0.69;
    let sites = vec![
        Site {
            coords: Vector3::new(5.55111512e-17, 5.00000000e-01, 5.00000000e-01),
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
    let min_r = (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0 - 1e-8;
    let max_r = (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0 + 1e-8;

    // println!("{min_r}, {max_r}");

    let recp_len = lattice.recip_lattice().abc();
    let r_max = ((max_r + 0.15) * recp_len / (2.0 * PI)).map(|x| x.ceil() as i32);
    // println!("{r_max}");

    let nmin = -r_max;
    let nmax = r_max;
    let mut peaks = Vec::new();
    for (hkl, g_hkl) in (nmin[0]..nmax[0])
        .cartesian_product(nmin[1]..nmax[1])
        .cartesian_product(nmin[2]..nmax[2])
        .map(|((a, b), c)| {
            (
                Vector3::<f64>::new(a as f64, b as f64, c as f64),
                lattice.mat * Vector3::<f64>::new(a as f64, b as f64, c as f64),
            )
        })
        .map(|(hkl, pos)| (hkl, pos.magnitude()))
        .filter(|(_hkl, dist)| (*dist <= max_r) && (*dist >= min_r))
    {
        if g_hkl == 0.0 {
            continue;
        }
        // bragg condition
        let theta = (wavelength_ams * g_hkl / 2.0).asin();

        let s = g_hkl / 2.0;
        let s2 = s.powi(2);

        let mut f_hkl = Complex::new(0.0, 0.0);
        for site in &sites {
            // g_dot_r = np.dot(frac_coords, np.transpose([hkl])).T[0]
            println!("{}", site.coords);
            let g_dot_r: f64 = site.coords.dot(&hkl);
            for species in &site.species {
                // el = site.specie
                // coeff = ATOMIC_SCATTERING_PARAMS[el.symbol]
                // fs = el.Z - 41.78214 * s2 * sum(
                //     [d[0] * exp(-d[1] * s2) for d in coeff])
                let z = species.el.z() as f64;
                let coef = atomic_scattering_params(species.el).unwrap();
                println!("{:?}", coef);
                let sum: f64 = coef.iter().map(|d|d[0] * (-d[1] * s2).exp()).sum();
                let fs = z - 41.78213 * s2 * sum;

                // TODO: Debye-Waller Correction 
                // (we ignore it for now, in the test data we don't have DW-factors)
                // dw_correction = np.exp(-dw_factors * s2)
                let dw_correction = 1.0;

                // f_hkl = np.sum(fs * occus * np.exp(2j * np.pi * g_dot_r) * dw_correction)
                let f_part = fs * site.occu as f64 * Complex::new(0.0, -2.0 * std::f64::consts::PI * g_dot_r).exp() * dw_correction;
                f_hkl += f_part;
            }
        }
        // Lorentz polarization correction for hkl
        // lorentz_factor = (1 + math.cos(2 * theta) ** 2) / (math.sin(theta) ** 2 * math.cos(theta))
        let lorentz_fact = (1.0 + (2.0 * theta).cos().powi(2)) / (theta.sin().powi(2) * theta.cos());

        // # Intensity for hkl is modulus square of structure factor
        // i_hkl = (f_hkl * f_hkl.conjugate()).real
        let i_hkl = (f_hkl * f_hkl.conjugate()).real();
        let two_theta = theta.to_degrees() * 2.0;
        peaks.push((two_theta, i_hkl * lorentz_fact));
    }
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
