use std::collections::HashMap;
use std::f64::consts::PI;

use itertools::Itertools;
use nalgebra::{Complex, ComplexField, Matrix3, Vector3};
use ordered_float::NotNan;

use crate::element::atomic_scattering_params;
use crate::species::Species;

const TWO_THETA_ABSTOL: f64 = 1e-5;
const SCALED_INTENSITY_TOL: f64 = 1e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    pub mat: Matrix3<f64>,
}

impl Lattice {
    fn recip_lattice(&self) -> Lattice {
        Self {
            mat: self.mat.try_inverse().unwrap().transpose() * 2.0 * std::f64::consts::PI,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Structure {
    pub lat: Lattice,
    pub sites: Vec<Site>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub coords: Vector3<f64>,
    pub species: Species,
    pub occu: u16,
}

impl Structure {
    pub fn get_pattern(&self, wavelength_ams: f64, two_theta_range: (f64, f64)) -> Vec<(f64, f64)> {
        let min_r = (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0 - 1e-8;
        let max_r = (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0 + 1e-8;

        let recp_len = self.lat.recip_lattice().abc();
        let r_max = ((max_r + 0.15) * recp_len / (2.0 * PI)).map(|x| x.ceil() as i32);

        let nmin = -r_max;
        let nmax = r_max;
        let mut agg = HashMap::new();
        for (hkl, g_hkl) in (nmin[0]..nmax[0])
            .cartesian_product(nmin[1]..nmax[1])
            .cartesian_product(nmin[2]..nmax[2])
            .map(|((a, b), c)| {
                (
                    Vector3::<f64>::new(a as f64, b as f64, c as f64),
                    self.lat.mat * Vector3::<f64>::new(a as f64, b as f64, c as f64),
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
            for site in &self.sites {
                // g_dot_r = np.dot(frac_coords, np.transpose([hkl])).T[0]
                let g_dot_r: f64 = site.coords.dot(&hkl);
                for species in &site.species {
                    // el = site.specie
                    // coeff = ATOMIC_SCATTERING_PARAMS[el.symbol]
                    // fs = el.Z - 41.78214 * s2 * sum(
                    //     [d[0] * exp(-d[1] * s2) for d in coeff])
                    let z = species.el.z() as f64;
                    let coef = atomic_scattering_params(species.el).unwrap();
                    let sum: f64 = coef.iter().map(|d| d[0] * (-d[1] * s2).exp()).sum();
                    let fs = z - 41.78213 * s2 * sum;

                    // TODO: Debye-Waller Correction
                    // (we ignore it for now, in the test data we don't have DW-factors)
                    // dw_correction = np.exp(-dw_factors * s2)
                    let dw_correction = 1.0;

                    // f_hkl = np.sum(fs * occus * np.exp(2j * np.pi * g_dot_r) * dw_correction)
                    let f_part = fs
                        * site.occu as f64
                        * Complex::new(0.0, -2.0 * std::f64::consts::PI * g_dot_r).exp()
                        * dw_correction;
                    f_hkl += f_part;
                }
            }
            // Lorentz polarization correction for hkl
            // lorentz_factor = (1 + math.cos(2 * theta) ** 2) / (math.sin(theta) ** 2 * math.cos(theta))
            let lorentz_fact =
                (1.0 + (2.0 * theta).cos().powi(2)) / (theta.sin().powi(2) * theta.cos());

            // # Intensity for hkl is modulus square of structure factor
            // i_hkl = (f_hkl * f_hkl.conjugate()).real
            let i_hkl = (f_hkl * f_hkl.conjugate()).real();
            let two_theta = NotNan::new(theta.to_degrees() * 2.0).unwrap();
            *agg.entry(two_theta).or_insert(NotNan::new(0.0).unwrap()) +=
                NotNan::new(i_hkl * lorentz_fact).unwrap();
        }

        let Some((_, vmax)) = agg.iter().max_by_key(|&(_, b)| b) else {
            return Vec::new();
        };
        let vmax = f64::from(*vmax);
        let agg = agg
            .iter()
            .sorted_by_key(|&(a, _)| a)
            .map(|(a, b)| (f64::from(*a), f64::from(*b)))
            .filter(|&(_, b)| b / vmax >= SCALED_INTENSITY_TOL as f64)
            .collect_vec();

        let mut compressed: Vec<(f64, f64)> = Vec::with_capacity(agg.len());
        for (two_theta, intens) in agg.iter() {
            match compressed.last_mut() {
                Some((lt, li)) if ((*two_theta - *lt) < TWO_THETA_ABSTOL) => {
                    *li += intens;
                }
                None | Some(&mut (_, _)) => compressed.push((*two_theta, *intens)),
            }
        }
        compressed
    }
}
