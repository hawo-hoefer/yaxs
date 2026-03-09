use itertools::Itertools;

use crate::math::linalg::{Mat3, Vec3};

#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    pub mat: Mat3<f64>,
}

impl Lattice {
    pub fn recip_lattice_crystallographic(&self) -> Lattice {
        Self {
            mat: self.mat.try_inverse().unwrap(),
        }
    }

    fn recip_lattice(&self) -> Lattice {
        Self {
            mat: self
                .mat
                .try_inverse()
                .unwrap()
                .scale(2.0 * std::f64::consts::PI),
        }
    }

    /// Returns the volume of this [`Lattice`] in amstrong cubed.
    pub fn volume(&self) -> f64 {
        self.a().cross(&self.b()).dot(&self.c()).abs()
    }

    pub fn abc(&self) -> Vec3<f64> {
        Vec3::new(
            self.a().magnitude(),
            self.b().magnitude(),
            self.c().magnitude(),
        )
    }

    pub fn iter_hkls<'a>(
        &'a self,
        min_r: f64,
        max_r: f64,
    ) -> impl Iterator<Item = (Vec3<f64>, Vec3<f64>, f64)> + use<'a> {
        const RADIUS_TOL: f64 = 1e-8;
        let recip_lat = self.recip_lattice_crystallographic();
        let recp_len = recip_lat.recip_lattice().abc();

        let r_cells = max_r + 1e-8;
        let r_max = (recp_len.scale((r_cells + 0.15) / (2.0 * std::f64::consts::PI)))
            .map(|x| x.ceil() as i32);
        let global_min = -max_r - RADIUS_TOL;
        let global_max = max_r + RADIUS_TOL;

        let n_min = -r_max.clone();
        let n_max = r_max;
        (n_min[0]..n_max[0])
            .cartesian_product(n_min[1]..n_max[1])
            .cartesian_product(n_min[2]..n_max[2])
            .filter_map(move |((a, b), c)| -> Option<_> {
                let hkl = Vec3::new(a as f64, b as f64, c as f64);
                // pos is the position of the peak in reciprocal space
                let pos = recip_lat.mat.matmul(&hkl);
                let g_hkl = pos.magnitude();

                // currently, we produce XRD patterns like pymatgen
                // Neighbor mapping from pymatgen.core.lattice.get_points_in_spheres
                // does not seem to have any effect if center_coords is the 0-vector
                // As far as I can tell, it only applies when center_coords are something
                // other than the 0-vector so we will ignore it for now.
                // i tested this using a modification of their code and random cifs from
                // the COD-database
                if (g_hkl < max_r + RADIUS_TOL && g_hkl > min_r - RADIUS_TOL)
                    && g_hkl > 0.0
                    && pos
                        .iter_values()
                        .map(|&x| (x > global_min) && (x < global_max))
                        .all(|x| x)
                {
                    Some((hkl, pos, g_hkl))
                } else {
                    None
                }
            })
    }

    pub fn a(&self) -> Vec3<f64> {
        self.mat.row(0)
    }

    pub fn b(&self) -> Vec3<f64> {
        self.mat.row(1)
    }

    pub fn c(&self) -> Vec3<f64> {
        self.mat.row(2)
    }

    pub fn from_abc_angles(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> Lattice {
        // from pymatgen.core.Lattice.from_parameters
        let val = ((alpha.cos() * beta.cos() - gamma.cos()) / (alpha.sin() * beta.sin()))
            .clamp(-1.0, 1.0);

        let gamma_star = val.acos();

        let va = [a * beta.sin(), 0.0, a * beta.cos()];
        let vb = [
            -b * alpha.sin() * gamma_star.cos(),
            b * alpha.sin() * gamma_star.sin(),
            b * alpha.cos(),
        ];

        let vc = [0.0, 0.0, c];
        Lattice {
            mat: Mat3::from_rows([va, vb, vc]),
        }
    }
}

impl std::fmt::Display for Lattice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Lattice(")?;
        for ri in 0..self.mat.rows() {
            writeln!(
                f,
                "  {:5.2}, {:5.2}, {:5.2}",
                self.mat[(ri, 0)],
                self.mat[(ri, 1)],
                self.mat[(ri, 2)]
            )?;
        }
        writeln!(f, ")")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lattice_from_abc_angles_90() {
        let a = 1.0;
        let b = 2.0;
        let c = 3.0;

        let right_angle = 90.0f64.to_radians();

        let l = Lattice::from_abc_angles(a, b, c, right_angle, right_angle, right_angle);

        assert_eq!(l.a().magnitude(), a);
        assert_eq!(l.b().magnitude(), b);
        assert_eq!(l.c().magnitude(), c);

        let abc = l.abc();
        assert_eq!(abc[0], a);
        assert_eq!(abc[1], b);
        assert_eq!(abc[2], c);
    }

    #[test]
    fn lattice_from_abc_angles() {
        let a = 1.0;
        let b = 2.0;
        let c = 3.0;

        let alpha = 45.0f64.to_radians();
        let beta = 62.0f64.to_radians();
        let gamma = 87.0f64.to_radians();

        let l = Lattice::from_abc_angles(a, b, c, alpha, beta, gamma);

        assert_eq!(l.a().magnitude(), a);
        assert_eq!(l.b().magnitude(), b);
        assert_eq!(l.c().magnitude(), c);

        let abc = l.abc();
        assert_eq!(abc[0], a);
        assert_eq!(abc[1], b);
        assert_eq!(abc[2], c);
    }
}
