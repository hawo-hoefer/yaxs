use itertools::Itertools;

use crate::math::linalg::{Mat3, Vec3};


#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    pub mat: Mat3<f64>,
}


impl Lattice {
    pub fn recip_lattice_crystallographic(&self) -> Lattice {
        Self {
            mat: self.mat.try_inverse().unwrap().transpose(),
        }
    }

    fn recip_lattice(&self) -> Lattice {
        Self {
            mat: self
                .mat
                .try_inverse()
                .unwrap()
                .transpose()
                .scale(2.0 * std::f64::consts::PI),
        }
    }

    /// Returns the volume of this [`Lattice`] in amstrong cubed.
    pub fn volume(&self) -> f64 {
        self.mat
            .row(0)
            .cross(&self.mat.row(1))
            .dot(&self.mat.row(2))
            .abs()
    }

    pub fn abc(&self) -> Vec3<f64> {
        let mut values = [0.0; 3];
        for i in 0..self.mat.rows() {
            values[i] = self.mat.row(i).magnitude();
        }
        Vec3::new(values[0], values[1], values[2])
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

impl<'a> Lattice {
    pub fn iter_hkls(
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
}
