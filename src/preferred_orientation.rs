use crate::math::linalg::{Mat3, Vec3, Vec4};
use rand::Rng;
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};

use crate::math::stats::BinghamDistribution;
use crate::structure::Lattice;

const HKL_NORM_TOL: f64 = 1e-3;

/// orientation distribution for perferred orientation in sample coordinates
///
/// * `orientation`: orientation of the distribution
/// * `axis_aligned_bingham_dist`: axis aligned bingham distribution
pub struct BinghamODF {
    pub orientation: Vec4<f64>,
    pub axis_aligned_bingham_dist: BinghamDistribution<4>,
}

impl BinghamODF {
    pub fn weight_crystallographic(
        &self,
        hkl: &Vec3<f64>,
        lat: &Lattice,
        chi: f64,
        phi: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        // how do we rotate around chi and phi
        let hkl_lattice = lat.mat.matmul(&hkl);

        // for edxrd ?
        // how about adxrd -> is that even a thing?
        //
        // Beam z axis is chi rotation (beam direction)
        // Beam y axis is phi rotation (up direction)
        // Beam x axis is perpendicular to beam and up (duh!)
        //
        // orientation distribution is relative to sample coordinates
        // therefore transform beam to sample coordinates using chi and phi
        //
        // detector
        //  \
        //    -
        //     \ scattered   ^             ^
        //       -     ray   | phi         | y
        //         \         |             |
        //           -       |             |
        //      theta  \     |     z, chi  |
        // - - - - - - - [sample]<---------o x
        //
        // for the orientation transformation from beam to sample:
        // first, the rotate around z by chi
        // then, rotate round y by phi
        //
        // XRD methods measure d-spacings in the direction of the Beam Z-Axis

        todo!("transform to quaternion");
        #[rustfmt::skip]
        let rot_z = Mat3::from_rows([
            [chi.cos(), -chi.sin(), 0.0],
            [chi.sin(),  chi.cos(), 0.0],
            [      0.0,        0.0, 1.0],
        ]);

        #[rustfmt::skip]
        let rot_y = Mat3::from_rows([
            [ phi.cos(), 0.0, phi.sin()],
            [       0.0, 1.0,       0.0],
            [-phi.sin(), 0.0, phi.cos()],
        ]);

        let beam_to_sample = rot_z.matmul(&rot_y);
        let sample_to_beam = beam_to_sample
            .try_inverse()
            .expect("rotation matrix must be invertible")
            .extend_to_homog();

        // bingham distribution describes orientations of domains relative to sample
        //
        // transform bingham distribution's orientation from sample to beam coords

        let bingham_alignment_in_beam = sample_to_beam.matmul(&self.orientation);

        let hkl_in_domain_coords = lat.mat.matmul(&hkl);
        // now, we need to compute how well the distribution over physical hkl
        // directions aligns with the direction (beam z unit vector)
        //
        // just sample many orientations (in beamline coordinates) and compute 
        // the dot product of beam direction (beam coords z axis) with the hkl
        // vector in that orientation

        let mut weight = 0.0;
        for _ in 0..N_SAMPLES {
            let domain_orientation_sample_coords = self.axis_aligned_bingham_dist.sample(rng);
            let domain_orientation_beam_coords = bingham_alignment_in_beam.quaternion_multiplication();
            todo!("transform the hkl");
            todo!("perform the dot product");
            todo!("compute the average");
        }

        weight
    }

    pub fn weight(&self, hkl: &Vec3<f64>, lat: &Lattice, chi: f64, phi: f64) -> f64 {}
}
