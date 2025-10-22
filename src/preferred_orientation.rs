use crate::math::linalg::{Mat3, Vec3, Vec4};
use rand::Rng;

use crate::structure::Lattice;

#[derive(Clone, Debug)]
pub struct BinghamParams {
    pub orientation: Vec4<f64>,
    pub ks: Vec4<f64>,
}

pub const N_SAMPLES: usize = 32768;

/// orientation distribution for perferred orientation in sample coordinates
///
/// * `orientation`: orientation of the distribution
/// * `axis_aligned_bingham_dist`: axis aligned bingham distribution
#[derive(Clone, Debug)]
pub struct BinghamODF {
    pub params: BinghamParams,
    pub axis_aligned_bingham_dist_samples: Vec<Vec4<f64>>,
}

impl BinghamODF {
    /// compute the weight scaling of a hkl peak according to the domain orientation
    /// distribution described by this bingham ODF
    ///
    /// * `hkl`: hkl vector
    /// * `lat`: lattice for hkl vector
    /// * `chi`: goniometer chi in degrees
    /// * `phi`: goniometer phi in degrees
    /// * `rng`: random number generator
    pub fn weight(&self, hkl: &Vec3<f64>, lat: &Lattice, chi: f64, phi: f64) -> f64 {
        // how do we rotate around chi and phi
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

        let rot_z = Vec4::quat_from_angle_axis(0.0, 0.0, 1.0, chi.to_radians());
        let rot_y = Vec4::quat_from_angle_axis(0.0, 1.0, 0.0, phi.to_radians());

        let beam_to_sample = rot_z.quaternion_multiplication(&rot_y);
        let sample_to_beam = beam_to_sample.quaternion_reciprocal();

        // bingham distribution describes orientations of domains relative to sample
        //
        // transform bingham distribution's orientation from sample to beam coords

        let bingham_alignment_in_beam =
            sample_to_beam.quaternion_multiplication(&self.params.orientation);

        let hkl_in_domain_coords = lat.mat.matmul(&hkl);
        // now, we need to compute how well the distribution over physical hkl
        // directions aligns with the direction (beam z unit vector)
        //
        // just sample many orientations (in beamline coordinates) and compute
        // the dot product of beam direction (beam coords z axis) with the hkl
        // vector in that orientation

        let kappa = 20.0f64;
        let norm_constant = kappa / (std::f64::consts::TAU * (kappa.exp() - (-kappa).exp()));

        let mut weight = 0.0;

        for domain_orientation_sample_coords in self.axis_aligned_bingham_dist_samples.iter() {
            let domain_orientation_beam_coords = bingham_alignment_in_beam
                .quaternion_multiplication(&domain_orientation_sample_coords);
            let hkl_in_beam_coords =
                domain_orientation_beam_coords.quaternion_transform(&hkl_in_domain_coords);
            let dot_with_beam_z = hkl_in_beam_coords.normalize()[2];
            // ad-hoc kernel density estimation

            // kernel density estimation using the von Mises-Fisher distribution
            // but without the scaling factor
            //
            // for testing it should be fine, but ideally we would want to scale
            // it properly, so XRD patterns produced using PO weighting
            // are compatible with Non preferred orientation ones
            weight += (kappa * dot_with_beam_z).exp() * norm_constant;
        }
        weight /= N_SAMPLES as f64;

        weight
    }
}
