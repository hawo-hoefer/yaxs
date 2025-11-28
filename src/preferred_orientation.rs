use crate::math::linalg::{Vec3, Vec4};

use crate::lattice::Lattice;

#[derive(Clone, Debug)]
pub struct BinghamParams {
    pub orientation: Vec4<f64>,
    pub ks: Vec4<f64>,
}

/// orientation distribution for perferred orientation in sample coordinates
/// approximated via Kernel density estimation
///
/// * `orientation`: orientation of the distribution
/// * `axis_aligned_bingham_dist_samples`: axis aligned bingham distribution samples
/// * `kappa`: kernel width parmeter
#[derive(Clone, Debug)]
pub struct KDEBinghamODF {
    pub params: BinghamParams,
    pub axis_aligned_bingham_dist_samples: Vec<Vec4<f64>>,
    pub kappa: f64,
}

impl KDEBinghamODF {
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
        // sample many orientations (in beamline coordinates) and compute
        // the dot product of beam direction (beam coords z axis) with the hkl
        // vector in that orientation

        // Von Mises-Fisher distribution normalization constant
        let norm_constant =
            self.kappa / (std::f64::consts::TAU * (self.kappa.exp() - (-self.kappa).exp()));

        let mut weight = 0.0;

        for domain_orientation_sample_coords in self.axis_aligned_bingham_dist_samples.iter() {
            let domain_orientation_beam_coords = bingham_alignment_in_beam
                .quaternion_multiplication(&domain_orientation_sample_coords);
            let hkl_in_beam_coords =
                domain_orientation_beam_coords.quaternion_transform(&hkl_in_domain_coords);
            // bingham_beam * domain_sample * hkl_domain * domain_beam^-1
            let dot_with_beam_z = hkl_in_beam_coords.normalize()[2];

            // kernel density estimation using the von Mises-Fisher distribution
            // normalization is applied below
            weight += (self.kappa * dot_with_beam_z).exp();
        }
        weight *= norm_constant;
        weight /= self.axis_aligned_bingham_dist_samples.len() as f64;

        weight
    }
}
