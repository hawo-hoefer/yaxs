use std::ffi::c_float;

use itertools::Itertools;

use crate::math::quaternion::Quaternion;
use crate::structure::ReflectionPart;

mod ffi {
    use std::ffi::c_float;

    #[repr(C)]
    pub struct Permutations {
        hkl_sizes: *const usize,
        n_permutations: usize,
    }

    impl Permutations {
        pub fn new(n_hkls: &[usize]) -> Self {
            Self {
                hkl_sizes: n_hkls.as_ptr(),
                n_permutations: n_hkls.len(),
            }
        }
    }

    #[repr(C)]
    pub struct FFIData {
        ori_samples: *const Quaternion, // [n_permutations * n_ori_per_alignment]
        bingham_alignments: *const Quaternion, // [n_permutations]
        hkls: *const Vec3<c_float>,         // [n_hkls_tot]
        phis: *const c_float,           // [n_phis]
        chis: *const c_float,           // [n_chis]

        n_phis: usize,
        n_chis: usize,
        n_ori_per_alignment: usize,
        n_hkls_tot: usize,
    }

    impl FFIData {
        pub fn new(
            ori_samples: &[Quaternion],
            bingham_alignments: &[Quaternion],
            phis: &[f32],
            chis: &[f32],
            hkls: &[Vec3<c_float>],
            n_ori_per_alignment: usize,
        ) -> Self {
            Self {
                ori_samples: ori_samples.as_ptr(),
                bingham_alignments: bingham_alignments.as_ptr(),
                hkls: hkls.as_ptr(),
                n_hkls_tot: hkls.len(),
                n_ori_per_alignment,
                phis: phis.as_ptr(),
                chis: chis.as_ptr(),
                n_phis: phis.len(),
                n_chis: chis.len(),
            }
        }
    }

    use crate::math::linalg::Vec3;
    use crate::math::quaternion::Quaternion;
    #[link(name = "cuda_lib")]
    extern "C" {
        pub fn weighted_i_hkls_single_structure(
            data: FFIData,
            chunks: Permutations,
            i_hkls_dst: *mut c_float,
            kappa: c_float,
            norm_const: c_float,
        ) -> bool;
    }
}

pub fn single_phase_weight_hkls(
    reflection_parts: &[ReflectionPart],
    ori_samples: &[Quaternion],
    bingham_alignments: &[Quaternion],
    phis: &[f32],
    chis: &[f32],
    n_hkls: &[usize],
    norm_const: f64,
    kappa: f64,
    samples_per_alignment: usize,
    alignments_per_measurement: usize,
    i_hkls: &mut Vec<f32>,
) {
    // [alignment, hkl, q]
    let hkls = reflection_parts
        .iter()
        .map(|rp| rp.pos.map(|x| *x as c_float))
        .collect_vec();

    let n_weights = hkls.len() * alignments_per_measurement;
    let required_allocation = (n_weights as isize - i_hkls.capacity() as isize).max(0) as usize;
    i_hkls.reserve_exact(required_allocation);

    unsafe { i_hkls.set_len(n_weights) };
    for (p, i_hkl) in reflection_parts.iter().zip(i_hkls.iter_mut()) {
        *i_hkl = *p.i_hkl as f32;
    }

    let ffidata = ffi::FFIData::new(
        ori_samples,
        bingham_alignments,
        phis,
        chis,
        &hkls,
        samples_per_alignment,
    );
    let permutations = ffi::Permutations::new(n_hkls);

    let ok = unsafe {
        ffi::weighted_i_hkls_single_structure(
            ffidata,
            permutations,
            i_hkls.as_mut_ptr(),
            kappa as c_float,
            norm_const as c_float,
        )
    };

    assert!(ok, "Error in cuda processing");
}
