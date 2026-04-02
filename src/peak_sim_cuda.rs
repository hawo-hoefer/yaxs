use std::ffi::c_float;

use itertools::Itertools;

use crate::math::quaternion::Quaternion;
use crate::structure::Peak;
use crate::uninit_vec;

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
        hkls: *const Vec3<c_float>,     // [n_hkls_tot]
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
    // flattened vector of peaks for multiple permutations of the same structure
    reflection_parts: &[Peak],
    // axis aligned orientation samples from the bingham distribution
    ori_samples: &[Quaternion],
    // base orientations of the permutation's bingham distribution (length n_permutations)
    bingham_alignments: &[Quaternion],
    // goniometer phis
    phis: &[f32],
    // goniometer chis
    chis: &[f32],
    // number of hkls for each permutation (lengt n_permutations)
    n_hkls: &[usize],
    // KDE normalization constant
    norm_const: f64,
    // KDE kappa
    kappa: f64,
    // number of orientation samples per alignment  (should this be len(ori_samples?))
    num_orientation_samples_per_alignment: usize,
    alignments_per_measurement: usize, // stride
    i_hkls: &mut Vec<f32>,
) {
    // [alignment, hkl]
    let hkls = reflection_parts
        .iter()
        .map(|rp| rp.pos.map(|x| *x as c_float))
        .collect_vec();

    let n_i_hkls_base = hkls.len() * chis.len() * phis.len();
    let required_allocation = (n_i_hkls_base as isize - i_hkls.len() as isize).max(0) as usize;
    i_hkls.reserve_exact(required_allocation);
    assert_eq!(i_hkls.capacity(), n_i_hkls_base);
    unsafe { i_hkls.set_len(n_i_hkls_base) };

    for (p, i_hkl) in reflection_parts.iter().zip(i_hkls.iter_mut()) {
        *i_hkl = *p.i_hkl as f32;
    }

    let ffidata = ffi::FFIData::new(
        ori_samples,
        bingham_alignments,
        phis,
        chis,
        &hkls,
        num_orientation_samples_per_alignment,
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

#[cfg(all(test, feature = "use-gpu"))]
pub mod test {
    use itertools::Itertools;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro128PlusPlus;
    use std::collections::HashMap;

    use crate::cfg::{POCfg, POGenerator, TextureMeasurement};
    use crate::cif::CifParser;
    use crate::cuda_common::{self, CUDA_DEVICE_INFO};
    use crate::math::linalg::Vec3;
    use crate::math::quaternion::Quaternion;
    use crate::peak_sim::Alignment;
    use crate::peak_sim_cuda::single_phase_weight_hkls;
    use crate::structure::{Peak, Structure};

    const FM3M_CIF_DATA: &'static str = "# generated using pymatgen
data_test
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.59420000
_cell_length_b   3.59420000
_cell_length_c   3.59420000
_cell_angle_alpha   90.00000000
_cell_angle_beta    90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Cu
_chemical_formula_sum   Cu4
_cell_volume   46.43085912
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_type_symbol
 _atom_type_oxidation_number
  Cu0+  0.0
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Cu0+  Cu1  1  0.00000000  0.00000000  0.00000000  1.0
  Cu0+  Cu1  1  0.00000000  0.50000000  0.50000000  1.0
  Cu0+  Cu1  1  0.50000000  0.00000000  0.50000000  1.0
  Cu0+  Cu1  1  0.50000000  0.50000000  0.00000000  1.0";

    fn get_peaks() -> ((usize, Vec<Peak>), Structure) {
        let mut p = CifParser::new(&FM3M_CIF_DATA);
        let d = p.parse().expect("valid cif contents");
        let s = Structure::try_from(&d).expect("valid cif contents");

        let two_theta_range = (10.0f64, 45.0f64);
        let wavelength_ams = 0.7093;

        let (min_r, max_r) = (
            (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0,
            (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0,
        );
        let mut scattering_parameters = HashMap::new();
        s.gather_scattering_params(&mut scattering_parameters);

        (
            s.get_hkl_intensities_spacings(min_r, max_r, &scattering_parameters, None),
            s,
        )
    }

    #[test]
    fn peak_weight_cpu_vs_gpu_single() {
        let ((n, peaks), s) = get_peaks();

        for d in CUDA_DEVICE_INFO.iter() {
            // just do something so this does not get optimized away (not sure if needed)
            println!("{}", d.device_name);
        }

        let ori = Quaternion::from_axis_angle(1.0, 2.0, 3.0, 50.0f32.to_radians());
        let mut rng = Xoshiro128PlusPlus::seed_from_u64(1128123);
        let mut po_gen = POGenerator::Exact {
            k: Vec3::new(1000.0, 0.5, 0.5),
            orientation: (&ori).into(),
            sampling: crate::cfg::KDEApprox { n: 10, kappa: 5.0 },
        };
        let bing = po_gen.sample(&mut rng);

        let (chi, phi) = (7.0, 79.0);
        let precomp_ori = bing.precompute_orientation(chi, phi);

        let mut peaks_cpu = peaks.clone();
        let mut peaks_gpu = peaks.clone();

        s.apply_alignment_to_peaks(&mut peaks_cpu, Alignment::Precomputed { po: &precomp_ori });
        s.finalize_peaks(&mut peaks_cpu);

        let mut i_hkls = peaks.iter().map(|p| *p.i_hkl as f32).collect_vec();
        single_phase_weight_hkls(
            &peaks_gpu,
            &bing.axis_aligned_bingham_dist_samples,
            &[bing.params.orientation.clone()],
            &[phi as f32],
            &[chi as f32],
            &[peaks_gpu.len()],
            bing.norm_const,
            bing.kappa,
            po_gen.sampling_parameters().n,
            1,
            &mut i_hkls,
        );

        let peaks_gpu = s.apply_precomputed_weights_to_hkls_intensities(&mut peaks_gpu, &i_hkls);

        const ATOL: f64 = 1e-4;

        for (p_g, p_c) in peaks_gpu.iter().zip(peaks_cpu) {
            assert_eq!(p_g.pos, p_c.pos);
            let hkl = p_c.hkl.map(|x| *x as i32);
            println!("{:.4} {:.4}", p_c.i_hkl, p_g.i_hkl);
            assert!(
                (p_g.i_hkl - p_c.i_hkl).abs() < ATOL,
                "failed for {hkl} with intens {i_g} vs {i_c}",
                i_g = p_g.i_hkl,
                i_c = p_c.i_hkl
            );
        }
    }

    #[test]
    fn peak_weight_cpu_vs_gpu_multiple() {
        let ((n, peaks), s) = get_peaks();

        for d in CUDA_DEVICE_INFO.iter() {
            // just do something so this does not get optimized away (not sure if needed)
            println!("{}", d.device_name);
        }

        let ori = Quaternion::from_axis_angle(1.0, 2.0, 3.0, 50.0f32.to_radians());
        let mut rng = Xoshiro128PlusPlus::seed_from_u64(1128123);
        let mut po_gen = POGenerator::Exact {
            k: Vec3::new(1000.0, 0.5, 0.5),
            orientation: (&ori).into(),
            sampling: crate::cfg::KDEApprox { n: 10, kappa: 5.0 },
        };
        let bing = po_gen.sample(&mut rng);

        let chis = [7.0, 4.0];
        let phis = [3.0, 5.0];

        let mut all_peaks_cpu = Vec::new();

        for (chi, phi) in chis.iter().cartesian_product(phis.iter()) {
            let precomp_ori = bing.precompute_orientation(*chi as f64, *phi as f64);

            let mut peaks_cpu = peaks.clone();

            s.apply_alignment_to_peaks(&mut peaks_cpu, Alignment::Precomputed { po: &precomp_ori });
            s.finalize_peaks(&mut peaks_cpu);
            all_peaks_cpu.push(peaks_cpu);
        }

        let mut peaks_gpu = peaks.clone();
        let mut i_hkls = peaks.iter().map(|p| *p.i_hkl as f32).collect_vec();
        single_phase_weight_hkls(
            &peaks_gpu,
            &bing.axis_aligned_bingham_dist_samples,
            &[bing.params.orientation.clone()],
            &phis,
            &chis,
            &[n],
            bing.norm_const,
            bing.kappa,
            po_gen.sampling_parameters().n,
            chis.len(),
            &mut i_hkls,
        );

        let mut all_peaks_gpu = Vec::new();
        let mut hkl_idx = 0;
        for _ in 0..chis.len() {
            for _ in 0..phis.len() {
                all_peaks_gpu.push(s.apply_precomputed_weights_to_hkls_intensities(
                    &mut peaks_gpu,
                    &i_hkls[hkl_idx..hkl_idx + n],
                ));
                hkl_idx += n;
            }
        }

        const ATOL: f64 = 1e-4;

        for (peaks_cpu, peaks_gpu) in all_peaks_cpu.iter().zip(all_peaks_gpu.iter()) {
            for (p_g, p_c) in peaks_gpu.into_iter().zip(peaks_cpu) {
                assert_eq!(p_g.pos, p_c.pos);
                let hkl = p_c.hkl.map(|x| *x as i32);
                println!("{:.4} {:.4}", p_c.i_hkl, p_g.i_hkl);
                assert!(
                    (p_g.i_hkl - p_c.i_hkl).abs() < ATOL,
                    "failed for {hkl} with intens {i_g} vs {i_c}",
                    i_g = p_g.i_hkl,
                    i_c = p_c.i_hkl
                );
            }
        }
    }
}
