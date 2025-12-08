#include "assert.h"
#include <cstddef>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

#include "./cuda_common.cu"

extern "C" {

struct Quaternion {
  float w;
  float x;
  float y;
  float z;
};

struct Vec3 {
  float x;
  float y;
  float z;
};

__device__ Vec3 vec3_new(float x, float y, float z) { return (Vec3){.x = x, .y = y, .z = z}; }

__device__ Vec3 vec3_cross(Vec3 a, Vec3 b) {
  return (Vec3){
      .x = a.y * b.z - a.z * b.y,
      .y = a.z * b.x - a.x * b.z,
      .z = a.x * b.y - a.y * b.x,
  };
}

__device__ Vec3 vec3_scale(Vec3 a, float s) {
  return (Vec3){
      .x = a.x * s,
      .y = a.y * s,
      .z = a.z * s,
  };
}

__device__ Vec3 vec3_add(Vec3 a, Vec3 b) {
  return (Vec3){
      .x = a.x + b.x,
      .y = a.y + b.y,
      .z = a.z + b.z,
  };
}

__device__ void vec3_normalize(Vec3 *v) {
  float mag = v->x * v->x + v->y * v->y + v->z * v->z;
  assert(mag != 0);
  float fac = 1.0 / sqrt(mag);
  v->x *= fac;
  v->y *= fac;
  v->z *= fac;
}

__device__ Vec3 unit_quat_tf_unchecked(Quaternion q, Vec3 v) {
  Vec3 qxyz = (Vec3){q.x, q.y, q.z};
  Vec3 t = vec3_cross(qxyz, v);
  Vec3 v_ = vec3_add(vec3_add(v, vec3_scale(t, q.w)), vec3_cross(qxyz, t));
  return v;
}

__global__ void reduce_weights_per_hkl_kde(float *w, float *results, float norm_const, size_t n_hkls,
                                           size_t n_ori_samples) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n_hkls)
    return;

  size_t hkl_index = tid % n_hkls;

  // iterate across orientation samples
  float weight = 0;
  for (size_t i = tid * n_ori_samples; i < (tid + 1) * n_ori_samples; ++i) {
    weight += w[n_ori_samples * hkl_index + i];
  }
  results[tid] = results[tid] * weight * norm_const;
}

struct Permutations {
  size_t *hkl_sizes;
  size_t n_permutations;
};

struct FFIData {
  Quaternion *ori_samples;        // [n_permutations * n_ori_per_alignment]
  Quaternion *bingham_alignments; // [n_permutations]
  Vec3 *hkls;                     // [n_hkls_tot]
  float *phis;                    // [n_phis]
  float *chis;                    // [n_chis]

  size_t n_phis;
  size_t n_chis;
  size_t n_ori_per_alignment;
  size_t n_hkls_tot;
};

/**
 * @brief compute the permutation index from hkl index
 *
 *              n_hkls_1        n_hkls_2
 * n_hkls: [ permutation 1 | permutation 2 | ... ]
 *                                ^
 *                            hkl_index
 *
 * @param hkl_index index into global hkl array
 * @param n_hkls total number of global , sum(n_hkls)
 * @param n_permutations number of permutations (length of n_hkls)
 * @return permutation index
 */
__device__ size_t permutation_index(size_t hkl_index, Permutations permutations) {
  size_t hkl_acc = 0;
  size_t permutation_idx = 0;

  for (size_t i = 0; i < permutations.n_permutations; ++i) {
    hkl_acc += permutations.hkl_sizes[i];
    if (hkl_acc >= hkl_index)
      break;

    permutation_idx += 1;
  }
  return permutation_idx;
}

/*
 * @verbatim
 *                ori * stride    ori * stride   ...    ori * stride
 * ori_samples [ permutation 1 | permutation 2 | ... | permutation n ]
 *
 * n_hkls      [   n_hkls_1    ,    n_hkls_2   , ... ,    n_hkls_n   ]
 * hkls        [ permutation 1 | permutation 2 | ... | permutation n ]
 *
 * n_partial_weights = ori * stride * (n_hkls_1 + n_hkls_2 ... n_hkls_n)
 *                   = ori * stride * n_hkls_tot
 *
 *   |----------------------------- n_hkls_tot ---------------------|
 *
 *       hkls_1     hkls_2      ...       ...             hkls_n
 * o |           |           |         |          |                 |
 * r |           |           |         |          |                 |
 * i |           |           |         |          |                 |
 *   |           |           |         |          |                 |
 * s |           |           |         |          |                 |
 *                                 ^
 * ^ inside set of permutations    |
 *                                 |
 *                            permutation_idx: "which chunk are we in"
 *
 * n_weights         = n_hkls_1 + n_hkls_2 + ... n_hkls_n = n_hkls_tot
 * @endverbatim
 */
__global__ void compute_single_hkl_ori_weight(Quaternion *q, Vec3 *h, float *w, Permutations permutations, float kappa,
                                              size_t total_ori_samples, size_t stride_in_alignments,
                                              size_t n_ori_per_alignment, size_t n_hkls_tot) {

  size_t partial_weight_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (partial_weight_index >= n_hkls_tot * stride_in_alignments * n_ori_per_alignment)
    return;

  // indexing in w:
  // [alignment_idx, hkl, ori_samples]

  size_t alignment_idx = partial_weight_index / (n_ori_per_alignment * n_hkls_tot);
  size_t r = partial_weight_index % (n_ori_per_alignment * n_hkls_tot);
  size_t orientation_index = r % n_ori_per_alignment;
  size_t hkl_index = r / n_ori_per_alignment;
  size_t permutation_idx = permutation_index(hkl_index, permutations);

  size_t global_orientation_idx = permutation_idx * stride_in_alignments * n_ori_per_alignment +
                                  alignment_idx * n_ori_per_alignment + orientation_index;

  Vec3 hkl_in_domain_coords = h[hkl_index];
  Quaternion domain_to_beam = q[global_orientation_idx];
  Vec3 hkl_in_beam_coords = unit_quat_tf_unchecked(domain_to_beam, hkl_in_domain_coords);
  float dot_with_beam_z = hkl_in_beam_coords.z;

  // kernel density estimation using the von Mises-Fisher distribution
  // normalization is applied below
  w[partial_weight_index] = expf(kappa * dot_with_beam_z);
}

__global__ void compute_hkl_weight(Quaternion *q, Vec3 *h, float *w, Permutations permutations, float kappa,
                                   float norm_const, size_t total_ori_samples, size_t stride_in_alignments,
                                   size_t n_ori_per_alignment, size_t n_hkls_tot) {

  size_t weight_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (weight_index >= n_hkls_tot * stride_in_alignments)
    return;

  // indexing in w:
  // [n_permutations, chi, phi, hkl]
  //
  // indexing in q:
  // [n_permutations, chi, phi, n_ori_per_alignment]

  // for every hkl, iterate all corresponding orientations per alignment and sum over the kde thing

  size_t alignment_idx = weight_index / (stride_in_alignments * n_hkls_tot);
  size_t r = weight_index % (n_ori_per_alignment * n_hkls_tot);

  size_t orientation_index = r % n_ori_per_alignment;
  size_t hkl_index = r / n_ori_per_alignment;
  size_t permutation_idx = permutation_index(hkl_index, permutations);

  Vec3 hkl_in_domain_coords = h[hkl_index];
  float w_sum = 0.0;
  for (size_t orientation_index = 0; orientation_index < n_ori_per_alignment; ++orientation_index) {
    size_t global_orientation_idx = permutation_idx * stride_in_alignments * n_ori_per_alignment +
                                    alignment_idx * n_ori_per_alignment + orientation_index;
    Quaternion domain_to_beam = q[global_orientation_idx];
    Vec3 hkl_in_beam_coords = unit_quat_tf_unchecked(domain_to_beam, hkl_in_domain_coords);
    float dot_with_beam_z = hkl_in_beam_coords.z;
    // kernel density estimation using the von Mises-Fisher distribution
    // normalization is applied below
    w_sum += expf(kappa * dot_with_beam_z);
  }

  // w[weight_index] contains i_hkl, so need to multiply kde density
  w[weight_index] *= w_sum * norm_const;
}

__global__ void normalize_hkls(Vec3 *h, size_t n_hkls_tot) {
  size_t hkl_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (hkl_idx >= n_hkls_tot)
    return;
  vec3_normalize(&h[hkl_idx]);
}

__device__ Quaternion q_hamilton(Quaternion q0, Quaternion q1) {
  return (Quaternion){
      .w = q0.w * q1.w - q0.x * q1.x - q0.y * q1.y - q0.z * q1.z,
      .x = q0.w * q1.x + q0.x * q1.w + q0.y * q1.z - q0.z * q1.y,
      .y = q0.w * q1.y - q0.x * q1.z + q0.y * q1.w + q0.z * q1.x,
      .z = q0.w * q1.z + q0.x * q1.y - q0.y * q1.x + q0.z * q1.w,
  };
}

__device__ Quaternion q_conj(Quaternion q) { return (Quaternion){.w = q.w, .x = -q.x, .y = -q.y, .z = -q.z}; }

__device__ Quaternion q_unit_recip(Quaternion q) {
  Quaternion conjug = q_conj(q);
  // since q is a unit quaternion, mag must be 1, so we omit scaling
  // let mag = conjug.magnitude();
  // conjug.scale_inplace(1.0 / (mag * mag));
  return conjug;
}

__device__ Quaternion q_from_angle_axis(float alpha, float x, float y, float z) {
  Vec3 v = vec3_new(x, y, z);
  vec3_normalize(&v);
  alpha = DEG2RAD(alpha);
  float alpha_half_sin = sinf(alpha / 2.0);

  return (Quaternion){
      .w = cosf(alpha / 2.0),
      .x = v.x * alpha_half_sin,
      .y = v.y * alpha_half_sin,
      .z = v.z * alpha_half_sin,
  };
}

__device__ Quaternion beam_to_sample_tf(float chi, float phi) {
  Quaternion beam_chi = q_from_angle_axis(0.0, 0.0, 1.0, chi);
  Quaternion beam_phi = q_from_angle_axis(0.0, 1.0, 0.0, phi);

  Quaternion chi_phi = q_hamilton(q_unit_recip(beam_chi), beam_phi);

  return q_hamilton(beam_chi, chi_phi);
}

__global__ void precompute_alignment_transformations(float *chis, float *phis, Quaternion *alignment_transformations,
                                                     size_t n_chis, size_t n_phis) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= n_phis * n_chis)
    return;

  size_t chi_index = tid / n_phis;
  size_t phi_index = tid % n_phis;
  float phi = phis[phi_index];
  float chi = chis[chi_index];

  alignment_transformations[tid] = beam_to_sample_tf(chi, phi);
}

/**
 * @brief precompute transformations from beam to bingham distribution orientation
 *
 * @param alignment_transformations precomputed transformations from beam to sample
 * @param bingham_orientations orientations of bingham distribution samples relative to sample
 * @param beam_to_bingham destination array of size [n_permutations, chis, phis] / [n_permutations, stride]
 * @param stride number of alignments to compute
 * @param n_permutations number of permutations
 */
__global__ void precompute_beam_to_bingham(Quaternion *alignment_transformations, Quaternion *bingham_orientations,
                                           Quaternion *beam_to_bingham, size_t stride, size_t n_permutations) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n_permutations * stride)
    return;

  size_t alignment_idx = tid % stride;
  size_t perm_idx = tid / stride;
  assert(perm_idx < n_permutations);

  Quaternion beam_to_sample = alignment_transformations[alignment_idx];
  Quaternion sample_to_bingham = bingham_orientations[perm_idx];
  beam_to_bingham[tid] = q_hamilton(beam_to_sample, sample_to_bingham);
}

/*
 * base_orientation_samples contains is an array of quaternions of shape
 * [permutations, n_ori_samples]
 */
__global__ void transform_quaternions(Quaternion *beam_to_bingham, Quaternion *base_orientation_samples,
                                      Quaternion *dst, size_t stride, size_t ori_samples_per_alignment,
                                      size_t n_permutations) {
  // indexing in base_orientation_samples
  // [n_permutations, ori_samples_per_alignment]
  //
  // expand each orientation sample in base_orientation_samples
  // into stride (n_chi * n_phi) orientation samples according to
  // bingham distribution orientation and goniometer orientation
  // (in alignment_transformations)
  //
  // indexing in beam_to_bingham
  // [n_permutations, stride] / [n_permutations, chi, phi]
  //
  // indexing in dst:
  // [n_permutations, chi, phi, ori_samples_per_alignment]
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  // tid is from 0 to n_permutations * n_chi * n_phi * n_ori_samples
  //                                   |  stride   |

  if (tid >= stride * n_permutations * ori_samples_per_alignment)
    return;

  size_t perm_idx = tid / (stride * ori_samples_per_alignment);
  size_t alignment_and_permutation_idx = tid / ori_samples_per_alignment;
  size_t alignment_idx = alignment_and_permutation_idx % stride;
  size_t ori_sample_idx = tid % ori_samples_per_alignment;

  size_t beam_to_bingham_idx = perm_idx * stride + alignment_idx;
  size_t base_orientation_idx = perm_idx;

  Quaternion beam_to_bingham_instance = beam_to_bingham[beam_to_bingham_idx];
  Quaternion base_ori = base_orientation_samples[base_orientation_idx];
  Quaternion sample_to_bingham = q_hamilton(beam_to_bingham_instance, base_ori);
  dst[tid] = sample_to_bingham;
}

/**
 * @brief compute hkl weights for a single permutation of a structure
 *
 * @param ori_samples samples of the orientation distribution function
 * @param hkls hkls to weight (as absolute positions in amstrong in the unit cell)
 * @param n_alignments number of alignments
 * @param n_ori_samples
 * @param n_hkls
 * @param weights_dst
 * @param kappa
 * @param norm_const
 * @param errfn
 * @param infofn
 * @param debugfn
 * @return
 */
bool weighted_i_hkls_single_structure(FFIData ffidata, Permutations permutations, float *i_hkls, float kappa,
                                      float norm_const) {
  infof("Beginning CUDA texture weight computation");

  size_t stride = ffidata.n_chis * ffidata.n_phis;

  // clang-format off
  size_t base_orientations_space  = ffidata.n_ori_per_alignment * permutations.n_permutations * sizeof(Quaternion);
  size_t alignment_tf_space    = stride * sizeof(Quaternion);
  size_t bingham_ori_space     = permutations.n_permutations * sizeof(Quaternion);
  size_t beam_to_bingham_space = stride * permutations.n_permutations * sizeof(Quaternion);
  size_t beam_to_domain_space  = permutations.n_permutations * stride * ffidata.n_ori_per_alignment * sizeof(Quaternion);
  size_t permutations_space    = permutations.n_permutations * sizeof(size_t);

  size_t phis_space            = ffidata.n_phis                                                    * sizeof(float);
  size_t chis_space            = ffidata.n_chis                                                    * sizeof(float);

  size_t hkl_space             = ffidata.n_hkls_tot                                                * sizeof(Vec3);
  size_t res_space             = ffidata.n_hkls_tot * stride                               * sizeof(float);
  // size_t partial_weights_space = ffidata.n_hkls_tot * stride * ffidata.n_ori_per_alignment * sizeof(float);
  // clang-format on

  // TODO: update memory requirements
  // clang-format off
  snprintf(tmp_str_buf, sizeof(tmp_str_buf),
           "Allocating device memory for computing weights of %ld hkls for %ld orientations (%ld permutations).\n"
           "Base Quaternions:            %10.3f MiB\n"
           "Alignment Quaternions:       %10.3f MiB\n"
           "Bingham Quaternions:         %10.3f Mib\n"
           "Beam to Bingham Quaternions: %10.3f MiB\n"
           "Transformed Quaternions:     %10.3f MiB\n"
           "HKL vectors:                 %10.3f MiB\n"
           // "Partial Weights:             %10.3f MiB\n"
           "Weights:                     %10.3f MiB\n"
           "HKL sizes:                   %10.3f B\n"
           "Total:                       %10.3f MiB",
           ffidata.n_hkls_tot, ffidata.n_ori_per_alignment, permutations.n_permutations,
           (float)base_orientations_space / 1e6, 
           (float)alignment_tf_space / 1e6,
           (float)bingham_ori_space / 1e6,
           (float)beam_to_bingham_space / 1e6,
           (float)beam_to_domain_space / 1e6,
           (float)hkl_space / 1e6,
           // (float)partial_weights_space / 1e6,
           (float)res_space / 1e6,
           (float)permutations_space,

           (float)(base_orientations_space + bingham_ori_space + alignment_tf_space + beam_to_domain_space +
                   beam_to_bingham_space + 
                   // partial_weights_space 
                   + hkl_space + res_space) /
               1e6);
  // clang-format on
  debugf(tmp_str_buf);

  Quaternion *base_orientation_samples; // device version of bingham samples
  Quaternion *alignment_tf;
  Quaternion *bingham_ori; // n_permutations long array of bingham orientations for each permutation
  Quaternion *beam_to_bingham;
  Quaternion *beam_to_domain;

  Quaternion *q_d;
  Vec3 *h_d;
  float *w_d;
  float *res;
  float *chis;
  float *phis;

  Permutations pm_gpu{.hkl_sizes = NULL, .n_permutations = permutations.n_permutations};

  // clang-format off
  cu_lerr(cudaMalloc(&base_orientation_samples, base_orientations_space), "allocating base quaternions");
  cu_lerr(cudaMalloc(&alignment_tf,             alignment_tf_space),      "allocating alignment transformations");
  cu_lerr(cudaMalloc(&beam_to_bingham,          beam_to_bingham_space),   "allocating beam_to_bingham orientations");
  cu_lerr(cudaMalloc(&bingham_ori,              bingham_ori_space),       "allocating bingham distribution orientations");
  cu_lerr(cudaMalloc(&beam_to_domain,           beam_to_domain_space),    "allocating bingham distribution orientations");

  cu_lerr(cudaMalloc(&h_d,                      hkl_space),               "allocating hkls");
  cu_lerr(cudaMalloc(&chis,                     chis_space),              "allocating chis");
  cu_lerr(cudaMalloc(&phis,                     phis_space),              "allocating phis");
  cu_lerr(cudaMalloc(&res,                      res_space),               "allocating summed weights");
  cu_lerr(cudaMalloc(&pm_gpu.hkl_sizes,         permutations_space),      "allocating n_hkls");

  cu_lerr(cudaMemcpy(base_orientation_samples, ffidata.ori_samples,    base_orientations_space, cudaMemcpyHostToDevice), "copying orientation samples to device");
  cu_lerr(cudaMemcpy(chis,                     ffidata.chis,           chis_space,           cudaMemcpyHostToDevice), "copying chis to device");
  cu_lerr(cudaMemcpy(phis,                     ffidata.phis,           phis_space,           cudaMemcpyHostToDevice), "copying phis to device");
  cu_lerr(cudaMemcpy(h_d,                      ffidata.hkls,           hkl_space,            cudaMemcpyHostToDevice), "copying hkls to device");
  cu_lerr(cudaMemcpy(res,                      i_hkls,                 res_space,            cudaMemcpyHostToDevice), "copying i_hkls to device");
  cu_lerr(cudaMemcpy(pm_gpu.hkl_sizes,         permutations.hkl_sizes, permutations_space,   cudaMemcpyHostToDevice), "copying hkl_sizes to device");
  // clang-format on

  // TODO: use mallocAsync and MemcpyAsync to perform operations while the rotated orientations are computed

  // precompute_alignment_transformations(float *chis, float *phis, Quaternion *alignment_transformations, size_t
  // n_chis, size_t n_phis)
  launch_kernel_sensibly_no_shmem(precompute_alignment_transformations, "precomputing alignment transformations",
                                  stride, chis, phis, alignment_tf, ffidata.n_chis, ffidata.n_phis);

  // precompute_beam_to_bingham(Quaternion *alignment_transformations, Quaternion *bingham_orientations, Quaternion
  // *beam_to_bingham, size_t stride, size_t n_permutations)
  launch_kernel_sensibly_no_shmem(precompute_beam_to_bingham, "precomputing beam to bingham transformations",
                                  permutations.n_permutations * stride, alignment_tf, bingham_ori, beam_to_bingham,
                                  stride, permutations.n_permutations);

  // transform_quaternions(Quaternion *beam_to_bingham, Quaternion *base_orientation_samples, Quaternion *dst, size_t
  // stride, size_t ori_samples_per_alignment, size_t n_permutations)
  launch_kernel_sensibly_no_shmem(transform_quaternions, "computing tranformed orientations",
                                  beam_to_domain_space / sizeof(Quaternion), // array size
                                  beam_to_bingham,                           // beam to bingham
                                  base_orientation_samples,                  // base orientation
                                  beam_to_domain,                            // target beam to domain
                                  stride, ffidata.n_ori_per_alignment, permutations.n_permutations);

  // normalize_hkls(Vec3 *h, size_t n_hkls_tot)
  launch_kernel_sensibly_no_shmem(normalize_hkls, "normalizing hkl vectors", ffidata.n_hkls_tot, h_d,
                                  ffidata.n_hkls_tot);

  // // compute_single_hkl_ori_weight(Quaternion *q, Vec3 *h, float *w, Permutations permutations, float kappa, size_t
  // // total_ori_samples, size_t stride_in_alignments, size_t n_ori_per_alignment, size_t n_hkls_tot)
  // launch_kernel_sensibly_no_shmem(compute_single_hkl_ori_weight, "computing weight elements",
  //                                 partial_weights_space / sizeof(float), q_d, h_d, w_d, pm_gpu, kappa,
  //                                 stride * ffidata.n_ori_per_alignment * permutations.n_permutations, stride,
  //                                 ffidata.n_ori_per_alignment, ffidata.n_hkls_tot);

  // // reduce_weights_per_hkl_kde(float *w, float *results, float norm_const, size_t n_hkls, size_t n_ori_samples
  // launch_kernel_sensibly_no_shmem(reduce_weights_per_hkl_kde, "reducing weights using kde", res_space /
  // sizeof(float),
  //                                 w_d, res, norm_const, ffidata.n_hkls_tot, ffidata.n_ori_per_alignment);

  launch_kernel_sensibly_no_shmem(compute_hkl_weight, "computing hkl weights",
                                  res_space / sizeof(float), // array size
                                  beam_to_domain,            // beam to domain
                                  h_d,                       // hkls device
                                  res,                       // weight results
                                  pm_gpu, kappa, norm_const,
                                  stride * ffidata.n_ori_per_alignment * permutations.n_permutations, stride,
                                  ffidata.n_ori_per_alignment, ffidata.n_hkls_tot);

  snprintf(tmp_str_buf, sizeof(tmp_str_buf), "copying %ld hkl weights (%.2f MiB) to cpu", res_space / sizeof(float),
           (float)res_space / 1e6);
  infof(tmp_str_buf);
  cu_lerr(cudaMemcpy(i_hkls, res, res_space, cudaMemcpyDeviceToHost), "copying resuts to host");

  cudaFree(base_orientation_samples);
  cudaFree(alignment_tf);
  cudaFree(bingham_ori);
  cudaFree(beam_to_bingham);
  cudaFree(beam_to_domain);
  cudaFree(q_d);
  cudaFree(h_d);
  // cudaFree(w_d);
  cudaFree(res);
  cudaFree(chis);
  cudaFree(phis);

  return true;
}
}
