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
  Quaternion *ori_samples;
  Vec3 *hkls;
  size_t total_ori_samples;
  size_t n_ori_per_alignment;
  size_t stride;
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

__global__ void normalize_hkls(Vec3 *h, size_t n_hkls_tot) {
  size_t hkl_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (hkl_idx >= n_hkls_tot)
    return;
  vec3_normalize(&h[hkl_idx]);
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

  // clang-format off
  size_t quaternion_space      = ffidata.total_ori_samples * sizeof(Quaternion);
  size_t permutations_space    = permutations.n_permutations * sizeof(size_t);

  size_t hkl_space             = ffidata.n_hkls_tot                                                * sizeof(Vec3);
  size_t res_space             = ffidata.n_hkls_tot * ffidata.stride                               * sizeof(float);
  size_t partial_weights_space = ffidata.n_hkls_tot * ffidata.stride * ffidata.n_ori_per_alignment * sizeof(float);
  // clang-format on

  snprintf(tmp_str_buf, sizeof(tmp_str_buf),
           "Allocating device memory for computing weights of %ld hkls for %ld orientations (%ld permutations).\n"
           "Quaternions:     %10.3f MiB\n"
           "HKL vectors:     %10.3f MiB\n"
           "Partial Weights: %10.3f MiB\n"
           "Weights:         %10.3f MiB\n"
           "HKL sizes:       %10.3f B\n"
           "Total:           %10.3f MiB",
           ffidata.n_hkls_tot, ffidata.n_ori_per_alignment, permutations.n_permutations, (float)quaternion_space / 1e6,
           (float)hkl_space / 1e6, (float)partial_weights_space / 1e6, (float)res_space / 1e6,
           (float)permutations_space, (float)(quaternion_space + partial_weights_space + hkl_space + res_space) / 1e6);

  debugf(tmp_str_buf);

  Quaternion *q_d;
  Vec3 *h_d;
  float *w_d;
  float *res;

  Permutations pm_gpu{.hkl_sizes = NULL, .n_permutations = permutations.n_permutations};

  // clang-format off
  cu_lerr(cudaMalloc(&q_d, quaternion_space),                "allocating quaternions");
  cu_lerr(cudaMalloc(&h_d, hkl_space),                       "allocating hkls");
  cu_lerr(cudaMalloc(&w_d, partial_weights_space),           "allocating weights");
  cu_lerr(cudaMalloc(&res, res_space),                       "allocating summed weights");
  cu_lerr(cudaMalloc(&pm_gpu.hkl_sizes, permutations_space), "allocating n_hkls");

  cu_lerr(cudaMemcpy(q_d,              ffidata.ori_samples,    quaternion_space,   cudaMemcpyHostToDevice), "copying orientation samples to device");
  cu_lerr(cudaMemcpy(h_d,              ffidata.hkls,           hkl_space,          cudaMemcpyHostToDevice), "copying hkls to device");
  cu_lerr(cudaMemcpy(res,              i_hkls,                 res_space,          cudaMemcpyHostToDevice), "copying i_hkls to device");
  cu_lerr(cudaMemcpy(pm_gpu.hkl_sizes, permutations.hkl_sizes, permutations_space, cudaMemcpyHostToDevice), "copying hkl_sizes to device");
  // clang-format on

  {
    debugf("Determining weight hkl normalization launch parameters");
    int block_size = 0;
    int min_grid_size = 0;
    int array_count = ffidata.n_hkls_tot;
    cu_lerr(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)normalize_hkls, 0, array_count),
            "Determining hkl normalization launch parameters");
    int grid_size = (array_count + block_size - 1) / block_size;
    debugf("launching hkl normalization");

    // clang-format off
    normalize_hkls<<<grid_size, block_size>>>(h_d, ffidata.n_hkls_tot);
    // clang-format on
    cu_lerr(cudaDeviceSynchronize(), "synchronizing device after hkl normalization");
  }

  {
    debugf("Determining weight element computation launch parameters");
    int block_size = 0;
    int min_grid_size = 0;
    int array_count = partial_weights_space / sizeof(float);
    cu_lerr(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)compute_single_hkl_ori_weight, 0,
                                               array_count),
            "Determining weight element computation launch parameters");
    int grid_size = (array_count + block_size - 1) / block_size;
    debugf("launching weight element computation");

    // clang-format off
    compute_single_hkl_ori_weight<<<grid_size, block_size>>>(
        q_d, h_d, w_d, 
        pm_gpu,
        kappa, 
        ffidata.total_ori_samples,
        ffidata.stride, 
        ffidata.n_ori_per_alignment,
        ffidata.n_hkls_tot
    );
    // clang-format on
    cu_lerr(cudaDeviceSynchronize(), "synchronizing device after per-orientation computation");
  }

  {
    debugf("Determining kde reduction launch parameters");
    int block_size = 0;
    int min_grid_size = 0;
    int array_count = res_space / sizeof(float);
    cu_lerr(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)reduce_weights_per_hkl_kde, 0,
                                               array_count),
            "Determining kde reduction launch parameters");
    int grid_size = (array_count + block_size - 1) / block_size;

    reduce_weights_per_hkl_kde<<<grid_size, block_size>>>(w_d, res, norm_const, ffidata.n_hkls_tot,
                                                          ffidata.n_ori_per_alignment);
    cu_lerr(cudaGetLastError(), "dispatching kde reduction kernel");

    snprintf(tmp_str_buf, sizeof(tmp_str_buf), "copying %ld hkl weights (%.2f MiB) to cpu", res_space / sizeof(float),
             (float)res_space / 1e6);
    infof(tmp_str_buf);
    cu_lerr(cudaMemcpy(i_hkls, res, res_space, cudaMemcpyDeviceToHost), "copying resuts to host");
    cu_lerr(cudaDeviceSynchronize(), "synchronizing device after memcopy to host");
  }

  cudaFree(res);
  cudaFree(q_d);
  cudaFree(pm_gpu.hkl_sizes);
  cudaFree(h_d);
  cudaFree(w_d);
  return true;
}
}
