#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
extern "C" {
#include <cassert>
#include <stdio.h>

const float abstol = 1e-3;

typedef struct {
  float pos;
  float intensity;
} Peak;

typedef struct {
  float weight, fwhm, eta;
} PeakInfo;

typedef struct {
  float *intensity;
  float *pos;
  float *weight;
  float *fwhm;
  float *eta;
  size_t n_peaks_tot;
} PeakSOA;

typedef struct {
  size_t start_idx;
  size_t n_peaks;
} CUDAPattern;

#define TAU 3.1415926535897932384626433832795028841972f * 2.0f
#define PI 3.1415926535897932384626433832795028841972f
#define log_cuda_err(ret, msg)                                                 \
  if (ret != cudaSuccess) {                                                    \
    fprintf(stderr, "%s:%d: CUDA Error %d while %s: %s\n", __FILE__, __LINE__, \
            ret, msg, cudaGetErrorString(ret));                                \
    return false;                                                              \
  }

__device__ float gauss(float dx, float sigma) {
  return expf(-0.5f * powf(dx / sigma, 2.0f)) / sqrtf(TAU * powf(sigma, 2.0f));
}

__device__ float lorentz(float dx, float gamma) {
  return 1.0f / ((1.0f + powf(dx / gamma, 2)) * PI * gamma);
}

__device__ float pseudo_voigt(float dx, float eta, float fwhm) {
  float two_sqrt_ln_2 = 2.0f * sqrtf(logf(2.0f) * 2.0f);
  float sigma = (1.0f / two_sqrt_ln_2) * fwhm;
  float gamma = fwhm / 2.0f;
  return eta * lorentz(dx, gamma) + (1.0f - eta) * gauss(dx, sigma);
}

__device__ float render_peak_into_pattern(float pos, float intensity,
                                           float eta, float weight,
                                           float fwhm, float *intensities,
                                           float *two_thetas, size_t pat_len) {

  size_t midpoint =
      (size_t)((pos - two_thetas[0]) /
               (two_thetas[pat_len - 1] - two_thetas[0]) * (float)pat_len);

  size_t i = midpoint;
  if (i > pat_len - 1) {
    i = pat_len - 1;
  }

  // left half
  while (true) {
    float dx = two_thetas[i] - pos;
    float di = weight * intensity * pseudo_voigt(dx, eta, fwhm);
    if (di < abstol) {
      break;
    }

    intensities[i] += di;
    if (i == 0) {
      break;
    }
    i -= 1;
  }

  // right half
  i = midpoint + 1;
  while (i < pat_len) {
    float dx = two_thetas[i] - pos;
    float di = weight * intensity * pseudo_voigt(dx, eta, fwhm);
    if (di < abstol) {
      break;
    }

    intensities[i] += di;
    i += 1;
  }
}

__global__ void
discretize_kernel_single_pattern(PeakSOA soa, CUDAPattern *pat_info,
                                 float *intensities, float *two_thetas,
                                 size_t n_patterns, size_t pat_len) {
  int pattern_idx = blockIdx.x;
  if (pattern_idx > n_patterns) {
    return;
  }
  CUDAPattern pat = pat_info[pattern_idx];
  size_t chunk_start = pat_len * pattern_idx + blockDim.x * threadIdx.x;

  for (size_t peak_index = pat.start_idx;
       peak_index < pat.n_peaks + pat.start_idx; ++peak_index) {
    float weight = soa.weight[peak_index];
    float pos = soa.pos[peak_index];
    float intensity = soa.intensity[peak_index];
    float eta = soa.eta[peak_index];
    float fwhm = soa.fwhm[peak_index];
    render_peak_into_pattern(pos, intensity, eta, weight, fwhm,
                             &intensities[chunk_start], two_thetas, pat_len);
  }
}

__global__ void discretize_kernel(PeakSOA soa, CUDAPattern *pat_info,
                                  float *intensities, float *two_thetas,
                                  size_t n_patterns, size_t pat_len) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  size_t pattern_idx = tid / pat_len;
  CUDAPattern pat = pat_info[pattern_idx];
  float pat_pos = two_thetas[tid - pattern_idx * pat_len];

  float delta_intens = 0.0;
  for (size_t peak_index = pat.start_idx;
       peak_index < pat.n_peaks + pat.start_idx; ++peak_index) {
    float dx = pat_pos - soa.pos[peak_index];
    float pv = pseudo_voigt(dx, soa.eta[peak_index], soa.fwhm[peak_index]);
    delta_intens += soa.weight[peak_index] * soa.intensity[peak_index] * pv;
  }

  intensities[tid] = delta_intens;
}

bool discretize_peaks(PeakSOA peaks_soa, CUDAPattern *pat_info,
                      float *intensities, float *two_thetas,
                      size_t n_patterns, size_t pat_len) {
  float *intensities_d, *two_thetas_d;
  float *peaks_d;
  CUDAPattern *patterns_d;

  // clang-format off
  cudaError_t ret = cudaMalloc(&two_thetas_d, pat_len * sizeof(float));
  log_cuda_err(ret, "allocating two_thetas buffer");
  ret = cudaMalloc(&intensities_d, n_patterns * pat_len * sizeof(float));
  log_cuda_err(ret, "allocating intensities buffer");
  ret = cudaMalloc(&patterns_d, n_patterns * sizeof(CUDAPattern));
  log_cuda_err(ret, "allocating pattern buffer");

  static_assert(sizeof(PeakSOA) == 6 * sizeof(size_t), "Number of Components in PeaksSOA has changed");
  cudaMalloc(&peaks_d, 5 * sizeof(float) * peaks_soa.n_peaks_tot);

  ret = cudaMemcpy(&peaks_d[0 * peaks_soa.n_peaks_tot], peaks_soa.intensity, sizeof(float) * peaks_soa.n_peaks_tot, cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying peaks_soa to device");
  ret = cudaMemcpy(&peaks_d[1 * peaks_soa.n_peaks_tot],       peaks_soa.pos, sizeof(float) * peaks_soa.n_peaks_tot, cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying peaks_soa to device");
  ret = cudaMemcpy(&peaks_d[2 * peaks_soa.n_peaks_tot],    peaks_soa.weight, sizeof(float) * peaks_soa.n_peaks_tot, cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying peaks_soa to device");
  ret = cudaMemcpy(&peaks_d[3 * peaks_soa.n_peaks_tot],      peaks_soa.fwhm, sizeof(float) * peaks_soa.n_peaks_tot, cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying peaks_soa to device");
  ret = cudaMemcpy(&peaks_d[4 * peaks_soa.n_peaks_tot],       peaks_soa.eta, sizeof(float) * peaks_soa.n_peaks_tot, cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying peaks_soa to device");
  PeakSOA peak_soa_d = (PeakSOA){
    .intensity = &peaks_d[0 * peaks_soa.n_peaks_tot],
    .pos = &peaks_d[1 * peaks_soa.n_peaks_tot],
    .weight = &peaks_d[2 * peaks_soa.n_peaks_tot],
    .fwhm = &peaks_d[3 * peaks_soa.n_peaks_tot],
    .eta= &peaks_d[4 * peaks_soa.n_peaks_tot],
    .n_peaks_tot = peaks_soa.n_peaks_tot,
  };

  ret = cudaMemcpy(patterns_d, pat_info, n_patterns * sizeof(CUDAPattern), cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying patterns to device");
  ret = cudaMemcpy(two_thetas_d, two_thetas, pat_len * sizeof(float), cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying two_thetas to device");
  // clang-format on

  int block_size = 0;
  int min_grid_size = 0;
  int array_count = n_patterns * pat_len;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                     (void *)discretize_kernel, 0, array_count);
  int grid_size = (array_count + block_size - 1) / block_size;

  discretize_kernel<<<grid_size, block_size>>>(
      peak_soa_d, patterns_d, intensities_d, two_thetas_d, n_patterns,
      pat_len);

  // discretize_kernel_single_pattern<<<grid_size, block_size>>>(
  //     peak_soa_d, patterns_d, intensities_d, two_thetas_d, n_patterns, pat_len);
  ret = cudaPeekAtLastError();
  log_cuda_err(ret, "launching discretization kernel");
  cudaDeviceSynchronize();

  ret =
      cudaMemcpy(intensities, intensities_d,
                 n_patterns * pat_len * sizeof(float), cudaMemcpyDeviceToHost);
  log_cuda_err(ret, "copying intensities from device to host");

  cudaFree(peaks_d);
  cudaFree(patterns_d);
  cudaFree(two_thetas_d);
  cudaFree(intensities_d);
  return true;
}

} // extern "C"
