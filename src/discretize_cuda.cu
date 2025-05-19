#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
extern "C" {
#include <cassert>
#include <stdio.h>

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

typedef enum {
  None,
  Exponential,
  Polynomial,
} BkgKind;

#define TAU 3.1415926535897932384626433832795028841972f * 2.0f
#define PI 3.1415926535897932384626433832795028841972f
#define log_cuda_err(ret, msg)                                                 \
  if (ret != cudaSuccess) {                                                    \
    fprintf(stderr, "%s:%d: CUDA Error %d while %s: %s\n", __FILE__, __LINE__, \
            ret, msg, cudaGetErrorString(ret));                                \
    fflush(stderr);                                                            \
    return false;                                                              \
  }

__device__ float gauss(float dx, float sigma) {
  return expf(-0.5f * powf(dx / sigma, 2.0f)) / sqrtf(TAU * powf(sigma, 2.0f));
}

__device__ float lorentz(float dx, float gamma) {
  return 1.0f / fma(powf(dx / gamma, 2.0f), PI * gamma, 1.0f);
}

__device__ float pseudo_voigt(float dx, float eta, float fwhm) {
  float two_sqrt_ln_2 = 2.0f * sqrtf(logf(2.0f) * 2.0f);
  float sigma = (1.0f / two_sqrt_ln_2) * fwhm;
  float gamma = fwhm / 2.0f;
  return eta * lorentz(dx, gamma) + (1.0f - eta) * gauss(dx, sigma);
}

__global__ void render_peaks(PeakSOA soa, CUDAPattern *pat_info,
                             float *intensities, float *two_thetas,
                             size_t n_patterns, size_t pat_len) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n_patterns * pat_len)
    return;
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

__global__ void render_exp_bkg(float *intensities, float *two_thetas,
                               float *bkg_slope, float *scales, size_t pat_len,
                               size_t n_patterns) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= pat_len * n_patterns)
    return;
  size_t pattern_idx = tid / pat_len;
  float pat_pos = two_thetas[tid - pattern_idx * pat_len];

  float slope = bkg_slope[pattern_idx];

  intensities[tid] += scales[pattern_idx] * expf(pat_pos * slope);
}

__global__ void render_poly_bkg(float *intensities, float *two_thetas,
                                float *coef, float *scales, size_t pat_len,
                                size_t n_patterns, size_t degree) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= pat_len * n_patterns)
    return;
  size_t pattern_idx = tid / pat_len;
  float pat_pos =
      (float)(tid - pattern_idx * pat_len) / (float)pat_len * 2.0f - 1.0;
  float *local_coef = &coef[pattern_idx * degree];

  float di = 0.0f;
  for (size_t power = 0; power < degree; ++power) {
    di += local_coef[power] * powf(pat_pos, power);
  }

  intensities[tid] += di * scales[pattern_idx];
}

bool render_backgrounds(float *intensities_d, float *two_thetas_d,
                        BkgKind background_kind, float *bkg_data,
                        size_t bkg_degree_if_poly,
                        float *bkg_scales_if_not_none, size_t n_patterns,
                        size_t pat_len, int grid_size, int block_size) {
  cudaError_t ret;
  switch (background_kind) {
  case None:
    // all good, do nothing
    break;

  case Exponential: {
    float *bkg_slope_d, *bkg_scales_d;
    ret = cudaMalloc(&bkg_slope_d, sizeof(float) * n_patterns);
    log_cuda_err(ret, "allocating background info buffer");
    ret = cudaMalloc(&bkg_scales_d, sizeof(float) * n_patterns);
    log_cuda_err(ret, "allocating background scale buffer");

    ret = cudaMemcpy(bkg_slope_d, bkg_data, sizeof(float) * n_patterns,
                     cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying background info to device");
    ret = cudaMemcpy(bkg_scales_d, bkg_scales_if_not_none,
                     sizeof(float) * n_patterns, cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying background scales to device");

    render_exp_bkg<<<grid_size, block_size>>>(intensities_d, two_thetas_d,
                                              bkg_slope_d, bkg_scales_d,
                                              pat_len, n_patterns);

    cudaFree(bkg_slope_d);
    cudaFree(bkg_scales_d);
    break;
  }

  case Polynomial: {
    float *bkg_coefs_d, *bkg_scales_d;
    ret = cudaMalloc(&bkg_coefs_d,
                     sizeof(float) * n_patterns * bkg_degree_if_poly);
    log_cuda_err(ret, "allocating background info buffer");
    // FIX: for some reason, we break here at 100 patterns
    ret = cudaMalloc(&bkg_scales_d, sizeof(float) * n_patterns);
    log_cuda_err(ret, "allocating background scale buffer");

    ret = cudaMemcpy(bkg_coefs_d, bkg_data,
                     sizeof(float) * n_patterns * bkg_degree_if_poly,
                     cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying background info to device");
    ret = cudaMemcpy(bkg_scales_d, bkg_scales_if_not_none,
                     sizeof(float) * n_patterns, cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying background scales to device");

    render_poly_bkg<<<grid_size, block_size>>>(
        intensities_d, two_thetas_d, bkg_coefs_d, bkg_scales_d, pat_len,
        n_patterns, bkg_degree_if_poly);

    ret = cudaPeekAtLastError();
    log_cuda_err(ret, "Error in dispatch or execution of render_poly_bkg");

    cudaFree(bkg_coefs_d);
    break;
  }

  default: {
    fprintf(stderr,
            "Error rendering background via CUDA backend. Unknown background "
            "kind: %d\n. Crashing...\n",
            background_kind);
    break;
  }
  }
  return true;
}

// TODO: find a more efficient way to do this!
//       maybe a fancy reduction kernel is in better, but this works for now
__global__ void get_extrema(float *intensities, size_t n_patterns,
                            size_t pat_len, float *all_minima,
                            float *all_maxima) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n_patterns)
    return;

  size_t p0 = tid * pat_len;
  float vmin = intensities[p0];
  float vmax = intensities[p0];
  for (size_t off = 1; off < pat_len; ++off) {
    vmin = fmin(vmin, intensities[p0 + off]);
    vmax = fmax(vmax, intensities[p0 + off]);
  }
  all_minima[tid] = vmin;
  all_maxima[tid] = vmax;
}

__global__ void normalize(float *intensities, size_t n_patterns, size_t pat_len,
                          float *all_minima, float *all_maxima) {

  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n_patterns * pat_len)
    return;
  size_t pattern_idx = tid / pat_len;
  intensities[tid] = (intensities[tid] - all_minima[pattern_idx]) /
                     (all_maxima[pattern_idx] - all_minima[pattern_idx]);
}

bool normalize_patterns(float *intensities, size_t n_patterns, size_t pat_len) {
  float *all_minima, *all_maxima;
  cudaError_t ret = cudaMalloc(&all_minima, 2 * n_patterns * sizeof(float));
  log_cuda_err(ret, "allocating extrema buffer for normalization");
  all_maxima = &all_minima[n_patterns];

  {
    int block_size = 0;
    int min_grid_size = 0;
    int array_count = n_patterns;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       (void *)get_extrema, 0, array_count);
    int grid_size = (array_count + block_size - 1) / block_size;

    get_extrema<<<grid_size, block_size>>>(intensities, n_patterns, pat_len,
                                           all_minima, all_maxima);
  }

  {
    int block_size = 0;
    int min_grid_size = 0;
    int array_count = n_patterns * pat_len;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       (void *)normalize, 0, array_count);
    int grid_size = (array_count + block_size - 1) / block_size;

    normalize<<<grid_size, block_size>>>(intensities, n_patterns, pat_len,
                                         all_minima, all_maxima);
  }

  cudaFree(all_minima);
  return true;
}

#define check_error(val)                                                       \
  do {                                                                         \
    if (!val) {                                                                \
      return_value = val;                                                      \
      goto defer;                                                              \
    }                                                                          \
  } while (0)

bool render_peaks_and_background(PeakSOA peaks_soa, CUDAPattern *pat_info,
                                 float *intensities, float *two_thetas,
                                 size_t n_patterns, size_t pat_len,
                                 BkgKind background_kind, float *bkg_data,
                                 size_t bkg_degree_if_poly,
                                 float *bkg_scales_if_not_none,
                                 bool normalize) {
  float *intensities_d, *two_thetas_d;
  float *peaks_d;
  CUDAPattern *patterns_d;
  bool return_value = true;

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
                                     (void *)render_peaks, 0, array_count);
  int grid_size = (array_count + block_size - 1) / block_size;

  render_peaks<<<grid_size, block_size>>>(peak_soa_d, patterns_d, intensities_d,
                                          two_thetas_d, n_patterns, pat_len);

  ret = cudaPeekAtLastError();
  log_cuda_err(ret, "launching discretization kernel");

  bool bkg_ok =
      render_backgrounds(intensities_d, two_thetas_d, background_kind, bkg_data,
                         bkg_degree_if_poly, bkg_scales_if_not_none, n_patterns,
                         pat_len, grid_size, block_size);
  check_error(bkg_ok);
  if (normalize) {
    bool normalize_ok = normalize_patterns(intensities_d, n_patterns, pat_len);
    check_error(normalize_ok);
  }

  ret = cudaDeviceSynchronize();
  log_cuda_err(ret, "synchronizing device after rendering");

  ret =
      cudaMemcpy(intensities, intensities_d,
                 n_patterns * pat_len * sizeof(float), cudaMemcpyDeviceToHost);
  log_cuda_err(ret, "copying intensities from "
                    "device to host");

defer:
  cudaFree(peaks_d);
  cudaFree(patterns_d);
  cudaFree(two_thetas_d);
  cudaFree(intensities_d);
  return return_value;
}

} // extern "C"
