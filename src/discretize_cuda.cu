#include <cstddef>
#include <cstdint>
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
  float *fwhm;
  float *eta;
  size_t n_peaks_tot;
} PeakSOA;

typedef struct {
  size_t start_idx;
  size_t n_peaks;
} CUDAPattern;

typedef enum {
  BkgNone,
  Exponential,
  Polynomial,
} BkgKind;

typedef enum {
  NoiseNone,
  Gaussian,
  Uniform,
} NoiseKind;

typedef struct {
  union {
    double *none;
    double *gaussian;
    struct {
      double *min;
      double *max;
    } uniform;
  } v;
  NoiseKind kind;
} Noise;

typedef void (*error_fn)(const char *, int, const char *, const char *);
typedef void (*info_fn)(const char *);
typedef void (*debug_fn)(const char *);
error_fn errf;
info_fn infof;
debug_fn debugf;

#define TAU 3.1415926535897932384626433832795028841972f * 2.0f
#define PI 3.1415926535897932384626433832795028841972f
#define log_cuda_err(ret, msg)                                                 \
  if (ret != cudaSuccess) {                                                    \
    errf(__FILE__, __LINE__, msg, cudaGetErrorString(ret));                    \
    fflush(stderr);                                                            \
    return false;                                                              \
  }

typedef struct {
  uint64_t s[4];
} Xoshiro256PlusPlus;

__device__ uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

/* Generate the next random float in the range [0, 1] using the Xoshiro256++
 * rng  as implemented by David Blackman and Sebastiano Vigna
 * from https://prng.di.unimi.it/xoshiro256plusplus.c
 *
 * The implementation here is adapted for use on the GPU
 */
__device__ double xoshiro256_plus_plus_next_double01(Xoshiro256PlusPlus *rng) {
  const uint64_t result = rotl(rng->s[0] + rng->s[3], 23) + rng->s[0];
  const uint64_t t = rng->s[1] << 17;

  rng->s[2] ^= rng->s[0];
  rng->s[3] ^= rng->s[1];
  rng->s[1] ^= rng->s[2];
  rng->s[0] ^= rng->s[3];

  rng->s[2] ^= t;

  rng->s[3] = rotl(rng->s[3], 45);

  // limit to [0, 1] by division using UINT64_MAX
  return (double)result / (double)UINT64_MAX;
}

/* generate a random float distributed according to the standard normal
 * distribution
 *
 * this function uses the xoshiro256++ algorithm to generate two uniform values
 * which are then used in the box-muller transform to generate a values
 * distributed according to the standard normal distribution.
 */
__device__ double
xoshiro256_plus_plus_box_muller_gaussian(Xoshiro256PlusPlus *rng) {
  const float u1 = xoshiro256_plus_plus_next_double01(rng);
  const float u2 = xoshiro256_plus_plus_next_double01(rng);
  return sqrt(-2.0 * log(u1)) * cos(TAU * u2);
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
    delta_intens += soa.intensity[peak_index] * pv;
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

typedef struct {
  double *min;
  double *max;
} UniformBounds;

__global__ void render_uniform_noise(float *intensities_d, UniformBounds bounds,
                                     Xoshiro256PlusPlus *rng_state,
                                     size_t pat_len, size_t n_patterns) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > n_patterns)
    return;
  const double lo = bounds.min[tid];
  const double hi = bounds.max[tid];

  Xoshiro256PlusPlus *rng = &rng_state[tid];

  for (size_t i = tid * pat_len; i < (tid + 1) * pat_len; ++i) {
    float noise =
        (float)(xoshiro256_plus_plus_next_double01(rng) * (hi - lo) + lo);
    if (tid == 0) {
      printf("%.2f\n", noise);
    }
    intensities_d[i] += noise;
  }
}

__global__ void render_gaussian_noise(float *intensities_d, double *sigmas,
                                      Xoshiro256PlusPlus *rng_state,
                                      size_t pat_len, size_t n_patterns) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > n_patterns)
    return;
  const double sigma = sigmas[tid];
  Xoshiro256PlusPlus *rng = &rng_state[tid];

  for (size_t i = tid * pat_len; i < (tid + 1) * pat_len; ++i) {
    intensities_d[i] += xoshiro256_plus_plus_box_muller_gaussian(rng) * sigma;
  }
}

bool render_noise(float *intensities_d, Noise noise, uint64_t *rng_state,
                  size_t pat_len, size_t n_patterns) {
  if (noise.kind == NoiseNone) {
    assert(rng_state == NULL);
    assert(noise.v.none == NULL);
    debugf("No noise specified");
    return true;
  }
  assert(rng_state);
  // assert that the data is not a nullpointer
  // (noise.v is a union, so checking one variant checks all)
  assert(noise.v.gaussian);

  int block_size = 0;
  int min_grid_size = 0;
  int array_count = n_patterns;

  debugf("Allocating noise RNG state");
  uint64_t *rng_state_d;
  double *noise_data_d;
  cudaError_t ret =
      cudaMalloc(&rng_state_d, sizeof(Xoshiro256PlusPlus) * n_patterns);
  log_cuda_err(ret, "allocating random state for noise generation");

  debugf("Initializing noise RNG state");
  ret = cudaMemcpy(rng_state_d, rng_state,
                   sizeof(Xoshiro256PlusPlus) * n_patterns,
                   cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying rng state to gpu during noise generation");

  switch (noise.kind) {
  case NoiseNone:
    assert(false && "unreachable, we early return");
  case Gaussian: {
    infof("rendering gaussian noise");
    // for gaussian noise, we only have standard deviations
    debugf("Allocating memory for standard deviations");
    ret = cudaMalloc(&noise_data_d, sizeof(double) * n_patterns);
    log_cuda_err(ret, "allocating device memory for gaussian noise sigmas");

    debugf("Initializing standard deviations");
    ret = cudaMemcpy(noise_data_d, noise.v.gaussian,
                     n_patterns * sizeof(double), cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying gaussian noise standard deviations to device");

    debugf("Determining gaussian noise kernel launch parameters");
    ret = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                             (void *)render_gaussian_noise, 0,
                                             array_count);
    log_cuda_err(ret, "finding cuda kernel launch configuration");
    int grid_size = (array_count + block_size - 1) / block_size;

    debugf("Launching gaussian noise kernel");
    render_gaussian_noise<<<grid_size, block_size>>>(
        intensities_d, noise_data_d, (Xoshiro256PlusPlus *)rng_state_d, pat_len,
        n_patterns);
  } break;
  case Uniform: {
    infof("Rendering uniform noise");
    debugf("Allocating memory for noise limits");
    ret = cudaMalloc(&noise_data_d, 2 * sizeof(double) * n_patterns);
    log_cuda_err(
        ret, "allocating device memory for uniform noise distribution limits");

    debugf("Initializing noise limits");
    ret = cudaMemcpy(noise_data_d, noise.v.uniform.min,
                     n_patterns * sizeof(double), cudaMemcpyHostToDevice);

    ret = cudaMemcpy(&noise_data_d[n_patterns], noise.v.uniform.max,
                     n_patterns * sizeof(double), cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying uniform maxima deviations to device");

    debugf("Determining uniform noise kernel launch parameters");
    ret = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                             (void *)render_uniform_noise, 0,
                                             array_count);
    log_cuda_err(ret, "finding cuda kernel launch configuration");
    int grid_size = (array_count + block_size - 1) / block_size;
    // bounds are stored the following way: all minima, then all maxima.
    UniformBounds bounds = {
        .min = &noise_data_d[0],
        .max = &noise_data_d[n_patterns],
    };
    debugf("Launching uniform noise kernel");
    render_uniform_noise<<<grid_size, block_size>>>(
        intensities_d, bounds, (Xoshiro256PlusPlus *)rng_state_d, pat_len,
        n_patterns);
  } break;
  default:
    assert(false && "unreachable: corrupted noise_kind");
  }

  cudaFree(noise_data_d);
  cudaFree(rng_state_d);

  ret = cudaDeviceSynchronize();
  log_cuda_err(ret, "synchronizing device after noise");

  return true;
}

bool render_backgrounds(float *intensities_d, float *two_thetas_d,
                        BkgKind background_kind, float *bkg_data,
                        size_t bkg_degree_if_poly,
                        float *bkg_scales_if_not_none, size_t n_patterns,
                        size_t pat_len, int grid_size, int block_size) {
  cudaError_t ret;
  switch (background_kind) {
  case BkgNone:
    debugf("No background specified");
    // all good, do nothing
    break;

  case Exponential: {
    infof("Rendering exponential background");
    debugf("Allocating memory for exponential background parameters");
    float *bkg_slope_d, *bkg_scales_d;
    ret = cudaMalloc(&bkg_slope_d, sizeof(float) * n_patterns);
    log_cuda_err(ret, "allocating background info buffer");
    ret = cudaMalloc(&bkg_scales_d, sizeof(float) * n_patterns);
    log_cuda_err(ret, "allocating background scale buffer");

    debugf("Initializing exponential background parameters");
    ret = cudaMemcpy(bkg_slope_d, bkg_data, sizeof(float) * n_patterns,
                     cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying background info to device");
    ret = cudaMemcpy(bkg_scales_d, bkg_scales_if_not_none,
                     sizeof(float) * n_patterns, cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying background scales to device");

    debugf("Launching exponential background kernel");
    render_exp_bkg<<<grid_size, block_size>>>(intensities_d, two_thetas_d,
                                              bkg_slope_d, bkg_scales_d,
                                              pat_len, n_patterns);

    cudaFree(bkg_slope_d);
    cudaFree(bkg_scales_d);
    break;
  }

  case Polynomial: {
    infof("Rendering polynomial background");
    debugf("Allocating memory for polynomial background parameters");
    float *bkg_coefs_d, *bkg_scales_d;
    ret = cudaMalloc(&bkg_coefs_d,
                     sizeof(float) * n_patterns * bkg_degree_if_poly);
    log_cuda_err(ret, "allocating background info buffer");
    // FIX: for some reason, we break here at 100 patterns
    ret = cudaMalloc(&bkg_scales_d, sizeof(float) * n_patterns);
    log_cuda_err(ret, "allocating background scale buffer");

    debugf("initializing polynomial background parameters");
    ret = cudaMemcpy(bkg_coefs_d, bkg_data,
                     sizeof(float) * n_patterns * bkg_degree_if_poly,
                     cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying background info to device");
    ret = cudaMemcpy(bkg_scales_d, bkg_scales_if_not_none,
                     sizeof(float) * n_patterns, cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying background scales to device");

    debugf("Launching polynomial background kernel");
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

  ret = cudaDeviceSynchronize();
  log_cuda_err(ret, "synchronizing device after background");
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
  infof("Normalizing patterns");

  debugf("Allocating memory for extrema");
  float *all_minima, *all_maxima;
  cudaError_t ret = cudaMalloc(&all_minima, 2 * n_patterns * sizeof(float));
  log_cuda_err(ret, "allocating extrema buffer for normalization");
  all_maxima = &all_minima[n_patterns];

  {
    debugf("determining get_extrema kernel launch parameters");
    int block_size = 0;
    int min_grid_size = 0;
    int array_count = n_patterns;
    cudaError_t ret = cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, (void *)get_extrema, 0, array_count);
    log_cuda_err(ret, "finding cuda kernel launch configuration");
    int grid_size = (array_count + block_size - 1) / block_size;

    debugf("launching get_extrema kernel");
    get_extrema<<<grid_size, block_size>>>(intensities, n_patterns, pat_len,
                                           all_minima, all_maxima);
  }

  {
    debugf("determining normalization kernel launch parameters");
    int block_size = 0;
    int min_grid_size = 0;
    int array_count = n_patterns * pat_len;
    cudaError_t ret = cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, (void *)normalize, 0, array_count);
    log_cuda_err(ret, "finding cuda kernel launch configuration");
    int grid_size = (array_count + block_size - 1) / block_size;

    debugf("launching normalization kernel");
    normalize<<<grid_size, block_size>>>(intensities, n_patterns, pat_len,
                                         all_minima, all_maxima);
  }

  cudaFree(all_minima);
  return true;
}

bool render_peaks_and_background(PeakSOA peaks_soa, CUDAPattern *pat_info,
                                 float *intensities, float *two_thetas,
                                 size_t n_patterns, size_t pat_len, Noise noise,
                                 uint64_t *rng_state, BkgKind background_kind,
                                 float *bkg_data, size_t bkg_degree_if_poly,
                                 float *bkg_scales_if_not_none, bool normalize,
                                 error_fn errfn, info_fn infofn,
                                 debug_fn debugfn) {
  errf = errfn;
  infof = infofn;
  debugf = debugfn;
  infof("Beginning CUDA rendering");

  float *intensities_d, *two_thetas_d;
  float *peaks_d;
  CUDAPattern *patterns_d;
  bool return_value = true;

  debugf("Allocating CUDA memory");
  // clang-format off
  cudaError_t ret = cudaMalloc(&two_thetas_d, pat_len * sizeof(float));
  log_cuda_err(ret, "allocating two_thetas buffer");
  ret = cudaDeviceSynchronize();
  log_cuda_err(ret, "synchronizing device after allocating two_thetas");

  ret = cudaMalloc(&intensities_d, n_patterns * pat_len * sizeof(float));
  log_cuda_err(ret, "allocating intensities buffer");
  ret = cudaDeviceSynchronize();
  log_cuda_err(ret, "synchronizing device after allocating intensities");

  ret = cudaMalloc(&patterns_d, n_patterns * sizeof(CUDAPattern));
  log_cuda_err(ret, "allocating pattern buffer");
  ret = cudaDeviceSynchronize();
  log_cuda_err(ret, "synchronizing device after allocating patterns");

  static_assert(sizeof(PeakSOA) == 5 * sizeof(size_t), "Number of Components in PeaksSOA has changed");
  ret = cudaMalloc(&peaks_d, 4 * sizeof(float) * peaks_soa.n_peaks_tot);
  log_cuda_err(ret, "allocating peak info buffer");
  ret = cudaDeviceSynchronize();
  log_cuda_err(ret, "synchronizing device after allocating peak info");

  if (!patterns_d) {
    errf("", 0, "allocation failed", "device pointer to peak info is null");
    return false;
  }
  debugf("Copying data to GPU");
  ret = cudaMemcpy(&peaks_d[0 * peaks_soa.n_peaks_tot], peaks_soa.intensity, sizeof(float) * peaks_soa.n_peaks_tot, cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying peak intensities to device");
  ret = cudaMemcpy(&peaks_d[1 * peaks_soa.n_peaks_tot],       peaks_soa.pos, sizeof(float) * peaks_soa.n_peaks_tot, cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying peak positions to device");
  ret = cudaMemcpy(&peaks_d[2 * peaks_soa.n_peaks_tot],      peaks_soa.fwhm, sizeof(float) * peaks_soa.n_peaks_tot, cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying peak fwhms to device");
  ret = cudaMemcpy(&peaks_d[3 * peaks_soa.n_peaks_tot],       peaks_soa.eta, sizeof(float) * peaks_soa.n_peaks_tot, cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying peak etas to device");
  PeakSOA peak_soa_d = (PeakSOA){
    .intensity = &peaks_d[0 * peaks_soa.n_peaks_tot],
    .pos = &peaks_d[1 * peaks_soa.n_peaks_tot],
    .fwhm = &peaks_d[2 * peaks_soa.n_peaks_tot],
    .eta= &peaks_d[3 * peaks_soa.n_peaks_tot],
    .n_peaks_tot = peaks_soa.n_peaks_tot,
  };

  ret = cudaMemcpy(patterns_d, pat_info, n_patterns * sizeof(CUDAPattern), cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying patterns to device");
  ret = cudaMemcpy(two_thetas_d, two_thetas, pat_len * sizeof(float), cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying two_thetas to device");
  // clang-format on

  debugf("Determining peak rendering kernel launch parameters");
  int block_size = 0;
  int min_grid_size = 0;
  int array_count = n_patterns * pat_len;
  ret = cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, (void *)render_peaks, 0, array_count);
  log_cuda_err(ret, "finding cuda kernel launch configuration");
  assert(block_size > 0 && "block size must be larger than 0");
  int grid_size = (array_count + block_size - 1) / block_size;

  infof("Rendering peaks");
  render_peaks<<<grid_size, block_size>>>(peak_soa_d, patterns_d, intensities_d,
                                          two_thetas_d, n_patterns, pat_len);

  ret = cudaPeekAtLastError();
  log_cuda_err(ret, "launching discretization kernel");

  ret = cudaDeviceSynchronize();
  log_cuda_err(ret, "synchronizing device after peak rendering");

  if (!render_noise(intensities_d, noise, rng_state, pat_len, n_patterns)) {
    return false;
  }

  if (!render_backgrounds(intensities_d, two_thetas_d, background_kind,
                          bkg_data, bkg_degree_if_poly, bkg_scales_if_not_none,
                          n_patterns, pat_len, grid_size, block_size)) {
    return false;
  }

  if (normalize) {
    if (!normalize_patterns(intensities_d, n_patterns, pat_len)) {
      return false;
    }

    ret = cudaDeviceSynchronize();
    log_cuda_err(ret, "synchronizing device after normalization");
  }

  infof("Copying patterns from GPU to CPU");
  ret =
      cudaMemcpy(intensities, intensities_d,
                 n_patterns * pat_len * sizeof(float), cudaMemcpyDeviceToHost);
  log_cuda_err(ret, "copying intensities from "
                    "device to host");

  cudaFree(peaks_d);
  cudaFree(patterns_d);
  cudaFree(two_thetas_d);
  cudaFree(intensities_d);
  return return_value;
}

} // extern "C"
