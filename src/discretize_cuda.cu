#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
extern "C" {
#include <cassert>
#include <stdio.h>

const double abstol = 1e-3;

typedef struct {
  double pos;
  double intensity;
} Peak;

typedef struct {
  double weight, fwhm, eta;
} PeakInfo;

typedef struct {
  Peak *peaks;
  PeakInfo *peak_info;
  size_t n_peaks;
} CUDAPattern;

typedef struct {
  double *wavelengths;
  size_t n_wavelengths;
} RenderMeta;

#define TAU 3.1415926535897932384626433832795028841972 * 2.0
#define PI 3.1415926535897932384626433832795028841972
#define log_cuda_err(ret, msg)                                                 \
  if (ret != cudaSuccess) {                                                    \
    fprintf(stderr, "%s:%d: CUDA Error %d while %s: %s\n", __FILE__, __LINE__, \
            ret, msg, cudaGetErrorString(ret));                                \
    return false;                                                              \
  }

__device__ double gauss(double dx, double sigma) {
  return exp(-0.5 * pow(dx / sigma, 2)) / sqrt(TAU * pow(sigma, 2));
}

__device__ double lorentz(double dx, double gamma) {
  return 1.0 / ((1.0 + pow(dx / gamma, 2)) * PI * gamma);
}

__device__ double pseudo_voigt(double dx, double eta, double fwhm) {
  double two_sqrt_ln_2 = 2.0 * sqrt(log(2.0) * 2.0);
  double sigma = (1.0 / two_sqrt_ln_2) * fwhm;
  double gamma = fwhm / 2.0;
  return eta * lorentz(dx, gamma) + (1.0 - eta) * gauss(dx, sigma);
}

__global__ void discretize_kernel_single_pattern(CUDAPattern *pat_info,
                                                 double *intensities,
                                                 double *two_thetas,
                                                 size_t n_patterns,
                                                 size_t pat_len) {
  int pattern_idx = blockIdx.x * blockDim.x + threadIdx.x;
  CUDAPattern pat = pat_info[pattern_idx];
  size_t chunk_start = pat_len * pattern_idx;

  for (size_t peak_index = 0; peak_index < pat.n_peaks; ++peak_index) {
    Peak peak = pat.peaks[peak_index];
    PeakInfo peak_info = pat.peak_info[peak_index];

    size_t midpoint =
        (size_t)((peak.pos - two_thetas[0]) /
                 (two_thetas[pat_len - 1] - two_thetas[0]) * (double)pat_len);

    size_t i = midpoint;
    if (i > pat_len - 1) {
      i = pat_len - 1;
    }

    // left half
    while (true) {
      double two_theta = two_thetas[i];
      double dx = two_theta - peak.pos;
      double di = peak_info.weight * peak.intensity *
                  pseudo_voigt(dx, peak_info.eta, peak_info.fwhm);
      if (di < abstol) {
        break;
      }

      intensities[chunk_start + i] += di;
      if (i == 0) {
        break;
      }
      i -= 1;
    }

    // right half
    i = midpoint + 1;
    while (i < pat_len) {
      double two_theta = two_thetas[i];
      double dx = two_theta - peak.pos;
      double di = peak_info.weight * peak.intensity *
                  pseudo_voigt(dx, peak_info.eta, peak_info.fwhm);
      if (di < abstol) {
        break;
      }

      intensities[chunk_start + i] += di;
      i += 1;
    }
  }
}

__global__ void discretize_kernel(CUDAPattern *pat_info, double *intensities,
                                  double *two_thetas, size_t n_patterns,
                                  size_t pat_len) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  size_t pattern_idx = tid / pat_len;
  CUDAPattern pat = pat_info[pattern_idx];
  double pat_pos = two_thetas[tid - pattern_idx * pat_len];

  double delta_intens = 0.0;
  for (size_t peak_index = 0; peak_index < pat.n_peaks; ++peak_index) {
    double dx = pat_pos - pat.peaks[peak_index].pos;
    double pv = pseudo_voigt(dx, pat.peak_info[peak_index].eta,
                             pat.peak_info[peak_index].fwhm);
    delta_intens +=
        pat.peak_info[peak_index].weight * pat.peaks[peak_index].intensity * pv;
  }

  intensities[tid] = delta_intens;
}

bool discretize_peaks(CUDAPattern *pat_info, double *intensities,
                      double *two_thetas, size_t n_patterns, size_t pat_len) {
  double *intensities_d, *two_thetas_d;
  Peak *peaks_d;
  PeakInfo *peak_info_d;
  CUDAPattern *patterns_d;

  // prepare cuda memory
  size_t n_peaks_total = 0;
  for (size_t i = 0; i < n_patterns; ++i) {
    n_peaks_total += pat_info[i].n_peaks;
  }

  cudaError_t ret = cudaMalloc(&two_thetas_d, pat_len * sizeof(double));
  log_cuda_err(ret, "allocating two_thetas buffer");

  ret = cudaMalloc(&intensities_d, n_patterns * pat_len * sizeof(double));
  log_cuda_err(ret, "allocating intensities buffer");

  ret = cudaMalloc(&peaks_d, n_peaks_total * sizeof(Peak));
  log_cuda_err(ret, "allocating peaks buffer");

  cudaMalloc(&peak_info_d, n_peaks_total * sizeof(PeakInfo));
  log_cuda_err(ret, "allocating peak_info buffer");

  cudaMalloc(&patterns_d, n_patterns * sizeof(CUDAPattern));
  log_cuda_err(ret, "allocating patterns buffer");

  CUDAPattern *gpu_pat_info =
      (CUDAPattern *)malloc(n_patterns * sizeof(CUDAPattern));
  assert(gpu_pat_info != NULL);

  size_t n_peaks_running = 0;
  for (size_t i = 0; i < n_patterns; ++i) {
    ret =
        cudaMemcpy(&peaks_d[n_peaks_running], pat_info[i].peaks,
                   pat_info[i].n_peaks * sizeof(Peak), cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying peaks to device");

    ret = cudaMemcpy(&peak_info_d[n_peaks_running], pat_info[i].peak_info,
                     pat_info[i].n_peaks * sizeof(PeakInfo),
                     cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying peak_info to device");

    ret = cudaMemcpy(two_thetas_d, two_thetas, pat_len * sizeof(double),
                     cudaMemcpyHostToDevice);
    log_cuda_err(ret, "copying two_thetas to device");

    gpu_pat_info[i] = (CUDAPattern){
        .peaks = &peaks_d[n_peaks_running],
        .peak_info = &peak_info_d[n_peaks_running],
        .n_peaks = pat_info[i].n_peaks,
    };
    n_peaks_running += pat_info[i].n_peaks;
  }

  ret = cudaMemcpy(patterns_d, gpu_pat_info, n_patterns * sizeof(CUDAPattern),
                   cudaMemcpyHostToDevice);
  log_cuda_err(ret, "copying patterns to device");

  // int block_size = 0;
  // int min_grid_size = 0;
  // int array_count = n_patterns * pat_len;
  // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
  //                                    (void *)discretize_kernel, 0,
  //                                    array_count);
  // int grid_size = (array_count + block_size - 1) / block_size;

  // printf("launching as (%d, %d)\n", grid_size, block_size);
  // discretize_kernel<<<grid_size, block_size>>>(
  //     patterns_d, intensities_d, two_thetas_d, n_patterns, pat_len);

  discretize_kernel_single_pattern<<<n_patterns / 32, 32>>>(
      patterns_d, intensities_d, two_thetas_d, n_patterns, pat_len);
  ret = cudaPeekAtLastError();
  log_cuda_err(ret, "launching discretization kernel");
  cudaDeviceSynchronize();

  ret =
      cudaMemcpy(intensities, intensities_d,
                 n_patterns * pat_len * sizeof(double), cudaMemcpyDeviceToHost);
  log_cuda_err(ret, "copying intensities from device to host");

  free(gpu_pat_info);

  cudaFree(peaks_d);
  cudaFree(peak_info_d);
  cudaFree(patterns_d);
  cudaFree(two_thetas_d);
  cudaFree(intensities_d);
  return true;
}

} // extern "C"
