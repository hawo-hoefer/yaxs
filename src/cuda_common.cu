#ifndef CUDA_COMMON
#define CUDA_COMMON

typedef void (*error_fn)(const char *, int, const char *, int, const char *);
typedef void (*info_fn)(const char *);
typedef void (*debug_fn)(const char *);

error_fn errf;
info_fn infof;
debug_fn debugf;

char tmp_str_buf[1024] = {0};

#define TAU 3.1415926535897932384626433832795028841972f * 2.0f
#define PI 3.1415926535897932384626433832795028841972f
#define cu_lerr(ret, msg)                                                                                              \
  if (ret != cudaSuccess) {                                                                                            \
    errf(__FILE__, __LINE__, (msg), (int)(ret), cudaGetErrorString((ret)));                                            \
    fflush(stderr);                                                                                                    \
    return false;                                                                                                      \
  }

#define DEG2RAD(x) ((x) / 180.0 * PI);

#define launch_kernel_sensibly_no_shmem(kernel, description, array_count, ...)                                         \
  do {                                                                                                                 \
    debugf("Determining " description " launch parameters");                                                           \
    int block_size = 0;                                                                                                \
    int min_grid_size = 0;                                                                                             \
    cu_lerr(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)(kernel), 0, (array_count)),       \
            "Determining " description " launch parameters");                                                          \
    int grid_size = ((array_count) + block_size - 1) / block_size;                                                     \
    debugf("launching kernel " description);                                                                           \
                                                                                                                       \
    (kernel)<<<grid_size, block_size>>>(__VA_ARGS__);                                                                  \
    cu_lerr(cudaDeviceSynchronize(), "synchronizing device after " description);                                       \
  } while (0)

#endif // CUDA_COMMON
