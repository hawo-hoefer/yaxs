#ifndef CUDA_COMMON
#define CUDA_COMMON

thread_local size_t n_chunks = 0;
thread_local size_t chunk_id = 0;
thread_local int device_id = -1;

typedef void (*error_fn)(int, size_t, size_t, const char *, int, const char *, int, const char *);
typedef void (*info_fn)(int, size_t, size_t, const char *);
typedef void (*debug_fn)(int, size_t, size_t, const char *);

error_fn _errf;
info_fn _infof;
debug_fn _debugf;

char tmp_str_buf[1024] = {0};

#define infof(msg) _infof(device_id, chunk_id, n_chunks, (msg))
#define debugf(msg) _debugf(device_id, chunk_id, n_chunks, (msg))
#define errf(ret, msg, err_string) _errf(device_id, chunk_id, n_chunks, __FILE__, __LINE__, (ret), (msg), (err_string))

#define TAU 3.1415926535897932384626433832795028841972f * 2.0f
#define PI 3.1415926535897932384626433832795028841972f
#define cu_lerr(ret, msg)                                                                                              \
  if (ret != cudaSuccess) {                                                                                            \
    errf((msg), (int)(ret), cudaGetErrorString((ret)));                                                                \
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
