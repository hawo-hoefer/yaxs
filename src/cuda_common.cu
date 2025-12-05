#ifndef CUDA_COMMON
#define CUDA_COMMON

typedef void (*error_fn)(const char *, int, const char *, int, const char *);
typedef void (*info_fn)(const char *);
typedef void (*debug_fn)(const char *);

error_fn errf;
info_fn infof;
debug_fn debugf;

char tmp_str_buf[512] = {0};

#define TAU 3.1415926535897932384626433832795028841972f * 2.0f
#define PI 3.1415926535897932384626433832795028841972f
#define cu_lerr(ret, msg)                                                                                              \
  if (ret != cudaSuccess) {                                                                                            \
    errf(__FILE__, __LINE__, (msg), (int)(ret), cudaGetErrorString((ret)));                                            \
    fflush(stderr);                                                                                                    \
    return false;                                                                                                      \
  }
#endif // CUDA_COMMON
