
#include "cuda_common.cu"
#include "stdio.h"

extern "C" {

struct DevInfo {
  char *device_name; // if device_name is null, init has failed
  size_t available_memory_bytes;
  size_t init_free_memory_bytes;
  int api_version;
  int runtime_version;
  int device_id;
};

bool dev_props(DevInfo *di) {
  int device = 0;
  cu_lerr(cudaGetDevice(&device), "getting cuda device");

  cudaDeviceProp prop;
  cu_lerr(cudaGetDeviceProperties(&prop, device), "getting cuda device properties");
  // this leaks a bit of memory, but I really don't care - it's not like we're calling this in a loop

  int rt_version;
  cu_lerr(cudaRuntimeGetVersion(&rt_version), "getting cuda runtime version");

  int api_version;
  cu_lerr(cudaDriverGetVersion(&api_version), "getting cuda api version");

  size_t dev_name_len_including_terminator = strlen(prop.name) + 1;
  char *device_name = (char *)malloc(dev_name_len_including_terminator);
  memcpy(device_name, prop.name, dev_name_len_including_terminator);
  size_t free_mem;
  size_t total_mem;

  cu_lerr(cudaMemGetInfo(&free_mem, &total_mem), "getting free memory");

  di->device_name = device_name;
  di->available_memory_bytes = prop.totalGlobalMem;
  di->api_version = api_version;
  di->runtime_version = rt_version;
  di->device_id = device;
  di->init_free_memory_bytes = free_mem;

  return true;
}

DevInfo init_get_dev_info(error_fn errfn, info_fn infofn, debug_fn debugfn) {
  errf = errfn;
  infof = infofn;
  debugf = debugfn;

  DevInfo dev = {0};
  if (!dev_props(&dev)) {
    dev.device_name = NULL;
  }

  return dev;
}
}
