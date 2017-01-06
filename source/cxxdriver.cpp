#include <cuda.h>
#include <nvrtc.h>

#include <helper_functions.h>
#include <drvapi_error_string.h>
#include <helper_cuda.h>

// using thrust::device_vector;

struct DeviceInfo {
  int id;
  int device;
  CUcontext context;
  static const int namelen = 100;
  char name[namelen];
};

CUresult cuMemAlloc_(unsigned long* dptr, size_t n) {
  return cuMemAlloc(reinterpret_cast<CUdeviceptr*>(dptr), n);
}

CUresult cuMemFree_(unsigned long dptr) {
  return cuMemFree(static_cast<CUdeviceptr>(dptr));
}

#define CHECK_CU_RESULT(r) { if (r != CUDA_SUCCESS) { return r; } }

CUresult cudaDeviceInit(int id) {
  DeviceInfo d = {id};
  int deviceCount = 0;
  CHECK_CU_RESULT(cuInit(0));
  CHECK_CU_RESULT(cuDeviceGetCount(&deviceCount));
  if (id > deviceCount-1) {
    return CUDA_ERROR_NO_DEVICE;
  }

  CHECK_CU_RESULT(cuDeviceGet(&d.device, d.id));
  CHECK_CU_RESULT(cuDeviceGetName(d.name, d.namelen, d.device));
  printf("> Using CUDA Device [%d]: %s\n", d.id, d.name);

  CHECK_CU_RESULT(cuCtxCreate(&d.context, 0, d.device));
  return CUDA_SUCCESS;
}

nvrtcResult nvrtcCreateProgram_(
  nvrtcProgram* prog, const char* code, const char* name,
  int numHeaders, char** headers, char** includeNames) {
  return nvrtcCreateProgram(prog, code, name, numHeaders, headers, includeNames);
}
const char* getErrorName(nvrtcResult result) {
  return nvrtcGetErrorString(result);
}


/*
const char* getErrorName(CUresult result) {
  const char* str;
  cuGetErrorString(result, &str);
  return str;
}
*/

/*
  switch (result)
  {
  case CUDA_SUCCESS:
    return "CUDA_SUCCESS";

  case CUDA_ERROR_INVALID_VALUE:
    return "CUDA_ERROR_INVALID_VALUE";

  case CUDA_ERROR_OUT_OF_MEMORY:
    return "CUDA_ERROR_OUT_OF_MEMORY";

  case CUDA_ERROR_NOT_INITIALIZED:
    return "CUDA_ERROR_NOT_INITIALIZED";

  case CUDA_ERROR_DEINITIALIZED:
    return "CUDA_ERROR_DEINITIALIZED";

  case CUDA_ERROR_PROFILER_DISABLED:
    return "CUDA_ERROR_PROFILER_DISABLED";

  case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
    return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

  case CUDA_ERROR_PROFILER_ALREADY_STARTED:
    return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

  case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
    return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

  case CUDA_ERROR_NO_DEVICE:
    return "CUDA_ERROR_NO_DEVICE";

  case CUDA_ERROR_INVALID_DEVICE:
    return "CUDA_ERROR_INVALID_DEVICE";

  case CUDA_ERROR_INVALID_IMAGE:
    return "CUDA_ERROR_INVALID_IMAGE";

  case CUDA_ERROR_INVALID_CONTEXT:
    return "CUDA_ERROR_INVALID_CONTEXT";

  case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
    return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

  case CUDA_ERROR_MAP_FAILED:
    return "CUDA_ERROR_MAP_FAILED";

  case CUDA_ERROR_UNMAP_FAILED:
    return "CUDA_ERROR_UNMAP_FAILED";

  case CUDA_ERROR_ARRAY_IS_MAPPED:
    return "CUDA_ERROR_ARRAY_IS_MAPPED";

  case CUDA_ERROR_ALREADY_MAPPED:
    return "CUDA_ERROR_ALREADY_MAPPED";

  case CUDA_ERROR_NO_BINARY_FOR_GPU:
    return "CUDA_ERROR_NO_BINARY_FOR_GPU";

  case CUDA_ERROR_ALREADY_ACQUIRED:
    return "CUDA_ERROR_ALREADY_ACQUIRED";

  case CUDA_ERROR_NOT_MAPPED:
    return "CUDA_ERROR_NOT_MAPPED";

  case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
    return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

  case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
    return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

  case CUDA_ERROR_ECC_UNCORRECTABLE:
    return "CUDA_ERROR_ECC_UNCORRECTABLE";

  case CUDA_ERROR_UNSUPPORTED_LIMIT:
    return "CUDA_ERROR_UNSUPPORTED_LIMIT";

  case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
    return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

  case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
    return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";

  case CUDA_ERROR_INVALID_PTX:
    return "CUDA_ERROR_INVALID_PTX";

  case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
    return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";

  case CUDA_ERROR_INVALID_SOURCE:
    return "CUDA_ERROR_INVALID_SOURCE";

  case CUDA_ERROR_FILE_NOT_FOUND:
    return "CUDA_ERROR_FILE_NOT_FOUND";

  case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
    return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

  case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
    return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

  case CUDA_ERROR_OPERATING_SYSTEM:
    return "CUDA_ERROR_OPERATING_SYSTEM";

  case CUDA_ERROR_INVALID_HANDLE:
    return "CUDA_ERROR_INVALID_HANDLE";

  case CUDA_ERROR_NOT_FOUND:
    return "CUDA_ERROR_NOT_FOUND";

  case CUDA_ERROR_NOT_READY:
    return "CUDA_ERROR_NOT_READY";

  case CUDA_ERROR_ILLEGAL_ADDRESS:
    return "CUDA_ERROR_ILLEGAL_ADDRESS";

  case CUDA_ERROR_LAUNCH_FAILED:
    return "CUDA_ERROR_LAUNCH_FAILED";

  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

  case CUDA_ERROR_LAUNCH_TIMEOUT:
    return "CUDA_ERROR_LAUNCH_TIMEOUT";

  case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
    return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

  case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
    return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

  case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
    return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

  case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
    return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

  case CUDA_ERROR_CONTEXT_IS_DESTROYED:
    return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

  case CUDA_ERROR_ASSERT:
    return "CUDA_ERROR_ASSERT";

  case CUDA_ERROR_TOO_MANY_PEERS:
    return "CUDA_ERROR_TOO_MANY_PEERS";

  case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
    return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

  case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
    return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

  case CUDA_ERROR_HARDWARE_STACK_ERROR:
    return "CUDA_ERROR_HARDWARE_STACK_ERROR";

  case CUDA_ERROR_ILLEGAL_INSTRUCTION:
    return "CUDA_ERROR_ILLEGAL_INSTRUCTION";

  case CUDA_ERROR_MISALIGNED_ADDRESS:
    return "CUDA_ERROR_MISALIGNED_ADDRESS";

  case CUDA_ERROR_INVALID_ADDRESS_SPACE:
    return "CUDA_ERROR_INVALID_ADDRESS_SPACE";

  case CUDA_ERROR_INVALID_PC:
    return "CUDA_ERROR_INVALID_PC";

  case CUDA_ERROR_NOT_PERMITTED:
    return "CUDA_ERROR_NOT_PERMITTED";

  case CUDA_ERROR_NOT_SUPPORTED:
    return "CUDA_ERROR_NOT_SUPPORTED";

  case CUDA_ERROR_UNKNOWN:
    return "CUDA_ERROR_UNKNOWN";
  }

  return "<unknown>";
}
*/

