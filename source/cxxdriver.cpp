/*
NOTE

## about names

use "_" suffix to every public identifiers to distinguish C++ from D.

## about exceptions

Exceptions are prohibited, because we can't communicate C++ & D by exceptions currently

## about resource handling

do not operate resource handling methods (eg. RAII) in C++ but we do them in D.

 */

#include <cuda.h>
#include <nvrtc.h>

#include <helper_functions.h>
#include <drvapi_error_string.h>
#include <helper_cuda.h>


#define CHECK_CU_RESULT(r) { if (r != CUDA_SUCCESS) { return r; } }


CUresult cuMemAlloc_(unsigned long* dptr, size_t n) {
  return cuMemAlloc(reinterpret_cast<CUdeviceptr*>(dptr), n);
}

CUresult cuMemFree_(unsigned long dptr) {
  return cuMemFree(static_cast<CUdeviceptr>(dptr));
}


struct DeviceInfo_ {
  int id;
  int device;
  CUcontext context;
  static const int namelen = 100;
  char name[namelen];
};

CUresult cudaDeviceInit_(int id) {
  DeviceInfo_ d = {id};
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
