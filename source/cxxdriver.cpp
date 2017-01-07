/*
  NOTE

  ## about names

  use "_" suffix to every public identifiers to distinguish C++ from D.

  ## about exceptions

  Exceptions are prohibited, because we can't communicate C++ & D by exceptions currently

  ## about resource handling

  do not operate resource handling methods (eg. RAII) in C++ but we do them in D.

*/

#include <string>
#include <memory>
#include <type_traits>

#include <cuda.h>
#include <nvrtc.h>

#include <helper_functions.h>
#include <drvapi_error_string.h>
#include <helper_cuda.h>


#define CHECK_RESULT(r) { if (r) { return r; } }


template<typename Tarray>
std::unique_ptr<Tarray> make_unique(std::size_t n) {
  using T = typename std::remove_extent<Tarray>::type;
  return std::unique_ptr<Tarray>(new T[n]);
}

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

using DCUdeviceptr = unsigned long; /// FIXME

CUresult cuMemAlloc_(DCUdeviceptr* dptr, size_t n) {
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
  CHECK_RESULT(cuInit(0));
  CHECK_RESULT(cuDeviceGetCount(&deviceCount));
  if (id > deviceCount-1) {
    return CUDA_ERROR_NO_DEVICE;
  }

  CHECK_RESULT(cuDeviceGet(&d.device, d.id));
  CHECK_RESULT(cuDeviceGetName(d.name, d.namelen, d.device));
  printf("> Using CUDA Device [%d]: %s\n", d.id, d.name);

  CHECK_RESULT(cuCtxCreate(&d.context, 0, d.device));
  return CUDA_SUCCESS;
}


#include <cuda_runtime.h>
#include <nvrtc_helper.h>


CUfunction compile_(const char* funcname, const char* code) {
  // compile
  std::string filename(funcname);
  filename += ".cu";
  nvrtcProgram prog;
  NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&prog, code, filename.c_str(), 0, NULL, NULL));
  NVRTC_SAFE_CALL("nvrtcCompileProgram", nvrtcCompileProgram(prog, 0, NULL));

  // dump log
  size_t logSize;
  NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(prog, &logSize));
  auto log = make_unique<char[]>(sizeof(char) * logSize + 1);
  NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log.get()));
  log[logSize] = '\x0';

  // fetch PTX
  size_t ptxSize;
  NVRTC_SAFE_CALL("nvrtcGetPTXSize", nvrtcGetPTXSize(prog, &ptxSize));
  auto ptx = make_unique<char[]>(sizeof(char) * ptxSize + 1);
  NVRTC_SAFE_CALL("nvrtcGetPTX", nvrtcGetPTX(prog, ptx.get()));
  NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&prog));

  CUmodule module = loadPTX(ptx.get(), 0, NULL); /// ???
  CUfunction kernel_addr;
  checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, funcname));
  return kernel_addr;
}


void call_(CUfunction kernel_addr, DCUdeviceptr* d_A, DCUdeviceptr* d_B, DCUdeviceptr* d_C, int numElements) {
  int threadsPerBlock = 256;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  dim3 cudaBlockSize(threadsPerBlock,1,1);
  dim3 cudaGridSize(blocksPerGrid, 1, 1);

  void *arr[] = { (void *)d_A, (void *)d_B, (void *)d_C, (void *)&numElements };
  checkCudaErrors(cuLaunchKernel(kernel_addr,
                                 cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
                                 cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, /* block dim */
                                 0,0, /* shared mem, stream */
                                 &arr[0], /* arguments */
                                 0));
  checkCudaErrors(cuCtxSynchronize());
}

#undef CHECK_RESULT
