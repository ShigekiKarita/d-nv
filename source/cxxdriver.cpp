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

CUresult cuMemFree_(DCUdeviceptr* dptr) {
  return cuMemFree(*reinterpret_cast<CUdeviceptr*>(dptr));
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


// FIXME: use CHECK_RESULT and return nvrtcResult instead of CUfunction
nvrtcResult compile_(void* kernel_addr, const char* funcname, const char* code) {
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
  checkCudaErrors(cuModuleGetFunction((CUfunction*) kernel_addr, module, funcname));
  return NVRTC_SUCCESS;
}


// FIXME: use CHECK_RESULT instead of CUfunction and return CUresult
CUresult call_(void* kernel_addr, DCUdeviceptr d_A, DCUdeviceptr d_B, DCUdeviceptr d_C, int numElements) {
  int threadsPerBlock = 256;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  dim3 cudaBlockSize(threadsPerBlock,1,1);
  dim3 cudaGridSize(blocksPerGrid, 1, 1);

  void *arr[] = { (void *)&d_A, (void *)&d_B, (void *)&d_C, (void *)&numElements };
  void ** vargs = arr;
  auto func = (CUfunction*) kernel_addr;
  CHECK_RESULT(cuLaunchKernel(*func,
                              cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
                              cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, /* block dim */
                              0,0, /* shared mem, stream */
                              &vargs[0], /* arguments */
                              0));
  CHECK_RESULT(cuCtxSynchronize());
  return CUDA_SUCCESS;
}

CUresult call_(void* kernel_addr) {
  int threadsPerBlock = 256;
  int blocksPerGrid =(100 + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  dim3 cudaBlockSize(threadsPerBlock,1,1);
  dim3 cudaGridSize(blocksPerGrid, 1, 1);

  void *arr[] = {};
  auto func = (CUfunction*) kernel_addr;
  CHECK_RESULT(cuLaunchKernel(*func,
                              cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
                              cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, /* block dim */
                              0,0, /* shared mem, stream */
                              &arr[0], /* arguments */
                              0));
  CHECK_RESULT(cuCtxSynchronize());
  return CUDA_SUCCESS;
}

CUresult launch_(void* kernel_addr, void* kernel_args) {
                 // dim3* grids, dim3* blocks,
                 // size_t shared = 0, CUstream stream = NULL) {
  CUfunction* func = (CUfunction*) kernel_addr;
  void** args = (void**) kernel_args;
  CHECK_RESULT(cuLaunchKernel(*func,
                              // grids.x, grids.y, grids.z,
                              // blocks.x, blocks.y, blocks.z,
                              256, 1, 1,
                              (10 + 256 - 1) / 256, 1, 1,
                              0, 0, // shared, stream, // FIXME
                              &args[0],
                              0)); // FIXME: what is this arg?
  CHECK_RESULT(cuCtxSynchronize());
  return CUDA_SUCCESS;
}


CUresult cuMemcpyDtoH_(void* dstHost, DCUdeviceptr srcDevice, size_t byteCount) {
  return cuMemcpyDtoH(dstHost, static_cast<CUdeviceptr>(srcDevice), byteCount);
}
CUresult cuMemcpyHtoD_(DCUdeviceptr dstDevice, const void* srcHost, size_t byteCount) {
  return cuMemcpyHtoD(static_cast<CUdeviceptr>(dstDevice), const_cast<void*>(srcHost), byteCount);
}


#undef CHECK_RESULT
