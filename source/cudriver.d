extern (C++):

import core.stdc.config;


alias CUdeviceptr = c_ulong;

CUresult cuMemAlloc_(CUdeviceptr* dptr, size_t bytesize);

CUresult cuMemFree_(CUdeviceptr dptr);

CUresult cudaDeviceInit_(int dev);


enum CUresult {
  CUDA_SUCCESS                              = 0,
  CUDA_ERROR_INVALID_VALUE                  = 1,
  CUDA_ERROR_OUT_OF_MEMORY                  = 2,
  CUDA_ERROR_NOT_INITIALIZED                = 3,
  CUDA_ERROR_DEINITIALIZED                  = 4,
  CUDA_ERROR_PROFILER_DISABLED              = 5,
  CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,
  CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7,
  CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,
  CUDA_ERROR_NO_DEVICE                      = 100,
  CUDA_ERROR_INVALID_DEVICE                 = 101,
  CUDA_ERROR_INVALID_IMAGE                  = 200,
  CUDA_ERROR_INVALID_CONTEXT                = 201,
  CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,
  CUDA_ERROR_MAP_FAILED                     = 205,
  CUDA_ERROR_UNMAP_FAILED                   = 206,
  CUDA_ERROR_ARRAY_IS_MAPPED                = 207,
  CUDA_ERROR_ALREADY_MAPPED                 = 208,
  CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,
  CUDA_ERROR_ALREADY_ACQUIRED               = 210,
  CUDA_ERROR_NOT_MAPPED                     = 211,
  CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,
  CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,
  CUDA_ERROR_ECC_UNCORRECTABLE              = 214,
  CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,
  CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,
  CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217,
  CUDA_ERROR_INVALID_PTX                    = 218,
  CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = 219,
  CUDA_ERROR_NVLINK_UNCORRECTABLE           = 220,
  CUDA_ERROR_INVALID_SOURCE                 = 300,
  CUDA_ERROR_FILE_NOT_FOUND                 = 301,
  CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
  CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,
  CUDA_ERROR_OPERATING_SYSTEM               = 304,
  CUDA_ERROR_INVALID_HANDLE                 = 400,
  CUDA_ERROR_NOT_FOUND                      = 500,
  CUDA_ERROR_NOT_READY                      = 600,
  CUDA_ERROR_ILLEGAL_ADDRESS                = 700,
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,
  CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,
  CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,
  CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,
  CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705,
  CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,
  CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,
  CUDA_ERROR_ASSERT                         = 710,
  CUDA_ERROR_TOO_MANY_PEERS                 = 711,
  CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
  CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,
  CUDA_ERROR_HARDWARE_STACK_ERROR           = 714,
  CUDA_ERROR_ILLEGAL_INSTRUCTION            = 715,
  CUDA_ERROR_MISALIGNED_ADDRESS             = 716,
  CUDA_ERROR_INVALID_ADDRESS_SPACE          = 717,
  CUDA_ERROR_INVALID_PC                     = 718,
  CUDA_ERROR_LAUNCH_FAILED                  = 719,
  CUDA_ERROR_NOT_PERMITTED                  = 800,
  CUDA_ERROR_NOT_SUPPORTED                  = 801,
  CUDA_ERROR_UNKNOWN                        = 999
}


enum nvrtcResult {
  NVRTC_SUCCESS = 0,
  NVRTC_ERROR_OUT_OF_MEMORY = 1,
  NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVRTC_ERROR_INVALID_INPUT = 3,
  NVRTC_ERROR_INVALID_PROGRAM = 4,
  NVRTC_ERROR_INVALID_OPTION = 5,
  NVRTC_ERROR_COMPILATION = 6,
  NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  NVRTC_ERROR_INTERNAL_ERROR = 11
}


struct CUfunction_st;
alias CUfunction = CUfunction_st*;

nvrtcResult compile_(void* kernel_addr, const(char*) funcname, const(char*) code);

CUresult call_(void* kernel_addr, CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int numElements);

CUresult call_(void* kernel_addr);

CUresult cuMemcpyDtoH_(void* dstHost, CUdeviceptr srcDevice, size_t byteCount);

CUresult cuMemcpyHtoD_(CUdeviceptr dstDevice, const(void*) srcHost, size_t byteCount);
