module dnv.driver;
import dnv.error;

import core.stdc.config;


extern (C++):

alias CUdeviceptr = c_ulong;

CUresult cuMemAlloc_(CUdeviceptr* dptr, size_t bytesize);

CUresult cuMemFree_(CUdeviceptr* dptr);

CUresult cudaDeviceInit_(int dev);

struct CUfunction_st;
alias CUfunction = CUfunction_st*;

nvrtcResult compile_(void* kernel_addr, const(char*) funcname, const(char*) code);


struct CUstream_st;
alias CUstream = CUstream_st*;

CUresult launch_(void* kernel_addr, void* kernel_args,
                 uint* grids, uint* blocks);
                 // size_t shared_memory=0, CUstream stream=null);

CUresult cuMemcpyDtoH_(void* dstHost, CUdeviceptr srcDevice, size_t byteCount);

CUresult cuMemcpyHtoD_(CUdeviceptr dstDevice, const(void*) srcHost, size_t byteCount);
