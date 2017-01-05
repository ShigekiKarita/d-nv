import std.stdio;
import std.conv;
import cudriver;

void check(CUresult result) {
  if (result != CUresult.CUDA_SUCCESS) {
    auto info = cudaGetErrorEnum(result);
    throw new Exception(info);
  }
}

class Array(T) {
  size_t size;
  CUdeviceptr ptr;
  this(size_t n, int dev = 1) {
    size = T.sizeof * n;
    check(cudaDeviceInit(dev));
    check(cuMemAlloc_(&ptr, size));
  }
}

class Kernel {
  string code;
  this(string s) {
    code = s;
  }
}

auto rtc(string code) {
  return new Kernel(code);
}

void call(T...)(T ts) {
}
