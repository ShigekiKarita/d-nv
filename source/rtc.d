import std.stdio;
import std.conv;
import std.format;
import std.string;
import std.traits;

import cudriver;


class CudaError(Result) : Exception {
  const Result result;

  this(Result r, string file = __FILE__, int line = __LINE__) {
    result = r;
    super(toString, file, line);
  }

  override string toString() {
    return Result.stringof ~ "." ~ result.to!string;
  }
}

void check(Result, string file = __FILE__, int line = __LINE__)(Result result) {
  if (result) {
    throw new CudaError!Result(result, file, line);
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
  ~this() {
    // TODO: should I ensure nothrow?
    check(cuMemFree_(ptr));
  }
}


class Kernel {
  string name;
  string ptx;
  string log;
  this(string name, string code) {
    name = name;
    nvrtcProgram p;
    check(nvrtcCreateProgram_(&p, code.toStringz, name.toStringz));
    // TODO: compile to ptx & get compiler's log
  }

  void call(T...)(int[3] threads, int[3] blocks, T ts) {
    /*
    CUfunction kernel;
    cuGetContext
    cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, null, args, 0)
    */
  }
}
