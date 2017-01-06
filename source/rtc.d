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
    check(cudaDeviceInit_(dev));
    check(cuMemAlloc_(&ptr, size));
  }
  ~this() {
    check(cuMemFree_(ptr));
  }
}


class Kernel {
  static immutable funcHead = `extern "C" __global__ void `;
  string name;
  CUfunction func;
  this(string funcName, string funcBody) {
    name = name;
    auto code = funcHead ~ funcName ~ funcBody;
    func = compile_(funcName.toStringz, code.toStringz);
  }

  void call(A)(A a, A b, A c, int n) {
    call_(func, a.ptr, b.ptr, c.ptr, n);
  }
}
