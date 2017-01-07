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
  int dev;
  size_t size;
  CUdeviceptr ptr;
  T[] cpu_storage;
  this(size_t n, int dev = 0) {
    dev = dev;
    size = T.sizeof * n;
    check(cudaDeviceInit_(dev));
    check(cuMemAlloc_(&ptr, size));
  }
  this(in T[] src, int dev = 0) {
    this(src.length);
    to_gpu(src);
  }
  ~this() {
    check(cuMemFree_(ptr));
  }
  void to_gpu(in T[] src) {
    cuMemcpyHtoD_(ptr, to!(const(void*))(src.ptr), size);
  }
  T[] to_cpu() {
    cpu_storage.length = size / T.sizeof;
    // check(cudaDeviceInit_(dev));
    check(cuMemcpyDtoH_(to!(void*)(cpu_storage.ptr), ptr, size));
    return cpu_storage;
  }
}

unittest {
  float[] h = [1,2,3];
  auto d = new Array!float(h);
  assert(h == d.to_cpu());
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
