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

struct Code {
  /*
    FIXME: template support
    - see: /usr/local/cuda/samples/0_Simple/simpleTemplates_nvrtc
   */
  immutable qualifier = `extern "C" __global__ `;
  immutable returnType = "void ";
  const string name;
  const string args;
  const string source;

  this(in string nameStr, in string argumentsStr, in string bodyStr) {
    name = nameStr;
    args = argumentsStr;
    source = qualifier ~ returnType ~ nameStr ~ args ~ bodyStr;
  }
}

unittest {
  auto saxpy = Code(
    "saxpy", "(float *A, float *B, float *C, int numElements)", ` {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      if (i < numElements) C[i] = A[i] + B[i];
    }`);

  assert(saxpy.source ==
   `extern "C" __global__ void saxpy(float *A, float *B, float *C, int numElements) {
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      if (i < numElements) C[i] = A[i] + B[i];
    }`);
}

class Kernel {
  /*
    FIXME: CUresult.CUDA_ERROR_INVALID_HANDLE
    - init device in this or opCall?
    - multi device support

    FIXME: support a setting of <<<threads, blocks, shared-memory, stream>>>
  */
  CUfunction func; // FIXME make func const

  void* vptr() {
    return to!(void*)(&func);
  }

  this(Code code) {
    check(compile_(vptr(), code.name.toStringz, code.source.toStringz));
  }

  void opCall()() {
    check(call_(vptr()));
  }

  void opCall(A)(A a, A b, A c, int n) {
    check(call_(vptr(), a.ptr, b.ptr, c.ptr, n));
  }
}

unittest {
  import std.random;
  import std.range;

  auto empty = new Kernel(Code(
    "empty", "()",
    "{int i = blockDim.x * blockIdx.x + threadIdx.x;}"));
  empty();

  int n = 10;
  auto gen = () => new Array!float(generate!(() => uniform(-1.0f, 1.0f)).take(n).array());
  auto a = gen();
  auto b = gen();
  auto c = gen();
  auto saxpy = new Kernel(Code(
    "saxpy", "(float *A, float *B, float *C, int numElements)", `{
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      if (i < numElements) C[i] = A[i] + B[i];
    }`));

  saxpy(a, b, c, n);
  foreach (ai, bi, ci; zip(a.to_cpu(), b.to_cpu(), c.to_cpu())) {
    assert(ai + bi == ci);
  }
}
