import std.conv;
import std.string;

import driver;
import typechecker;


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

unittest {
  import std.exception;
  assertThrown(new Array!float(3, -1));
}


class Array(T) {
  alias Element = T;
  alias Storage = T*;

  int dev;
  size_t rawLength;
  size_t length;
  CUdeviceptr ptr;
  T[] cpu_storage;
  this(size_t n, int dev = 0) {
    dev = dev;
    length = n;
    rawLength = T.sizeof * n;
    check(cudaDeviceInit_(dev));
    check(cuMemAlloc_(&ptr, rawLength));
  }
  this(in T[] src, int dev = 0) {
    this(src.length);
    to_gpu(src);
  }
  ~this() {
    check(cuMemFree_(&ptr));
  }
  void to_gpu(in T[] src) {
    cuMemcpyHtoD_(ptr, to!(const(void*))(src.ptr), rawLength);
  }
  T[] to_cpu() {
    cpu_storage.length = rawLength / T.sizeof;
    // check(cudaDeviceInit_(dev));
    check(cuMemcpyDtoH_(to!(void*)(cpu_storage.ptr), ptr, rawLength));
    return cpu_storage;
  }
}

unittest {
  float[] h = [1,2,3];
  auto d = new Array!float(h);
  assert(h == d.to_cpu());
  static assert(is(typeof(d).Storage == float*));
  static assert(is(typeof(d).Element == float));
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
    source = qualifier ~ returnType ~ nameStr ~
      "(" ~ args ~ ")" ~
      "{" ~ bodyStr ~ "}";
  }
}

unittest {
  auto saxpy = Code(
    "saxpy", "float *A, float *B, float *C, int numElements", `
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      if (i < numElements) C[i] = A[i] + B[i];
    `);
  assert(saxpy.source ==
   `extern "C" __global__ void saxpy(float *A, float *B, float *C, int numElements){
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      if (i < numElements) C[i] = A[i] + B[i];
    }`);
}

void* vptr(F)(ref F f) {
  static if (is(typeof(f.ptr))) {
    return to!(void*)(&f.ptr);
  } else {
    return to!(void*)(&f);
  }
}

class Kernel {
  /*
    FIXME: CUresult.CUDA_ERROR_INVALID_HANDLE
    - init device in this or opCall?
    - multi device support

    FIXME: support a setting of <<<threads, blocks, shared-memory, stream>>>
  */
  CUfunction func; // FIXME make func const
  const Code code;

  void* vfunc() {
    return to!(void*)(&func);
  }

  this(Code c) {
    code = c;
    check(compile_(vfunc, code.name.toStringz, code.source.toStringz));
  }

  void opCall() {}

  void opCall(Ts...)(Ts targs) {
    // TODO: type check between targs and code.args (runtime ver.)
    void[] vargs;
    foreach (i, t; targs) {
      vargs ~= [vptr(targs[i])];
    }
    uint[] grids = [256, 1, 1];
    uint bx = to!uint((grids[0] + targs[0].length - 1) / grids[0]);
    uint[] blocks = [bx, 1, 1];
    check(launch_(vptr(func), vargs.ptr, grids.ptr, blocks.ptr));
  }
}


class TypedKernel(Code c) {
  /*
    FIXME: CUresult.CUDA_ERROR_INVALID_HANDLE
    - init device in this or opCall?
    - multi device support

    FIXME: support a setting of <<<threads, blocks, shared-memory, stream>>>
  */
  CUfunction func; // FIXME make func const
  immutable Code code = c;
  immutable string kargs = c.args;

  void* vfunc() {
    return to!(void*)(&func);
  }

  this() {
    check(compile_(vfunc, code.name.toStringz, code.source.toStringz));
  }

  void opCall() {}

  void opCall(Ts...)(Ts targs) {
    // TODO: convert targs[i] to targs[i].ptr if it has .ptr method
    staticAssert!(AssignableArgTypes, kargs)(targs);
    void[] vargs;
    foreach (i, t; targs) {
      vargs ~= [vptr(targs[i])];
    }
    uint[] grids = [256, 1, 1];
    uint bx = to!uint((grids[0] + targs[0].length - 1) / grids[0]);
    uint[] blocks = [bx, 1, 1];
    check(launch_(vptr(func), vargs.ptr, grids.ptr, blocks.ptr));
  }
}

unittest {
  import std.stdio;
  import std.random;
  import std.range;


  auto empty = new Kernel
    (Code("empty", "", "int i = blockDim.x * blockIdx.x + threadIdx.x;"));
  empty();

  int n = 10;
  auto gen = () => new Array!float(generate!(() => uniform(-1f, 1f)).take(n).array());
  auto a = gen();
  auto b = gen();
  auto c = new Array!float(n);
  auto saxpy = new Kernel(
    Code(
      "saxpy", q{float *A, float *B, float *C, int numElements},
      q{
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < numElements) C[i] = A[i] + B[i];
      })
    );
  saxpy(a, b, c, n);
  foreach (ai, bi, ci; zip(a.to_cpu(), b.to_cpu(), c.to_cpu())) {
    assert(ai + bi == ci);
  }

  enum code = Code(
      "saxpy", q{float *A, float *B, float *C, int numElements},
      q{
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < numElements) C[i] = A[i] + B[i];
      });
  auto tsaxpy = new TypedKernel!(code)();
  tsaxpy(a, b, c, n);
  foreach (ai, bi, ci; zip(a.to_cpu(), b.to_cpu(), c.to_cpu())) {
    assert(ai + bi == ci);
  }
}