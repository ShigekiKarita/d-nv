module dnv.kernel;

import std.conv : to;

import dnv.storage;
import dnv.compiler;
import dnv.driver;
import dnv.error;

import derelict.cuda.driverapi : CUfunction;


void* vptr(F)(ref F f) {
  static if (is(typeof(f.ptr))) {
    return to!(void*)(&f.ptr);
  } else {
    return to!(void*)(&f);
  }
}


class KernelBase(Compiler, Launcher) {
  /*
    FIXME: CUresult.CUDA_ERROR_INVALID_HANDLE
    - init device in this or opCall?
    - multi device support

    NOTE: Kernel can be factored into Kernel!(Compiler, Launcher)
    - Compiler: Static, Dynamic, Unsafe ... and user-defined
    - Launcher: Simple, Heavy, Shared, Async ... and user-defined
  */
  CUfunction func;
  Launcher launch;
  Compiler compiler;

  // FIXME: need better prediction
  static if (is(typeof(compiler.cargs))) {
    this() {
      compiler.build(vfunc, compiler.code);
    }
  } else {
    this(Code code) {
      compiler.build(vfunc, code);
    }
  }

  void* vfunc() {
    return to!(void*)(&func);
  }

  void opCall() {}

  void opCall(Ts...)(Ts targs) {
    compiler.assertArgs(targs);

    void[] vargs;
    foreach (i, t; targs) {
      vargs ~= [vptr(targs[i])];
    }

    launch.setup(targs);
    check(dnv.driver.launch(vptr(func), vargs.ptr, launch.grids.ptr, launch.blocks.ptr));
  }
}

alias RuntimeKernel(L = SimpleLauncher) = KernelBase!(UnsafeCompiler, L);

alias TypedKernel(Code code, L = SimpleLauncher) = KernelBase!(StaticCompiler!code, L);


unittest {
  import std.stdio;
  import std.random;
  import std.range;


  auto empty = new RuntimeKernel!()
    (Code("empty", "", "int i = blockDim.x * blockIdx.x + threadIdx.x;"));
  empty();

  int n = 10;
  auto gen = () => new Array!float(generate!(() => uniform(-1f, 1f)).take(n).array());
  auto a = gen();
  auto b = gen();
  auto c = new Array!float(n);
  auto saxpy = new RuntimeKernel!()(
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
  auto tsaxpy = new TypedKernel!(code);
  tsaxpy(a, b, c, n);
  foreach (ai, bi, ci; zip(a.to_cpu(), b.to_cpu(), c.to_cpu())) {
    assert(ai + bi == ci);
  }
}
