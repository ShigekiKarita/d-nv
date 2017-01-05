import std.stdio;
import std.conv;
import std.format;
import std.string;
import cudriver;

class CudaError : Exception {
  const CUresult result;

  this(CUresult r, string file = __FILE__, size_t line = __LINE__) {
    result = r;
    auto err = cudaGetErrorEnum(r);
    auto msg = "%s>> raised from %s(L%d)".format(err, file, line);
    super(msg, file, line);
  }
}

void check(CUresult result) {
  if (result != CUresult.CUDA_SUCCESS) {
    throw new CudaError(result);
  }
}

/*
struct Program {
  string log;
  string ptx;
  this(string name, string code) {
    // TODO: check nvrtcResult

    // TODO: support headers (numHeaders, headers, includeNames)
    nvrtcProgram data;
    nvrtcCreateProgram(&data, code, name, 0, null, null);

    // TODO: support opts (numOpts, opts)
    nvrtcCompileProgram(&data, 0, null);
    nvrtcGetProgramLog(prog, log);
    nvrtcGetPTX(prog, ptx);
  }

  void call(Ts...)(Ts args) {
    CUfunction kernel;
    cuGetContext
    cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, null, args, 0)
  }
}
*/

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
    nvrtcResult res = nvrtcCreateProgram_(&p, code.toStringz, name.toStringz);
    nvrtcGetErrorString_(res).fromStringz.writeln;
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
