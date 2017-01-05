import std.stdio;
import std.conv;
import std.format;
import cudriver;

class CudaError : Exception {
  const CUresult result;

  this(CUresult r, string file = __FILE__, size_t line = __LINE__) {
    result = r;
    auto msg = cudaGetErrorEnum(r);
    msg ~= ">> raised from %s(L%d)".format(file, line);
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
