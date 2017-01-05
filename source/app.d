import std.stdio;
import std.process;
import std.string;

import nv = rtc;
import cudriver;

string nvidia_smi() {
  auto p = executeShell("nvidia-smi");
  if (p.status != 0) throw new Exception("Failed to execute nvidia-smi");
  return p.output;
}

void main() {
  int n = 1000;
  auto a = new nv.Array!float(n, 0);
  auto b = new nv.Array!float(n, 0);
  auto c = new nv.Array!float(n, 0);
  nvidia_smi().writeln;
  try {
    auto c1 = new nv.Array!float(n, 1);
    nvidia_smi().writeln;
  } catch (nv.CudaError e) {
    writeln(e.msg ~ "\n there seems to be no device1");
  }

  auto code = `
extern "C" __global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] + B[i];
}`;

  nvrtcProgram p;
  nvrtcResult res = nvrtcCreateProgram_(&p, code.toStringz, "add.cu");
  if (res != nvrtcResult.NVRTC_SUCCESS) {
    throw new Exception("NVRTC error");
  }

  /*
  int[] threads = [256, 1, 1];
  int[] blocks = [(n + threads[0] - 1) / threads[0], 1, 1];
  nv.call(kernel, threads, blocks, a, b, c, n);
  */

  writeln(">>> finished <<<");
}
