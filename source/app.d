import std.stdio;
import std.process;

import nv = nvrtc;
import cudriver;

string nvidia_smi() {
  auto p = execute(["nvidia-smi"]);
  if (p.status != 0) throw new Exception("Failed to execute nvidia-smi");
  return p.output;
}

void main() {
  int n = 100;
  auto a = new nv.Array!float(n, 0);
  nvidia_smi().writeln;
  auto b = new nv.Array!float(n, 0);
  auto c = new nv.Array!float(n, 0);
  auto c1 = new nv.Array!float(n, 1);
  nvidia_smi().writeln;

  nv.Kernel kernel = nv.rtc(`
extern "C" __global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] + B[i];
}`);

  int[] threads = [256, 1, 1];
  int[] blocks = [(n + threads[0] - 1) / threads[0], 1, 1];
  nv.call(kernel, threads, blocks, a, b, c, n);
}
