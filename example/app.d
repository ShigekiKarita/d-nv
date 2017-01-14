import std.stdio;
import std.process;
import std.stdio;
import std.random;
import std.range;

import dnv.kernel;
import dnv.storage;
import dnv.compiler;


string nvidia_smi() {
  auto p = executeShell("nvidia-smi");
  if (p.status != 0) throw new Exception("Failed to execute nvidia-smi");
  return p.output;
}

void main() {
  int n = 10;
  auto gen = () => new Array!float(generate!(() => uniform(-1f, 1f)).take(n).array());
  auto a = gen();
  auto b = gen();
  auto c = new Array!float(n);
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

  nvidia_smi().writeln;
}
