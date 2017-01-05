import nv = nvrtc;

void main() {
  int n = 100;
  auto a = new nv.Array!float(n);
  auto b = new nv.Array!float(n);
  auto c = new nv.Array!float(n);

  nv.Kernel kernel = nv.rtc(
`
extern "C" __global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] + B[i];
}`);

  int threads[] = [256, 1, 1];
  int blocks[] = [(n + threads[0] - 1) / threads[0], 1, 1];
  nv.call(kernel, threads, blocks, a, b, c, n);
}
