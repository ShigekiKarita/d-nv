#!/usr/bin/env dub
/+ dub.json:
 {
   "name": "dot_product",
   "dependencies": {"mir": "*", "d-nvrtc": "*" },
   "dflags-ldc": ["-mcpu=native"]
 }
 +/
/+
 $ ldc2 --version # cannot run executable
  based on DMD v2.071.2 and LLVM 3.7.1
  built with DMD64 D Compiler v2.071.1
  Default target: x86_64-pc-linux-gnu
  Host CPU: haswell
  http://dlang.org - http://wiki.dlang.org/LDC

 $ dub run --build=release-nobounds
 FLOAT:
 numeric.dotProduct, arrays = 60 ms, 958 μs, and 8 hnsecs
       ndReduce vectorized = 315 ms, 245 μs, and 9 hnsecs
                  ndReduce = 324 ms and 190 μs
numeric.dotProduct, slices = 351 ms, 497 μs, and 7 hnsecs
            gpuDot, arrays = 696 ms, 432 μs, and 6 hnsecs
              zip & reduce = 2 secs, 262 ms, 945 μs, and 8 hnsecs


 +/

import dnv;

import std.numeric : dotProduct;
import std.typecons;
import std.datetime;
import std.stdio;
import std.range;
import std.algorithm;
import std.conv;
import std.math;

import mir.ndslice;
import mir.ndslice.internal : fastmath;

alias F = float;

static @fastmath F fmuladd(F a, F b, F c) @safe pure nothrow @nogc
{
  return a + b * c;
}

// __gshared is used to prevent specialized optimization for input data
__gshared F result;
immutable N = 8000;
__gshared n = N;
__gshared F[] a;
__gshared F[] b;
__gshared Slice!(1, F*) asl;
__gshared Slice!(1, F*) bsl;


void main()
{
  a = iota(n).map!(to!F).array;
  b = a.dup;
  asl = a.sliced;
  bsl = b.sliced;

  auto da = new Array!float(a);
  auto db = new Array!float(b);

  enum uint gx = 256;
  enum uint bx = min(32, to!uint((gx + N - 1) / gx));
  auto dc = new Array!float(n);

  struct CustomLauncher {
    uint[3] grids = [gx, 1, 1];
    uint[3] blocks = [bx, 1, 1];

    void setup(Args...)(Args targs) {}
  }

  enum dotCode = Code
    ("dot", q{float *a, float *b, float *c, size_t n},
     q{
       // TODO: support header code
       float temp = 0;
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       int bgx = blockDim.x * gridDim.x;
       for (; idx < n; idx += bgx) {
         temp += a[idx] * b[idx];
       }

       __shared__ int cache[256]; // 256 == grids[0]
       int tid = threadIdx.x;
       cache[tid] = temp;
       __syncthreads();

       for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
         if (tid < stride) {
           cache[tid] += cache[tid + stride];
         }
         __syncthreads();
       }
       if (tid == 0) {
         c[blockIdx.x] = cache[0];
       }
     });
  auto cudot = new TypedKernel!(dotCode, CustomLauncher);

  float gpuDot(Array!float da, Array!float db) {
    cudot(da, db, dc, da.length);
    size_t cn = cudot.launch.grids[0];
    return sum(dc.to_cpu()[0 .. cn]);
  }

  float c = gpuDot(da, db);
  writefln("gpu %f == cpu %f", c, dotProduct(a, b));

  Duration[6] bestBench = Duration.max;

  foreach(_; 0 .. 10)
    {
      auto bench = benchmark!(
                              { result = ndReduce!(fmuladd, Yes.vectorized)(F(0), asl, bsl); },
                              { result = ndReduce!(fmuladd)(F(0), asl, bsl); },
                              { result = dotProduct(a, b); },
                              { result = dotProduct(a.sliced, b.sliced); },
                              { result = reduce!"a + b[0] * b[1]"(F(0), zip(a, b)); },
                              { result = gpuDot(da, db); },
                              )(2000);
      foreach(i, ref b; bestBench)
        b = min(bench[i].to!Duration, b);
    }

  writefln("%26s = %s", "ndReduce vectorized", bestBench[0]);
  writefln("%26s = %s", "ndReduce", bestBench[1]);
  writefln("%26s = %s", "numeric.dotProduct, arrays", bestBench[2]);
  writefln("%26s = %s", "numeric.dotProduct, slices", bestBench[3]);
  writefln("%26s = %s", "zip & reduce", bestBench[4]);
  writefln("%26s = %s", "gpuDot, arrays", bestBench[5]);
}
