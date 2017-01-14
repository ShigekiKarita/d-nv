# d-nvrtc

[![Coverage Status](https://coveralls.io/repos/github/ShigekiKarita/d-nvrtc/badge.svg?branch=master)](https://coveralls.io/github/ShigekiKarita/d-nvrtc?branch=master)

just work in progress

usage (from [example/app.d](/example/app.d))

``` d
import dnv;

import std.stdio;
import std.random;
import std.range;

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
auto saxpy = new TypedKernel!(code);
saxpy(a, b, c, n); // type-checked at compile-time. 
// compile error: saxpy(a, b, c), saxpy(a, b, c, 3f)

foreach (ai, bi, ci; zip(a.to_cpu(), b.to_cpu(), c.to_cpu())) {
  assert(ai + bi == ci);
}
```

## how to use

install as follows

```
$ git clone https://github.com/ShigekiKarita/d-nvrtc.git
$ cd d-nvrtc && make all 
$ dub add-local .
$ export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
$ export LIBRARY_PATH=`pwd`:$LIBRARY_PATH
```

add bellow to your project file `dub.json`
``` json
    "dependencies": {
        "d-nvrtc": "*"
    }
```
and then `$ dub run`

or add this header to your single app.d
``` json
#!/usr/bin/env dub
/+ dub.json:
{
    "name": "your-app",
    "targetType":"executable",
    "dependencies": {
        "d-nvrtc": "*"
    }
}
+/
```

and then `$ dub app.d`

## roadmap

(- v1.0)

1. (DONE) allocate memory on multiple devices with CUDA Driver API
1. (DONE) GPU device <-> CPU host memory transfer
1. (DONE) compile a kernel of raw string with NVRTC
1. (DONE) launch a kernel function
1. (DONE) type-check of kernel's arguments at compile-time
1. (DONE) build with dub
1. (DONE) Coveralls support using [doveralls](https://github.com/ColdenCullen/doveralls)
1. (WIP) add benchmark and example using d-nvrtc as a library
1. (WIP) naive type-check of kernel's arguments at run-time
1. (WIP) user-friendly config of `<<<grids, blocks, shared-memory, stream>>>`
1. support template kernels
1. support static compilation of CUDA kernel (just linking objects without NVRTC?)

(v1.0 -)

1. thrust support
1. fully multi-device support
1. more D code ratio

## issues

+ fix the CUdeviceptr definition to use cuMemAlloc directly
+ think about how to cleanup resources
+ add unit tests

## referrence

+ /usr/local/cuda/samples/0_Simple/ vectorAdd_nvrtc, simpleTemplates_nvrtc 
+ /usr/local/cuda/doc/pdf/NVRTC_User_Guide.pdf
+ /usr/local/cuda/doc/pdf/


## development

how to build and unittest 

```
$ make coverage
```

[current coverage](coverage)
