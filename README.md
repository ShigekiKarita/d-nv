# d-nvrtc

just work in progress

## roadmap

(- v1.0)
1. (DONE) allocate memory on multiple devices with CUDA Driver API
1. (DONE) GPU device <-> CPU host memory transfer
1. (DONE) compile a kernel of raw string with NVRTC
1. (WIP) launch a kernel function
1. support template kernels

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

