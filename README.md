# d-nvrtc

just work in progress

roadmap

1. (DONE) allocate memory on multiple devices with CUDA Driver API
1. compile a kernel of raw string with NVRTC
1. launch a kernel function
1. mult-device support
1. pure D code

issues

+ fix the CUdeviceptr definition to use cuMemAlloc directly
+ think about how to cleanup resources
