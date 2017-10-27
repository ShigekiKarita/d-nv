module dnv.storage;

import std.conv : to;

import dnv.error;
import dnv.driver;

import derelict.cuda.driverapi; // : cuMemAlloc, cuMemcpyDtoH, cuMemcpyHtoD, cuMemFree, CUdeviceptr, cuCtxSynchronize;

class Array(T) : DriverBase {
    alias Element = T;
    alias Storage = T*;

    int dev;
    size_t rawLength;
    size_t length;
    CUdeviceptr ptr;
    T[] cpu_storage;
    this(size_t n, int dev = 0) {
        dev = dev;
        length = n;
        rawLength = T.sizeof * n;
        cuCheck(this.deviceInit(dev));
        cuCheck(cuMemAlloc(&ptr, rawLength));
    }
    this(in T[] src, int dev = 0) {
        this(src.length);
        to_gpu(src);
    }
    ~this() {
        cuCheck(cuMemFree(ptr));
    }
    auto to_gpu(in T[] src) {
        // TODO: use cuMemcpyHtoDAsync and CUstream
        cuCheck(cuMemcpyHtoD(ptr, to!(const(void*))(src.ptr), rawLength));
        return this;
    }
    T[] to_cpu() {
        cpu_storage.length = rawLength / T.sizeof;
        // check(cudaDeviceInit_(dev));
        cuCheck(cuMemcpyDtoH(to!(void*)(cpu_storage.ptr), ptr, rawLength));
        return cpu_storage;
    }

    auto data() {
        return cast(T*) this.ptr;
    }
}

unittest {
    float[] h1 = [1,2,3];
    auto d = new Array!float(h1.length);

    auto a1 = d.to_gpu(h1).to_cpu();
    assert(h1 == a1);

    float[] h2 = [3,2,1];
    d.to_gpu(h2);
    auto hd = d.to_cpu();
    assert(h2 == hd);
    static assert(is(typeof(d).Storage == float*));
    static assert(is(typeof(d).Element == float));
    delete d;
}
