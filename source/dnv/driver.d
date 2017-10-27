module dnv.driver;

import derelict.cuda.driverapi;
import std.stdio : writefln, writeln;
import dnv.error : nvrtcCheck, cuCheck;
static import dnv.error;


class DriverBase {
    static bool[int] initialized;

    private shared static this () {
        DerelictCUDADriver.load();
        cuInit(0);
    }

    static auto deviceInit() {
        int deviceCount = 0;
        cuDeviceGetCount(&deviceCount);
        foreach (i; 0 .. deviceCount) {
            auto result = deviceInit(i);
            if (CUDA_SUCCESS != result) {
                return result;
            }
        }
        return cast(dnv.error.CUresult) CUDA_SUCCESS;
    }

    static auto deviceInit(int id) {
        import std.string;
        if (id !in initialized) {
            int deviceCount = 0;
            cuDeviceGetCount(&deviceCount);
            if (id >= deviceCount) {
                return CUresult(CUDA_ERROR_NO_DEVICE);
            }

            int device;
            CUcontext context;
            enum int namelen = 256;
            char[namelen] name;
            cuDeviceGet(&device, id);
            cuDeviceGetName(name.ptr, namelen, device);
            // TODO: use logger
            import std.conv : to;
            writefln(">>> Using CUDA Device [%d]: %s", id, name.ptr.fromStringz);

            // get compute capabilities and the devicename
            // cudaDeviceProp dev;
            // check(cast(CUresult) cudaGetDeviceProperties(&dev, id));
            // writeln(dev);
            cuCheck(cuCtxCreate(&context, 0, device));
            initialized[id] = true;
        }
        return cast(dnv.error.CUresult) CUDA_SUCCESS;
    }
}

unittest {
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    DriverBase.deviceInit();
    assert(DriverBase.initialized.length == deviceCount);
}


CUmodule loadPTX(char *ptx, int argc, char **argv, int cuDevice)
{
    CUmodule cumodule;
    CUcontext context;

    cuCheck(cuInit(0));
    cuCheck(cuDeviceGet(&cuDevice, 0));
    cuCheck(cuCtxCreate(&context, 0, cuDevice));
    cuCheck(cuModuleLoadDataEx(&cumodule, ptx, 0U, null, null));
    return cumodule;
}


auto compile(void* kernel_addr, string funcname, string code, int cuDevice=0)
{
    import derelict.nvrtc; //  : DerelictNVRTC, nvrtcProgram, nvrtcCreateProgram, nvrtcCompileProgram;
    DerelictNVRTC.load();

    import std.stdio : writeln, writefln;
    import std.algorithm : map;
    import std.array : array, empty;
    import std.ascii : isWhite;
    import std.string : toStringz, fromStringz, strip;
    // compile
    auto filename = funcname ~ ".cu";
    nvrtcProgram prog;
    nvrtcCheck(nvrtcCreateProgram(&prog, code.toStringz, filename.toStringz, 0, null, null));
    auto opts = ["--use_fast_math", "-arch=compute_30", "--std=c++11"].map!(a => cast(const char*) a.toStringz).array;
    nvrtcCheck(nvrtcCompileProgram(prog, 3, opts.ptr));

    // dump log
    size_t logSize;
    nvrtcCheck(nvrtcGetProgramLogSize(prog, &logSize));
    char[] log;
    log.length = logSize + 1;
    nvrtcCheck(nvrtcGetProgramLog(prog, log.ptr));
    log[logSize] = '\0';
    auto slog = log.ptr.fromStringz.strip;
    if (logSize > 0 && !slog.empty) {
        writefln(">>> NVRTC log: %s\n%s", funcname, slog);
    }

    // load compiled ptx
    size_t ptxSize;
    nvrtcCheck(nvrtcGetPTXSize(prog, &ptxSize));
    char[] ptx;
    ptx.length = ptxSize;
    nvrtcCheck(nvrtcGetPTX(prog, ptx.ptr));
    nvrtcCheck(nvrtcDestroyProgram(&prog));

    // FIXME: split these to another function to return 
    CUmodule cumodule = loadPTX(ptx.ptr, 0, null, cuDevice); // ???
    cuCheck(cuModuleGetFunction(cast (CUfunction*) kernel_addr, cumodule, funcname.toStringz));
    return cast(dnv.error.nvrtcResult) NVRTC_SUCCESS;
}


auto launch(void* kernel_addr, void* kernel_args, uint* grids, uint* blocks)
                 // size_t shared_memory=0, CUstream stream=null);
{
    auto func = cast(CUfunction*) kernel_addr;
    auto args = cast(void**) kernel_args;
    void** extra = null;
    cuCheck(cuLaunchKernel(*func,
                           grids[0], grids[1], grids[2],
                           blocks[0], blocks[1], blocks[2],
                           0U, // shared, stream, // FIXME
                           null,
                           &args[0],
                           null)); // FIXME: what is this arg?
    cuCheck(cuCtxSynchronize());
    return cast(dnv.error.CUresult) CUDA_SUCCESS;
}
