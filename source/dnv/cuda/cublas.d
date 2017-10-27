module dnv.cuda.cublas;



import derelict.util.loader;

private
{
    import derelict.util.system;

    static if(Derelict_OS_Windows)
        enum libNames = "nvcublas.dll";
    else static if (Derelict_OS_Mac)
        enum libNames = "libcublas.dylib";
    else static if (Derelict_OS_Linux)
        enum libNames = "libcublas.so";
    else
        static assert(0, "Need to implement CUDA libNames for this operating system.");
}

extern (C) {

alias cublasStatus_t = int;
enum : cublasStatus_t {
    CUBLAS_STATUS_SUCCESS, // The operation completed successfully.

    CUBLAS_STATUS_NOT_INITIALIZED, // The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup. To correct: call cublasCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.

    CUBLAS_STATUS_ALLOC_FAILED,  //Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure. To correct: prior to the function call, deallocate previously allocated memory as much as possible.

    CUBLAS_STATUS_INVALID_VALUE, // An unsupported value or parameter was passed to the function (a negative vector size, for example). To correct: ensure that all the parameters being passed have valid values.

    CUBLAS_STATUS_ARCH_MISMATCH, // The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision. To correct: compile and run the application on a device with appropriate compute capability, which is 1.3 for double precision.

    CUBLAS_STATUS_MAPPING_ERROR, // An access to GPU memory space failed, which is usually caused by a failure to bind a texture. To correct: prior to the function call, unbind any previously bound textures.

    CUBLAS_STATUS_EXECUTION_FAILED, // The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons. To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.

    CUBLAS_STATUS_INTERNAL_ERROR, //An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure. To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.

    CUBLAS_STATUS_NOT_SUPPORTED, // The functionnality requested is not supported

    CUBLAS_STATUS_LICENSE_ERROR // The functionnality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly. 
}

struct cublasContext;
alias cublasHandle_t = cublasContext*;

alias cublasOperation_t = int;
enum : cublasOperation_t {
    CUBLAS_OP_N, // the non-transpose operation is selected
    CUBLAS_OP_T, // the transpose operation is selected
    CUBLAS_OP_C // the conjugate transpose operation is selected
}


// TODO: parse and retrieve cublas_api.h
cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc);


}

unittest {
    import dnv.storage : Array;
    import dnv.cuda.cublas;

    cublasHandle_t handle;
    auto status = cublasCreate_v2(&handle);
    scope(exit) cublasDestroy_v2(handle);

    assert(status == CUBLAS_STATUS_SUCCESS);

    float[] A = [1, 2, 3,
                 4, 5, 6]; // M=3 x K=2
    float[] B = [1, 2,
                 3, 4,
                 5, 6,
                 7, 8]; // N=4 x k=2
    auto M = 3;
    auto N = 4;
    auto K = 2;
    float alpha = 1.0f;
    float beta = 0.0f;
    auto d_A = new Array!float(A);
    auto d_B = new Array!float(B);
    auto d_C = new Array!float(M * N);

    // cublas driver API
    status = cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
                            &alpha, d_A.data, M, d_B.data, N, &beta, d_C.data, M);
    assert(status == CUBLAS_STATUS_SUCCESS);

    auto d_D = new Array!float(N * M);
    status = cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
                            &alpha, d_B.data, N, d_A.data, M, &beta, d_D.data, N);
    assert(status == CUBLAS_STATUS_SUCCESS);

    // check C = D.T
    auto C = d_C.to_cpu();     // C = A x B.T
    auto D = d_D.to_cpu();     // D = B x A.T
    foreach (m; 0 .. M) {
        foreach (n; 0 .. N) {
            assert(C[m + n * M] == D[n + m * N]);
        }
    }
}

class DerelictCublasLoader : SharedLibLoader {
    this () {
        super(libNames);
    }

    protected override void loadSymbols() {
        bindFunc(cast(void**)&cublasSgemm_v2, "cublasSgemm_v2");
    }
}


__gshared DerelictCublasLoader DerelictCublas;

shared static this() {
    DerelictCublas = new DerelictCublasLoader();
}
