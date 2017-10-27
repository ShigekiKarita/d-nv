module dnv.cuda.cublas;


extern (C):

alias cublasStatus_t = int;
enum : cublasStatus_t {
    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14,
    CUBLAS_STATUS_NOT_SUPPORTED   =15,
    CUBLAS_STATUS_LICENSE_ERROR   =16
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



version (LDC)
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
    // import std.stdio;
    // status.writeln;
    // only LDC1.4.0 pass but
    // DMD2.075.1 and GDC7.2.0 cause CUBLAS_STATUS_INTERNAL_ERROR
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

