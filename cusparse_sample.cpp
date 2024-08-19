#include <iostream>
#include <cusparse.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) {                                \
    cudaError_t err = call;                               \
    if(err != cudaSuccess) {                              \
        std::cerr << "CUDA error in " << __FILE__         \
                  << " at line " << __LINE__ << ": "      \
                  << cudaGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);                               \
    }                                                     \
}

#define CHECK_CUSPARSE(call) {                            \
    cusparseStatus_t status = call;                       \
    if(status != CUSPARSE_STATUS_SUCCESS) {               \
        std::cerr << "cuSPARSE error in " << __FILE__     \
                  << " at line " << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);                               \
    }                                                     \
}

int main() {
    // Define the problem size
    const int N = 4;  // Number of ODEs

    // Time step and number of steps
    const double dt = 0.01;
    const int num_steps = 100;

    // Define a simple sparse matrix A in CSR format (4x4 matrix)
    int h_csrRowPtr[N+1] = {0, 2, 4, 6, 8};  // Row pointer
    int h_csrColInd[8] = {0, 1, 1, 2, 2, 3, 3, 0};  // Column indices
    double h_csrVal[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};  // Non-zero values

    // Initial conditions for y
    double h_y[N] = {1.0, 0.0, 0.0, 0.0};

    // Device memory allocation
    int *d_csrRowPtr, *d_csrColInd;
    double *d_csrVal, *d_y, *d_y_new;
    CHECK_CUDA(cudaMalloc(&d_csrRowPtr, (N+1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csrColInd, 8 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csrVal, 8 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y_new, N * sizeof(double)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (N+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrColInd, h_csrColInd, 8 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrVal, h_csrVal, 8 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice));

    // cuSPARSE handle
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Sparse matrix-vector multiplication descriptor
    cusparseMatDescr_t descr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    // Time integration using Euler method
    for (int step = 0; step < num_steps; ++step) {
        double alpha = dt;
        double beta = 1.0;

        // d_y_new = beta * d_y + alpha * A * d_y
        CHECK_CUSPARSE(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      N, N, 8, &alpha, descr, d_csrVal, d_csrRowPtr,
                                      d_csrColInd, d_y, &beta, d_y_new));

        // Copy d_y_new back to d_y for the next step
        CHECK_CUDA(cudaMemcpy(d_y, d_y_new, N * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Output the result
    std::cout << "y after " << num_steps << " steps:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }

    // Clean up
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    cudaFree(d_y);
    cudaFree(d_y_new);

    return 0;
}
