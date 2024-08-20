#include <iostream>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

// Define the function f(t, y) for the system of ODEs dy/dt = f(t, y)
__global__ void ODEFunction(double *y, double *dy, double t, int n) {
    // ODE system: dy/dt = A * y, where A is a diagonal matrix with different coefficients
    for (int i = 0; i < n; ++i) {
        switch (i) {
            case 0: dy[i] = -2.0 * y[i]; break;   // dy1/dt = -2 * y1
            case 1: dy[i] = -1.0 * y[i]; break;   // dy2/dt = -y2
            case 2: dy[i] = -0.5 * y[i]; break;   // dy3/dt = -0.5 * y3
            case 3: dy[i] = -0.25 * y[i]; break;  // dy4/dt = -0.25 * y4
            case 4: dy[i] = -0.1 * y[i]; break;   // dy5/dt = -0.1 * y5
        }
    }
}

// Solve the system of ODEs using BDF
void solveODEBDF(double t0, double t1, double *y0, int n, int steps) {
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    double dt = (t1 - t0) / steps;
    double t = t0;

    // Allocate memory for y and dy
    double *d_y, *d_dy;
    cudaMalloc(&d_y, n * sizeof(double));
    cudaMalloc(&d_dy, n * sizeof(double));

    // Initialize y with initial conditions
    cudaMemcpy(d_y, y0, n * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < steps; ++i) {
        // Compute dy = f(t, y)
        ODEFunction<<<1, n>>>(d_y, d_dy, t, n);
        cudaDeviceSynchronize();

        // Solve the linear system (I - dt * J) * y_new = y_old
        // For this example, we assume J is identity and dy is small, so BDF simplifies to:
        // y_new = y_old + dt * f(t, y)

        double alpha = dt;
        cublasDaxpy(cublasHandle, n, &alpha, d_dy, 1, d_y, 1); // y_new = y_old + dt * dy

        // Update time
        t += dt;
    }

    // Copy the result back to host
    cudaMemcpy(y0, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_y);
    cudaFree(d_dy);
    cublasDestroy(cublasHandle);
}

int main() {
    double t0 = 0.0;
    double t1 = 1.0;
    int n = 5;  // Number of ODEs
    int steps = 10;

    // Initial conditions for the 5 ODEs
    std::vector<double> y0 = {1.0, 1.0, 1.0, 1.0, 1.0};
    for(int loop=0; loop < 100; loop++){
        solveODEBDF(t0, t1, y0.data(), n, steps);
        for (int i = 0; i < n; ++i) {
        std::cout << "y" << i + 1 << " = " << y0[i] << std::endl;
    }
        printf("\n");
        t0 = t1;
        t1 = t1+0.5;
    }
    

    // Output the solutions at t = t1
    std::cout << "Solutions at t = " << t1 << " are:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "y" << i + 1 << " = " << y0[i] << std::endl;
    }

    return 0;
}
