#include <stdio.h>

/**
 * 02_unified_memory.cu
 *
 * Demonstrates Unified Memory in CUDA:
 * 1. cudaMallocManaged simplifies memory management by providing a single pointer
 *    accessible from both CPU and GPU.
 * 2. cudaMemPrefetchAsync provides hints to migrate data before it's needed,
 *    optimizing performance (avoiding page faults).
 */

__global__ void unified_kernel(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + y[idx];
    }
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float *x, *y;

    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaError_t err_x = cudaMallocManaged(&x, bytes);
    cudaError_t err_y = cudaMallocManaged(&y, bytes);

    if (err_x != cudaSuccess || err_y != cudaSuccess) {
        printf("Unified memory allocation failed: %s\\n", cudaGetErrorString(err_x));
        return 0; // Exit gracefully on non-CUDA systems
    }

    // Initialize data on the CPU
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Get the active device ID
    int deviceId;
    cudaGetDevice(&deviceId);

    // Prefetch the data to the GPU (optional, but good for performance)
    cudaMemPrefetchAsync(x, bytes, deviceId);
    cudaMemPrefetchAsync(y, bytes, deviceId);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    unified_kernel<<<gridSize, blockSize>>>(x, y, n);

    // Wait for the GPU to finish before CPU access
    cudaDeviceSynchronize();

    // Prefetch the data back to the CPU (cudaCpuDeviceId)
    cudaMemPrefetchAsync(y, bytes, cudaCpuDeviceId);

    printf("Unified Memory Demo.\\n");
    printf("First element (expected 3.0): %f\\n", y[0]);

    // Free unified memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
