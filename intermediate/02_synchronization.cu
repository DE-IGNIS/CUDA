#include <stdio.h>

/**
 * 02_synchronization.cu
 *
 * Demonstrates synchronization techniques in CUDA:
 * 1. Block-level synchronization with __syncthreads()
 * 2. Global synchronization using atomic operations (atomicAdd)
 */

__global__ void sum_reduction_atomic(int *d_in, int *d_out, int n) {
    // Compute local sum for each thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for block-level reduction
    extern __shared__ int sdata[];

    // Load data into shared memory
    if (idx < n) {
        sdata[threadIdx.x] = d_in[idx];
    } else {
        sdata[threadIdx.x] = 0;
    }

    // Wait for all threads in the block to load their data
    __syncthreads();

    // Do a reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        // Wait for all threads to finish this step before proceeding
        __syncthreads();
    }

    // Thread 0 of each block writes the block sum to global memory
    if (threadIdx.x == 0) {
        // Use atomicAdd to safely add the block sum to the global total
        atomicAdd(d_out, sdata[0]);
    }
}

int main() {
    int n = 1000;
    size_t bytes = n * sizeof(int);

    int *h_in = (int*)malloc(bytes);
    int h_out = 0;

    for (int i = 0; i < n; i++) {
        h_in[i] = 1; // Sum should be exactly 1000
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(int));

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &h_out, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel with dynamic shared memory (3rd parameter)
    sum_reduction_atomic<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_in, d_out, n);

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Expected sum: %d\\n", n);
    printf("Computed sum: %d\\n", h_out);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);

    return 0;
}
