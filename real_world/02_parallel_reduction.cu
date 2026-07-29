#include <stdio.h>
#include <stdlib.h>

/**
 * 02_parallel_reduction.cu
 *
 * Demonstrates a tree-based parallel reduction sum.
 * This is a fundamental algorithm in parallel computing used to sum
 * (or find max, min) of a large array efficiently.
 */

#define BLOCK_SIZE 256

__global__ void reduce_sum_kernel(float *in, float *out, int n) {
    // Allocate shared memory for the block
    __shared__ float sdata[BLOCK_SIZE];

    // Each thread loads two elements from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load first element
    if (i < n) {
        sdata[tid] = in[i];
    } else {
        sdata[tid] = 0.0f;
    }

    // Load second element and add
    if (i + blockDim.x < n) {
        sdata[tid] += in[i + blockDim.x];
    }

    __syncthreads();

    // Do reduction in shared memory (tree-based)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 1 << 20; // 1 million elements
    size_t bytes = n * sizeof(float);

    float *h_in = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_in[i] = 1.0f; // Sum should be exactly n
    }

    float *d_in;
    cudaError_t err = cudaMalloc(&d_in, bytes);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(err));
        free(h_in);
        return 0; // Exit gracefully
    }

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Grid setup: Each block processes 2 * BLOCK_SIZE elements
    int grid_size = (n + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);

    float *d_out;
    cudaMalloc(&d_out, grid_size * sizeof(float));

    // First pass
    reduce_sum_kernel<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, n);

    // If the grid_size > 1, we need to reduce the partial sums again
    // For simplicity, if grid_size is small, we just copy back and finish on CPU
    float *h_out = (float*)malloc(grid_size * sizeof(float));
    cudaMemcpy(h_out, d_out, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    float final_sum = 0.0f;
    for (int i = 0; i < grid_size; i++) {
        final_sum += h_out[i];
    }

    printf("Parallel Reduction Demo.\\n");
    printf("Expected sum: %f\\n", (float)n);
    printf("Computed sum: %f\\n", final_sum);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
