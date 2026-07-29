#include <stdio.h>
#include <stdlib.h>

/**
 * 01_memory_optimization.cu
 *
 * Demonstrates intermediate memory concepts in CUDA:
 * 1. Pinned (Page-Locked) Memory for faster Host <-> Device transfers
 * 2. Understanding memory coalescing (global memory access patterns)
 */

__global__ void coalesced_access(float *data, int n) {
    // Standard contiguous access pattern - nicely coalesced
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void strided_access(float *data, int n, int stride) {
    // Strided access pattern - uncoalesced and slower
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    int n = 1 << 20; // ~1 million elements
    size_t bytes = n * sizeof(float);

    // Using pinned (page-locked) memory instead of regular malloc
    // This allows faster DMA transfers to/from the GPU
    float *h_data;
    cudaError_t err = cudaMallocHost((void**)&h_data, bytes); // Pinned host memory

    if (err != cudaSuccess) {
        printf("Failed to allocate pinned host memory. Error: %s\n", cudaGetErrorString(err));
        printf("Falling back to standard malloc...\n");
        h_data = (float*)malloc(bytes);
    }

    if (h_data != NULL) {
        for (int i = 0; i < n; i++) {
            h_data[i] = 1.0f;
        }

        float *d_data;
        cudaMalloc(&d_data, bytes);

        // This transfer is typically faster than standard cudaMemcpy from pageable memory
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        // Launching a coalesced memory access kernel
        coalesced_access<<<gridSize, blockSize>>>(d_data, n);

        cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

        printf("Memory Optimization Demo.\n");
        printf("First element (expected 2.0): %f\n", h_data[0]);

        // Cleanup
        cudaFree(d_data);
        if (err == cudaSuccess) {
            cudaFreeHost(h_data); // Use cudaFreeHost for pinned memory
        } else {
            free(h_data);
        }
    }

    return 0;
}
