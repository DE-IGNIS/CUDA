#include <stdio.h>

/**
 * 02_memory_types.cu
 *
 * This file demonstrates different types of memory in CUDA:
 * 1. Global Memory (cudaMalloc, large, slow)
 * 2. Shared Memory (__shared__, small, fast, shared among threads in a block)
 * 3. Constant Memory (__constant__, cached, read-only from device)
 * 4. Local Memory (automatic variables in kernels, registers or spilled to global)
 */

// Constant memory declaration (global scope)
__constant__ float d_const_factor;

// Kernel demonstrating global, shared, and local memory
__global__ void memory_demo(float *d_global_in, float *d_global_out, int n) {
    // 1. Local Memory: Thread-private variables (usually in registers)
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Shared Memory: Fast memory shared among all threads in THIS block
    // Size must be known at compile time, or passed as 3rd launch arg (extern __shared__)
    __shared__ float shared_mem[64];

    if (gid < n) {
        // Load from Global Memory into Shared Memory
        shared_mem[tid] = d_global_in[gid];

        // Wait for all threads in the block to finish loading
        __syncthreads();

        // Some local computation using shared memory and constant memory
        // A simple moving average (using shared memory for fast access)
        float sum = shared_mem[tid];
        if (tid > 0) sum += shared_mem[tid - 1];
        if (tid < blockDim.x - 1 && gid < n - 1) sum += shared_mem[tid + 1];

        // Use constant memory factor
        float local_result = sum * d_const_factor;

        // Write back to Global Memory
        d_global_out[gid] = local_result;
    }
}

int main() {
    int n = 32; // Small array for demo
    size_t bytes = n * sizeof(float);

    float h_in[32];
    float h_out[32];

    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_in[i] = 1.0f;
    }

    // 3. Constant Memory Setup
    float h_factor = 0.5f;
    // Copy data to constant memory symbol
    cudaMemcpyToSymbol(d_const_factor, &h_factor, sizeof(float));

    // Allocate Global Memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Copy to Global Memory
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    // We use a block size of 32 (a warp), so all threads fit in one block's shared memory array
    memory_demo<<<1, 32>>>(d_in, d_out, n);

    // Wait and copy back
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Print first few results
    printf("Memory Types Demo Output:\\n");
    for (int i = 0; i < 5; i++) {
        printf("Index %d: %f\\n", i, h_out[i]);
    }
    printf("...\\n");

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
