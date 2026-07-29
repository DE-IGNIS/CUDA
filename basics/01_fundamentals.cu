#include <stdio.h>

/**
 * 01_fundamentals.cu
 *
 * This file covers the very basics of CUDA programming:
 * 1. Defining a kernel function that runs on the GPU (__global__)
 * 2. Launching a kernel with a grid and block dimension (<<<grid, block>>>)
 * 3. Understanding the thread hierarchy (threadIdx, blockIdx, blockDim, gridDim)
 * 4. Basic error handling for kernel launches
 */

// A simple kernel that prints the ID of each thread
__global__ void hello_cuda() {
    // threadIdx.x is the ID of the thread within its block
    // blockIdx.x is the ID of the block within the grid
    // blockDim.x is the number of threads per block

    // Calculate a unique global thread ID
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from block %d, thread %d (Global ID: %d)\\n",
           blockIdx.x, threadIdx.x, global_id);
}

// A basic vector addition kernel to demonstrate memory transfer
__global__ void add_vectors(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    printf("--- Part 1: Hello CUDA ---\\n");
    // Launch hello_cuda with 2 blocks, each containing 4 threads
    hello_cuda<<<2, 4>>>();

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish before proceeding
    cudaDeviceSynchronize();

    printf("\\n--- Part 2: Vector Addition ---\\n");
    int n = 10;
    size_t bytes = n * sizeof(int);

    // Host data
    int h_a[10], h_b[10], h_c[10];
    for(int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device data pointers
    int *d_a, *d_b, *d_c;

    // Allocate memory on the GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Calculate grid size based on number of elements and block size
    int blockSize = 4;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    add_vectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print result
    for(int i = 0; i < n; i++) {
        printf("%d + %d = %d\\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
