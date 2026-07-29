#include <stdio.h>
#include <stdlib.h>

/**
 * 01_matrix_multiplication.cu
 *
 * Demonstrates a real-world optimization: Tiled Matrix Multiplication
 * Uses shared memory to cache sub-matrices (tiles), reducing global memory bandwidth.
 */

#define TILE_SIZE 16

// Basic Matrix Multiplication (for comparison)
__global__ void matMulBasic(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; ++i) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Tiled Matrix Multiplication using Shared Memory
__global__ void matMulTiled(float *A, float *B, float *C, int width) {
    // Shared memory tiles for A and B
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Global row and column index
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles needed to compute the output
    int numTiles = (width + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A into shared memory (with bounds checking)
        if (row < width && t * TILE_SIZE + tx < width) {
            sA[ty][tx] = A[row * width + t * TILE_SIZE + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory (with bounds checking)
        if (t * TILE_SIZE + ty < width && col < width) {
            sB[ty][tx] = B[(t * TILE_SIZE + ty) * width + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        // Wait for the entire tile to be loaded before computation
        __syncthreads();

        // Compute dot product for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }

        // Wait for all threads to finish using the tile before loading the next one
        __syncthreads();
    }

    // Write the result
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

int main() {
    int width = 256; // 256x256 matrix
    size_t size = width * width * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < width * width; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f; // A * B should result in all elements being width * 2 = 512.0
    }

    float *d_A, *d_B, *d_C;
    cudaError_t err = cudaMalloc(&d_A, size);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(err));
        free(h_A); free(h_B); free(h_C);
        return 0; // Exit gracefully
    }

    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((width + TILE_SIZE - 1) / TILE_SIZE, (width + TILE_SIZE - 1) / TILE_SIZE);

    // Launch tiled kernel
    matMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, width);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Tiled Matrix Multiplication Demo.\\n");
    printf("Matrix size: %dx%d\\n", width, width);
    printf("Expected result element: %f\\n", (float)(width * 2));
    printf("Computed result element: %f\\n", h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
