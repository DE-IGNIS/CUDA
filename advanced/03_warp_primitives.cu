#include <stdio.h>

/**
 * 03_warp_primitives.cu
 *
 * Demonstrates Warp-level primitives:
 * Warp shuffle instructions allow threads within the same warp (32 threads)
 * to exchange data without using shared memory, which is faster.
 */

// Warp-level reduction using shuffle down
__inline__ __device__ int warpReduceSum(int val) {
    // __shfl_down_sync shifts data from higher lanes (thread IDs within a warp) to lower lanes
    // The mask 0xffffffff means all threads in the warp participate
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val; // Lane 0 holds the sum of all threads in this warp
}

__global__ void warp_reduce_kernel(int *in, int *out, int n) {
    int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element
    if (tid < n) {
        sum = in[tid];
    }

    // Perform warp-level reduction
    sum = warpReduceSum(sum);

    // The first thread of each warp writes the result
    // threadIdx.x % warpSize == 0 identifies the first thread (lane 0) of each warp
    if ((threadIdx.x % warpSize) == 0) {
        // atomicAdd handles block-level and grid-level reduction securely
        atomicAdd(out, sum);
    }
}

int main() {
    int n = 1024; // 32 warps (since 1024/32 = 32)
    size_t bytes = n * sizeof(int);

    int *h_in = (int*)malloc(bytes);
    int h_out = 0;

    for (int i = 0; i < n; i++) {
        h_in[i] = 1; // Sum should be 1024
    }

    int *d_in, *d_out;
    cudaError_t err = cudaMalloc(&d_in, bytes);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(err));
        free(h_in);
        return 0; // Exit gracefully on non-CUDA systems
    }

    cudaMalloc(&d_out, sizeof(int));

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &h_out, sizeof(int), cudaMemcpyHostToDevice);

    // Launch a single block of 1024 threads
    warp_reduce_kernel<<<1, 1024>>>(d_in, d_out, n);

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Warp Primitives Demo.\\n");
    printf("Expected sum: 1024\\n");
    printf("Computed sum: %d\\n", h_out);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);

    return 0;
}
