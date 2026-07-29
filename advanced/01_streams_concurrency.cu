#include <stdio.h>

/**
 * 01_streams_concurrency.cu
 *
 * Demonstrates advanced concepts: CUDA Streams and Concurrency
 * CUDA streams allow overlapping of computation and data transfers
 * for better device utilization.
 */

__global__ void process_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Some arbitrary computation
        data[idx] = data[idx] * data[idx] + 1.0f;
    }
}

int main() {
    int num_streams = 4;
    int chunk_size = 1 << 18; // Size of each chunk per stream
    int n = num_streams * chunk_size;
    size_t bytes_per_chunk = chunk_size * sizeof(float);

    // Allocate pinned memory for asynchronous transfers
    float *h_data;
    cudaError_t err = cudaMallocHost((void**)&h_data, n * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to allocate pinned memory: %s\\n", cudaGetErrorString(err));
        // Fallback for systems without GPU just for compilation/runnable verification
        h_data = (float*)malloc(n * sizeof(float));
    }

    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    // Create an array of streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Initialize host data
    if (h_data != NULL) {
        for (int i = 0; i < n; i++) {
            h_data[i] = (float)(i % 100);
        }

        int blockSize = 256;
        int gridSize = (chunk_size + blockSize - 1) / blockSize;

        // Loop over chunks and overlap operations across streams
        for (int i = 0; i < num_streams; i++) {
            int offset = i * chunk_size;

            // Asynchronous copy H2D
            cudaMemcpyAsync(&d_data[offset], &h_data[offset], bytes_per_chunk, cudaMemcpyHostToDevice, streams[i]);

            // Kernel launch in specific stream
            process_kernel<<<gridSize, blockSize, 0, streams[i]>>>(&d_data[offset], chunk_size);

            // Asynchronous copy D2H
            cudaMemcpyAsync(&h_data[offset], &d_data[offset], bytes_per_chunk, cudaMemcpyDeviceToHost, streams[i]);
        }

        // Wait for all streams to finish
        cudaDeviceSynchronize();

        printf("Streams and Concurrency Demo.\\n");
        printf("First element (expected 1.0): %f\\n", h_data[0]);
    }

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_data);
    if (err == cudaSuccess) {
        cudaFreeHost(h_data);
    } else {
        free(h_data);
    }

    return 0;
}
