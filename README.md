# CUDA Basics and Beyond

Welcome to the **CUDA Basics and Beyond** repository! This repository is designed to be a comprehensive learning resource for NVIDIA CUDA programming. It takes you on a structured learning path from the absolute basics to intermediate optimizations, and finally to advanced concepts and real-world examples.

## 🗂️ Repository Structure

The code is logically separated by topic to guide your learning:

### 1. Basics (`basics/`)
Start here to understand the core concepts of CUDA programming.
* **`01_fundamentals.cu`**: Introduces GPU architecture basics, the thread hierarchy (grid, block, thread), and how to launch basic kernels.
* **`02_memory_types.cu`**: Explores the different memory spaces available on the GPU: Global, Shared, Local, and Constant memory.

### 2. Intermediate (`intermediate/`)
Once you are comfortable with the basics, move on to writing more efficient code.
* **`01_memory_optimization.cu`**: Demonstrates the performance benefits of coalesced memory access and the use of pinned (page-locked) host memory.
* **`02_synchronization.cu`**: Covers block-level synchronization (`__syncthreads()`) and global atomic operations (`atomicAdd`).

### 3. Advanced (`advanced/`)
Dive into more complex CUDA features that allow for maximum hardware utilization.
* **`01_streams_concurrency.cu`**: Shows how to use CUDA Streams to overlap computation with memory transfers.
* **`02_unified_memory.cu`**: Demonstrates `cudaMallocManaged` for unified CPU/GPU memory access and prefetching for performance.
* **`03_warp_primitives.cu`**: Explores warp-level programming (`__shfl_down_sync`) to allow fast data exchange between threads within the same warp without using shared memory.

### 4. Real World (`real_world/`)
See how the concepts you've learned are applied to standard parallel computing problems.
* **`01_matrix_multiplication.cu`**: Implements a tiled matrix multiplication algorithm, a classic optimization using shared memory to reduce global memory bandwidth.
* **`02_parallel_reduction.cu`**: Implements a tree-based parallel reduction, fundamental for operations like finding the sum, min, or max of a massive array.

### 5. Legacy Notebooks (`notebooks/`)
* Contains the original Jupyter Notebooks (`Basics.ipynb`, `Problems.ipynb`) that this repository was built upon, preserved for reference.

---

## 🛠️ How to Compile and Run

To run these examples, you will need a system with an NVIDIA GPU and the CUDA Toolkit installed.

You can compile any of the `.cu` files using the NVIDIA CUDA Compiler (`nvcc`):

```bash
# Example: Compiling the fundamentals file
cd basics
nvcc 01_fundamentals.cu -o fundamentals

# Run the compiled executable
./fundamentals
```

Each file contains complete, runnable examples and is heavily commented. We recommend reading through the source code comments before running the executables.

---

## 🎯 Learning Path Recommendation

If you are self-studying, follow the directories in order:

1. Read and run all files in `basics/`.
2. Move to `intermediate/`, focusing on how memory layout affects performance.
3. Understand `advanced/` concepts, paying special attention to how warps execute instructions.
4. Finally, study `real_world/` to see how shared memory and reductions are put together in practical algorithms.

Happy coding!
