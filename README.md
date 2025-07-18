# NN.c-uda

A simple academic implementation of a Neural Network library written in C, accelerated with NVIDIA's CUDA for matrix operations.

## Description

This project is a simple implementation of a basic neural network library. It focuses on demonstrating the performance benefits of using CUDA for parallelizing matrix operations.

The library provides fundamental components for building and training simple, fully-connected neural networks.

## Features

-   **Core Components**: Basic building blocks like matrices, layers, and activation functions.
-   **CUDA Acceleration**: Key matrix operations are offloaded to the GPU using custom CUDA kernels for significant speed-up.
-   **Simple API**: A straightforward interface for creating, training, and evaluating neural networks.
-   **Example Implementation**: Includes an example of a network that learns the XOR function.

## Project Structure

```
.
├── CMakeLists.txt
├── README.md
└── src
    ├── cuda_kernels.cu     # CUDA kernels for GPU-based matrix operations
    ├── cuda_kernels.h      # Headers for the CUDA kernels
    ├── main.c              # Benchmark program
    ├── matrix.c            # matrix functions
    ├── matrix.h            # Headers for matrix functions
    ├── nn.c                # Core neural network logic
    ├── nn.h                # Headers for the neural network
    └── xor.c               # Example: Training a network to solve XOR
```

## Requirements

-   A C Compiler (e.g., GCC, Clang)
-   CMake (version 3.18 or higher)
-   NVIDIA CUDA Toolkit (for GPU acceleration)

## Building the Project

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd nn.c-uda
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Run CMake and build the project:**
    ```bash
    cmake ..
    cmake --build .
    ```

## Usage

After building, two executables will be available in the build directory (or a subdirectory like `build/Debug` on Windows).

1.  **Run the XOR Example**:
    This program trains a small network to solve the XOR problem.
    ```bash
    ./nn_c_uda_xor
    ```

2.  **Run the Performance Benchmark**:
    This program trains a larger network on a big batch of random data, first using the GPU and then using only the CPU, to compare the time taken.
    ```bash
    ./nn_c_uda_benchmark
