#include "cuda_kernels.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrix_multiply_kernel(double* d_c, const double* d_a, const double* d_b, int a_rows, int a_cols, int b_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        double sum = 0.0;
        for (int k = 0; k < a_cols; ++k) {
            sum += d_a[row * a_cols + k] * d_b[k * b_cols + col];
        }
        d_c[row * b_cols + col] = sum;
    }
}

#define CUDA_CHECK(err) __cuda_check_error(err, __FILE__, __LINE__)

static void __cuda_check_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
    }
}

int matrix_multiply_gpu(Matrix* c, const Matrix* a, const Matrix* b) {
    double *d_a = NULL, *d_b = NULL, *d_c = NULL;
    cudaError_t err;

    size_t size_a = a->rows * a->cols * sizeof(double);
    err = cudaMalloc((void**)&d_a, size_a);
    if (err != cudaSuccess) { CUDA_CHECK(err); goto cleanup; }

    size_t size_b = b->rows * b->cols * sizeof(double);
    err = cudaMalloc((void**)&d_b, size_b);
    if (err != cudaSuccess) { CUDA_CHECK(err); goto cleanup; }

    size_t size_c = c->rows * c->cols * sizeof(double);
    err = cudaMalloc((void**)&d_c, size_c);
    if (err != cudaSuccess) { CUDA_CHECK(err); goto cleanup; }


    err = cudaMemcpy(d_a, a->data, size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { CUDA_CHECK(err); goto cleanup; }

    err = cudaMemcpy(d_b, b->data, size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { CUDA_CHECK(err); goto cleanup; }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((b->cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (a->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_c, d_a, d_b, a->rows, a->cols, b->cols);
    CUDA_CHECK(cudaGetLastError());

    err = cudaMemcpy(c->data, d_c, size_c, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { CUDA_CHECK(err); goto cleanup; }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;

cleanup:
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    return 1;
}
