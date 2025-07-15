#ifndef NN_LIB_CUDA_H
#define NN_LIB_CUDA_H

struct Matrix;

struct Matrix* matrix_multiply_gpu(const struct Matrix* a, const struct Matrix* b);
struct Matrix* matrix_add_gpu(const struct Matrix* a, const struct Matrix* b);
struct Matrix* matrix_subtract_gpu(const struct Matrix* a, const struct Matrix* b);
struct Matrix* matrix_transpose_gpu(const struct Matrix* m);
struct Matrix* matrix_map_gpu(const struct Matrix* m, double (*func)(double));

void print_gpu_info();


#endif // NN_LIB_CUDA_H