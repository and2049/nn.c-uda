#ifndef NN_C_UDA_CUDA_KERNELS_H
#define NN_C_UDA_CUDA_KERNELS_H

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif


    int matrix_multiply_gpu(Matrix* c, const Matrix* a, const Matrix* b);

#ifdef __cplusplus
}
#endif

#endif //NN_C_UDA_CUDA_KERNELS_H
