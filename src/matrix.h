#ifndef NN_C_UDA_MATRIX_H
#define NN_C_UDA_MATRIX_H

#include <stdlib.h>
#include <stdio.h>


typedef struct {
    int rows;
    int cols;
    double* data; // Stored in row-major order
} Matrix;


Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix* m);
Matrix* matrix_copy(const Matrix* m);

void matrix_print(const Matrix* m);
void matrix_fill_random(Matrix* m, double range);

Matrix* matrix_multiply(const Matrix* a, const Matrix* b);
Matrix* matrix_add(const Matrix* a, const Matrix* b);
Matrix* matrix_subtract(const Matrix* a, const Matrix* b);
Matrix* matrix_transpose(const Matrix* m);
Matrix* matrix_elementwise_multiply(const Matrix* a, const Matrix* b);

void matrix_apply_function(Matrix* m, double (*func)(double));

#endif //NN_C_UDA_MATRIX_H