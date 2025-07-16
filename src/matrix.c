#include "matrix.h"
#include "cuda_kernels.h"
#include <assert.h>
#include <stdio.h>

static int use_gpu_flag = 1;

void matrix_set_gpu_mode(int enabled) {
    use_gpu_flag = enabled;
    if (use_gpu_flag) {
        printf("INFO: GPU mode has been ENABLED.\n");
    } else {
        printf("INFO: GPU mode has been DISABLED. Forcing CPU computation.\n");
    }
}

static void matrix_multiply_cpu(Matrix* result, const Matrix* a, const Matrix* b) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
}

Matrix* matrix_multiply(const Matrix* a, const Matrix* b) {
    assert(a->cols == b->rows);
    Matrix* result = matrix_create(a->rows, b->cols);

    if (use_gpu_flag) {
        int cuda_status = matrix_multiply_gpu(result, a, b);
        if (cuda_status != 0) {
            fprintf(stderr, "WARNING: CUDA multiplication failed. Falling back to CPU.\n");
            matrix_multiply_cpu(result, a, b);
        }
    } else {
        matrix_multiply_cpu(result, a, b);
    }
    return result;
}

/**
 * @brief Adds a column vector to every column of a matrix.
 * This is used to apply the bias to all samples in a batch.
 */
void matrix_broadcast_add_column(Matrix* m, const Matrix* bias_vector) {
    assert(m->rows == bias_vector->rows && bias_vector->cols == 1);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] += bias_vector->data[i];
        }
    }
}

/**
 * @brief Sums all columns of a matrix into a single column vector.
 * This is used to aggregate the bias gradients from all samples in a batch.
 */
Matrix* matrix_sum_columns(const Matrix* m) {
    Matrix* result = matrix_create(m->rows, 1);
    for (int i = 0; i < m->rows; i++) {
        double sum = 0;
        for (int j = 0; j < m->cols; j++) {
            sum += m->data[i * m->cols + j];
        }
        result->data[i] = sum;
    }
    return result;
}

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double));
    return m;
}

void matrix_free(Matrix* m) {
    if (m != NULL) {
        free(m->data);
        free(m);
    }
}

Matrix* matrix_copy(const Matrix* m) {
    Matrix* copy = matrix_create(m->rows, m->cols);
    for(int i = 0; i < m->rows * m->cols; i++) {
        copy->data[i] = m->data[i];
    }
    return copy;
}

void matrix_print(const Matrix* m) {
    if (m == NULL) {
        printf("NULL Matrix\n");
        return;
    }
    printf("Matrix (%d rows, %d cols):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%9.4f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_fill_random(Matrix* m, double range) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((double)rand() / RAND_MAX) * 2.0 * range - range;
    }
}

Matrix* matrix_add(const Matrix* a, const Matrix* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    Matrix* result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

Matrix* matrix_subtract(const Matrix* a, const Matrix* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    Matrix* result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

Matrix* matrix_transpose(const Matrix* m) {
    Matrix* result = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[j * result->cols + i] = m->data[i * m->cols + j];
        }
    }
    return result;
}

Matrix* matrix_elementwise_multiply(const Matrix* a, const Matrix* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    Matrix* result = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

void matrix_apply_function(Matrix* m, double (*func)(double)) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = func(m->data[i]);
    }
}
