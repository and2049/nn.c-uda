#include "matrix.h"
#include "cuda_kernels.h"
#include <assert.h>
#include <stdio.h>


// Static variable to control GPU usage. Default is 1 (enabled).
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

/**
 * @brief Performs matrix multiplication based on the selected mode.
 */
Matrix* matrix_multiply(const Matrix* a, const Matrix* b) {
    assert(a->cols == b->rows);
    Matrix* result = matrix_create(a->rows, b->cols);

    // If the GPU flag is enabled, attempt to use CUDA.
    if (use_gpu_flag) {
        int cuda_status = matrix_multiply_gpu(result, a, b);
        // If CUDA fails, fall back to CPU.
        if (cuda_status != 0) {
            fprintf(stderr, "WARNING: CUDA multiplication failed. Falling back to CPU.\n");
            matrix_multiply_cpu(result, a, b);
        }
    } else {
        // If the GPU flag is disabled, go directly to the CPU implementation.
        matrix_multiply_cpu(result, a, b);
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
