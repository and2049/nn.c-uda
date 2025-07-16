#include "matrix.h"
#include <assert.h>
#include <time.h>

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double)); // Use calloc for zero-initialization
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
    // Note: This function should only be called after srand() is seeded in main.
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((double)rand() / RAND_MAX) * 2.0 * range - range; // Range [-range, range]
    }
}

Matrix* matrix_multiply(const Matrix* a, const Matrix* b) {
    assert(a->cols == b->rows);
    Matrix* result = matrix_create(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
    return result;
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
