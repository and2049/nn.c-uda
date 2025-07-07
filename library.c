#include "library.h"

Matrix* matrix_create(int rows, int columns) {
    Matrix* m = malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->rows = rows;
    m->columns = columns;
    m->data = malloc(rows * columns * sizeof(double));
    if (!m->data) {
        free(m);
        return NULL;
    }
    return m;
}

void matrix_free(Matrix* m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

void matrix_view(const Matrix* m) {
    if (m) {
        for (int i = 0; i < m->rows; i++) {
            for (int j = 0; j < m->columns; j++) {
                printf("%f", m->data[i * m->columns + j]);
            }
            printf("\n");
        }
    }
}

void matrix_randomize(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->columns; j++) {
            m->data[i * m->columns + j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

Matrix* matrix_multiply(const Matrix* a, const Matrix* b) {
    if (a->columns != b->rows) {
        fprintf(stderr, "Error: Matrix dimensions not compatible for multiplication. Matrix A is %dx%d, Matrix B is %dx%d.\n", a->rows, a->columns, b->rows, b->columns);
        return NULL;
    }
    Matrix* result = matrix_create(a->rows, b->columns);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->columns; j++) {
            double sum = 0;
            for (int k = 0; k < a->columns; k++) {
                sum += a->data[i * a->columns + k] * b->data[k * b->columns + j];
            }
            result->data[i * result->columns + j] = sum;
        }
    }
    return result;
}

Matrix* matrix_add(const Matrix* a, const Matrix* b) {
    if ((a->columns != b->columns) || (a->rows != b->rows)) {
        fprintf(stderr, "Error: Matrix dimensions not compatible for addition. Matrix A is %dx%d, Matrix B is %dx%d.\n", a->rows, a->columns, b->rows, b->columns);
        return NULL;
    }
    Matrix* result = matrix_create(a->rows, b->columns);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->columns; j++) {
            int index = i * a->columns + j;
            result->data[index] = a->data[index] + b->data[index];
        }
    }
    return result;
}

