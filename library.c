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