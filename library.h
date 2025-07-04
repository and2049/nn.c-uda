#ifndef NN_C_UDA_LIBRARY_H
#define NN_C_UDA_LIBRARY_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// 2D Matrix of weights, biases, activations...
typedef struct {
    double* data;
    int rows;
    int columns;
} Matrix;

// Represents an individual layer within the neural network
typedef struct {
    Matrix* weights;
    Matrix* bias;
    void (*activation)(Matrix*);
    void (*activation_derivative)(Matrix*, Matrix*);
} Layer;

// Collection of layers.
typedef struct {
    Layer* layers;
    int num_layers;
    double learning_rate;
} Network;


/**
* @brief Creates and allocates memory for new matrix
* @param rows Number of rows
* @param columns Number of colums
* @return pointer to new matrix
*/

Matrix* matrix_create(int rows, int columns);

/**
* @brief Frees memory allocated to matrix
* @param m Matrix pointer
*/
void matrix_free(Matrix* m);

/**
* @brief Prints elements of matrix
* @param m Matrix pointer
*/
void matrix_view(const Matrix* m);


#endif //NN_C_UDA_LIBRARY_H