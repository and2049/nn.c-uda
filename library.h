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

} Layer;

// Collection of layers.
typedef struct {
    Layer* layers;
    int num_layers;
    double learning_rate;
} Network;


#endif //NN_C_UDA_LIBRARY_H