#ifndef NN_LIB_H
#define NN_LIB_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef enum {
    RELU,
    SIGMOID,
    SOFTMAX
} ActivationType;

// 2D Matrix of weights, biases, activations...
typedef struct {
    double* data;
    int rows;
    int columns;
} Matrix;

// Represents an individual layer within the neural network
typedef struct {
    Matrix* weights;
    Matrix* biases;
    void (*activation)(Matrix*);
    void (*activation_derivative)(Matrix*, Matrix*);
} Layer;

// Collection of layers.
typedef struct {
    Layer* layers;
    int num_layers;
    double learning_rate;
} Network;

// Matrix functions

Matrix* matrix_create(int rows, int columns);
void matrix_free(Matrix* m);
void matrix_view(const Matrix* m);
void matrix_randomize(Matrix* m);

Matrix* matrix_multiply(const Matrix* a, const Matrix* b);
Matrix* matrix_add(const Matrix* a, const Matrix* b);
Matrix* matrix_subtract(const Matrix* a, const Matrix* b);
Matrix* matrix_transpose(const Matrix* m);

void matrix_map(Matrix* m, double (*func) (double));

// Activation functions
double sigmoid(double x);
double sigmoid_derivative(double x);
void activate_sigmoid(Matrix* m);
void derivative_sigmoid(Matrix* activated, Matrix* derivative);

double relu(double x);
double relu_derivative(double x);
void activate_relu(Matrix* m);
void derivative_relu(Matrix* activated, Matrix* derivative);

void activate_softmax(Matrix* m);

// Network logic
Network* network_create(int* layer_sizes, int num_layers,ActivationType* activation_functions, double learning_rate);
void network_free(Network* net);
Matrix* network_forward(Network* net, const Matrix* input);
void network_backprop(Network* net, const Matrix* input, const Matrix* target);
void network_train(Network* net, Matrix** inputs, Matrix** target, int num_samples, int epochs);

#endif //NN_C_UDA_LIBRARY_H