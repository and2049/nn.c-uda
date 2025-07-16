#ifndef NN_C_UDA_NN_H
#define NN_C_UDA_NN_H

#include "matrix.h"

// --- Activation Function Pointer Type ---
// Simplifies declaring functions and passing them as arguments.
typedef double (*activation_func)(double);

// --- Activation Functions & Derivatives ---
// All derivative functions now take z (the pre-activation value) as input.
double sigmoid(double x);
double sigmoid_derivative(double z);
double relu(double x);
double relu_derivative(double z);
double tanh_activation(double x);
double tanh_derivative(double z);

typedef struct {
    Matrix* weights;
    Matrix* biases;
    Matrix* z;       // Pre-activation values: W*a_prev + b
    Matrix* a;       // Post-activation values: activation(z)

    activation_func activation;
    activation_func activation_derivative;
} Layer;

typedef struct {
    int num_layers;
    int* layer_sizes;
    Layer* layers;
} NeuralNetwork;

NeuralNetwork* nn_create(int num_layers, int* layer_sizes, activation_func* activations, activation_func* activation_derivatives);
void nn_free(NeuralNetwork* nn);

Matrix* forward_propagation(NeuralNetwork* nn, const Matrix* input);
void backpropagation(NeuralNetwork* nn, const Matrix* x, const Matrix* y, double learning_rate);
void nn_train(NeuralNetwork* nn, const Matrix* x_train, const Matrix* y_train, int epochs, double learning_rate);

#endif //NN_C_UDA_NN_H
