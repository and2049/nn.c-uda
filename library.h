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

// Matrix functions

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

/**
 * @brief Fills matrix with random values [-1.0,1.0]
 * @param m Matrix pointer
*/
void matrix_randomize(Matrix* m);

/**
* @brief Matrix multiplication
* @param a First matrix
* @param b Second matrix
* @return product of two matrices
*/
Matrix* matrix_multiply(const Matrix* a, const Matrix* b);

/**
* @brief Matrix addition
* @param a First matrix
* @param b Second matrix
* @return sum of two matrices
*/
Matrix* matrix_add(const Matrix* a, const Matrix* b);

/**
* @brief Matrix subtraction
* @param a First matrix
* @param b Second matrix
* @return difference of two matrices
*/
Matrix* matrix_subtract(const Matrix* a, const Matrix* b);

/**
* @brief Matrix transpose, interchanging rows and columns of original matrix
* @param m Matrix
* @return transpose of matrix m
*/
Matrix* matrix_transpose(const Matrix* m);

/**
* @brief Applies a function to each element of a matrix
* @param m Matrix
* @param func Activation function to apply to matrix element
*/
void matrix_map(Matrix* m, double (*func) (double));

// Activation functions

/**
* @brief Sigmoid activation function
* @param x input value
* @return sigmoid value of x
*/
double sigmoid(double x);

/**
* @brief Derivative of sigmoid function
* @param x input value
* @return derivative of sigmoid at original value
*/
double sigmoid_derivative(double x);

/**
* @brief Applies sigmoid function to each element of a matrix
* @param m Matrix
*/
void activate_sigmoid(Matrix* m);

/**
* @brief Computes derivative of sigmoid function and applies value to derivative matrix
* @param activated Matrix following sigmoid activation
* @param derivative Output matrix storing derivatives
*/
void derivative_sigmoid(Matrix* activated, Matrix* derivative);

/**
* @brief Rectified linear unit activation
* @param x input value
* @return ReLu of x
*/
double relu(double x);

/**
 * @brief Derivative of ReLu function
 * @param x input value
 * @return dertivative of ReLu at x
*/
double relu_derivative(double x);

/**
* @brief Applies ReLu function to elements of a matrix
* @param m matrix
*/
void activate_relu(Matrix* m);

/**
* @brief Computes derivative
* @param activated Matrix after activation
* @param derivative Output matrix to store derivative values
*/
void derivative_relu(Matrix* activated, Matrix* derivative);

/**
 * @brief Creates neural network
 * @param layer_sizes Array of neurons
 * @param num_layers Total number of layers, including input and output layers
 * @param learning_rate Learning rate for gradient descent
 * @return pointer to network
*/
Network* network_create(int* layer_sizes, int num_layers, double learning_rate);

/**
* @brief Frees memory associated with neural network
* @param n Network to free
*/
void network_free(Network* n);

/**
*
*/
Matrix* network_forward(Network* n, const Matrix* input);

/**
 *
*/
void network_backprop(Network* n, const Matrix* input, Matrix* target);

/**
*
*/
void nn_train(Network* n, Matrix** inputs, Matrix** target, int num_samples, int epochs);
#endif //NN_C_UDA_LIBRARY_H