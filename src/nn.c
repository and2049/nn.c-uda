#include "nn.h"
#include <math.h>
#include <assert.h>


double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double z) { double s = sigmoid(z); return s * (1.0 - s); }
double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double z) { return z > 0 ? 1.0 : 0.0; }
double tanh_activation(double x) { return tanh(x); }
double tanh_derivative(double z) { double t = tanh(z); return 1.0 - t * t; }

NeuralNetwork* nn_create(int num_layers, int* layer_sizes, activation_func* activations, activation_func* activation_derivatives) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for(int i = 0; i < num_layers; i++) { nn->layer_sizes[i] = layer_sizes[i]; }
    nn->layers = (Layer*)malloc((num_layers - 1) * sizeof(Layer));
    for (int i = 0; i < num_layers - 1; i++) {
        Layer* layer = &nn->layers[i];
        layer->weights = matrix_create(layer_sizes[i+1], layer_sizes[i]);
        layer->biases = matrix_create(layer_sizes[i+1], 1);
        layer->activation = activations[i];
        layer->activation_derivative = activation_derivatives[i];
        matrix_fill_random(layer->weights, 1.0);
        layer->z = NULL;
        layer->a = NULL;
    }
    return nn;
}

void nn_free(NeuralNetwork* nn) {
    if (nn) {
        for (int i = 0; i < nn->num_layers - 1; i++) {
            matrix_free(nn->layers[i].weights);
            matrix_free(nn->layers[i].biases);
            matrix_free(nn->layers[i].z);
            matrix_free(nn->layers[i].a);
        }
        free(nn->layers);
        free(nn->layer_sizes);
        free(nn);
    }
}

Matrix* forward_propagation(NeuralNetwork* nn, const Matrix* input) {
    Matrix* current_a = matrix_copy(input);
    for (int i = 0; i < nn->num_layers - 1; i++) {
        Layer* layer = &nn->layers[i];
        matrix_free(layer->z);
        matrix_free(layer->a);

        Matrix* z_unbiased = matrix_multiply(layer->weights, current_a);
        matrix_broadcast_add_column(z_unbiased, layer->biases);
        layer->z = z_unbiased;

        layer->a = matrix_copy(layer->z);
        matrix_apply_function(layer->a, layer->activation);

        matrix_free(current_a);
        current_a = matrix_copy(layer->a);
    }
    return current_a;
}

/**
 * @brief Performs backpropagation using batch gradient descent.
 *
 * This function is now structured in two main phases for stability:
 * 1. A backward pass to compute the gradients (dW, dB) for all layers.
 * 2. A separate loop to apply the updates to the network's weights and biases.
 */
void backpropagation(NeuralNetwork* nn, const Matrix* x, const Matrix* y, double learning_rate) {
    Matrix* output = nn->layers[nn->num_layers - 2].a;
    int batch_size = x->cols;
    int num_weight_layers = nn->num_layers - 1;

    Matrix** dW = (Matrix**)malloc(num_weight_layers * sizeof(Matrix*));
    Matrix** dB = (Matrix**)malloc(num_weight_layers * sizeof(Matrix*));

    Matrix* error = matrix_subtract(y, output);
    Layer* last_layer = &nn->layers[num_weight_layers - 1];
    Matrix* d_z_last = matrix_copy(last_layer->z);
    matrix_apply_function(d_z_last, last_layer->activation_derivative);
    Matrix* delta = matrix_elementwise_multiply(error, d_z_last);
    matrix_free(error);
    matrix_free(d_z_last);

    for (int i = num_weight_layers - 1; i >= 0; i--) {
        Matrix* prev_a = (i == 0) ? (Matrix*)x : nn->layers[i-1].a;
        Matrix* prev_a_t = matrix_transpose(prev_a);

        dW[i] = matrix_multiply(delta, prev_a_t);
        dB[i] = matrix_sum_columns(delta);
        matrix_free(prev_a_t);

        if (i > 0) {
            Matrix* weights_t = matrix_transpose(nn->layers[i].weights);
            Matrix* next_error = matrix_multiply(weights_t, delta);
            matrix_free(weights_t);

            Matrix* d_z_prev = matrix_copy(nn->layers[i-1].z);
            matrix_apply_function(d_z_prev, nn->layers[i-1].activation_derivative);

            Matrix* next_delta = matrix_elementwise_multiply(next_error, d_z_prev);
            matrix_free(next_error);
            matrix_free(d_z_prev);
            matrix_free(delta);
            delta = next_delta;
        }
    }
    matrix_free(delta);

    for (int i = 0; i < num_weight_layers; i++) {
        for (int j = 0; j < dW[i]->rows * dW[i]->cols; j++) {
            nn->layers[i].weights->data[j] += learning_rate * (dW[i]->data[j] / batch_size);
        }
        for (int j = 0; j < dB[i]->rows * dB[i]->cols; j++) {
            nn->layers[i].biases->data[j] += learning_rate * (dB[i]->data[j] / batch_size);
        }

        matrix_free(dW[i]);
        matrix_free(dB[i]);
    }
    free(dW);
    free(dB);
}

void nn_train(NeuralNetwork* nn, const Matrix* x_train, const Matrix* y_train, int epochs, double learning_rate, int print_interval) {
    printf("Training on a batch of %d samples for %d epochs...\n", x_train->cols, epochs);

    for (int e = 0; e < epochs; e++) {
        Matrix* output = forward_propagation(nn, x_train);
        backpropagation(nn, x_train, y_train, learning_rate);

        if (e == 0 || e == epochs - 1 || (print_interval > 0 && (e % print_interval) == 0)) {
            Matrix* error = matrix_subtract(y_train, output);
            double total_loss = 0;
            for(int k=0; k < error->rows * error->cols; k++) {
                total_loss += 0.5 * pow(error->data[k], 2);
            }
            printf("Epoch %5d/%d | Loss: %f\n", e, epochs, total_loss / x_train->cols);
            matrix_free(error);
        }

        matrix_free(output);
    }
}
