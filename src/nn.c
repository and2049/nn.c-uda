#include "nn.h"
#include <math.h> // Corrected from <math.hh>
#include <assert.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double z) {
    double s = sigmoid(z);
    return s * (1.0 - s);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double z) {
    return z > 0 ? 1.0 : 0.0;
}

double tanh_activation(double x) {
    return tanh(x);
}

double tanh_derivative(double z) {
    double t = tanh(z);
    return 1.0 - t * t;
}


NeuralNetwork* nn_create(int num_layers, int* layer_sizes, activation_func* activations, activation_func* activation_derivatives) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for(int i = 0; i < num_layers; i++) {
        nn->layer_sizes[i] = layer_sizes[i];
    }

    nn->layers = (Layer*)malloc((num_layers - 1) * sizeof(Layer));

    for (int i = 0; i < num_layers - 1; i++) {
        Layer* layer = &nn->layers[i];
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i+1];

        layer->weights = matrix_create(output_size, input_size);
        layer->biases = matrix_create(output_size, 1);

        // Assign the specified activation functions for this layer
        layer->activation = activations[i];
        layer->activation_derivative = activation_derivatives[i];

        // Initialize weights with small random values
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
        layer->z = matrix_add(z_unbiased, layer->biases);
        matrix_free(z_unbiased);

        layer->a = matrix_copy(layer->z);
        matrix_apply_function(layer->a, layer->activation);

        matrix_free(current_a);
        current_a = matrix_copy(layer->a);
    }
    return current_a;
}

void backpropagation(NeuralNetwork* nn, const Matrix* x, const Matrix* y, double learning_rate) {
    Matrix* output = nn->layers[nn->num_layers - 2].a;

    Matrix* error = matrix_subtract(y, output);

    Layer* last_layer = &nn->layers[nn->num_layers - 2];
    Matrix* d_z_last = matrix_copy(last_layer->z);
    matrix_apply_function(d_z_last, last_layer->activation_derivative);

    Matrix* delta = matrix_elementwise_multiply(error, d_z_last);
    matrix_free(error);
    matrix_free(d_z_last);

    Matrix** dW = (Matrix**)malloc((nn->num_layers - 1) * sizeof(Matrix*));
    Matrix** dB = (Matrix**)malloc((nn->num_layers - 1) * sizeof(Matrix*));

    for (int i = nn->num_layers - 2; i >= 0; i--) {
        Matrix* prev_a = (i == 0) ? (Matrix*)x : nn->layers[i-1].a;
        Matrix* prev_a_t = matrix_transpose(prev_a);

        dW[i] = matrix_multiply(delta, prev_a_t);
        dB[i] = matrix_copy(delta);
        matrix_free(prev_a_t);

        if (i > 0) {
            Layer* current_layer = &nn->layers[i];
            Layer* prev_layer = &nn->layers[i-1];

            Matrix* weights_t = matrix_transpose(current_layer->weights);
            Matrix* next_error = matrix_multiply(weights_t, delta);
            matrix_free(weights_t);

            Matrix* d_z_prev = matrix_copy(prev_layer->z);
            matrix_apply_function(d_z_prev, prev_layer->activation_derivative);

            Matrix* next_delta = matrix_elementwise_multiply(next_error, d_z_prev);
            matrix_free(next_error);
            matrix_free(d_z_prev);

            matrix_free(delta);
            delta = next_delta;
        }
    }
    matrix_free(delta);

    for(int i = 0; i < nn->num_layers - 1; i++) {
        for(int j = 0; j < nn->layers[i].weights->rows * nn->layers[i].weights->cols; j++) {
            nn->layers[i].weights->data[j] += learning_rate * dW[i]->data[j];
        }
        for(int j = 0; j < nn->layers[i].biases->rows * nn->layers[i].biases->cols; j++) {
            nn->layers[i].biases->data[j] += learning_rate * dB[i]->data[j];
        }
        matrix_free(dW[i]);
        matrix_free(dB[i]);
    }
    free(dW);
    free(dB);
}

void nn_train(NeuralNetwork* nn, const Matrix* x_train, const Matrix* y_train, int epochs, double learning_rate) {
    printf("Training for %d epochs with a learning rate of %.2f...\n", epochs, learning_rate);

    for (int e = 0; e < epochs; e++) {
        double total_loss = 0;
        for (int i = 0; i < x_train->rows; i++) {
            Matrix* x_sample = matrix_create(nn->layer_sizes[0], 1);
            Matrix* y_sample = matrix_create(nn->layer_sizes[nn->num_layers-1], 1);

            for (int j = 0; j < x_sample->rows; j++) {
                x_sample->data[j] = x_train->data[i * x_train->cols + j];
            }
            for (int j = 0; j < y_sample->rows; j++) {
                y_sample->data[j] = y_train->data[i * y_train->cols + j];
            }

            Matrix* output = forward_propagation(nn, x_sample);

            backpropagation(nn, x_sample, y_sample, learning_rate);

            Matrix* error = matrix_subtract(y_sample, output);
            for(int k=0; k < error->rows * error->cols; k++) {
                total_loss += 0.5 * pow(error->data[k], 2);
            }

            matrix_free(x_sample);
            matrix_free(y_sample);
            matrix_free(output);
            matrix_free(error);
        }

        if ((e % 1000) == 0 || e == epochs - 1) {
            printf("Epoch %5d/%d | Loss: %f\n", e, epochs, total_loss / x_train->rows);
        }
    }
}
