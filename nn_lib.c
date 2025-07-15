#include "nn_lib.h"
#include <time.h>
#include <float.h> // For DBL_MAX

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
            m->data[i * m->columns + j] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
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

Matrix* matrix_subtract(const Matrix* a, const Matrix* b) {
    if ((a->columns != b->columns) || (a->rows != b->rows)) {
        fprintf(stderr, "Error: Matrix dimensions not compatible for addition. Matrix A is %dx%d, Matrix B is %dx%d.\n", a->rows, a->columns, b->rows, b->columns);
        return NULL;
    }
    Matrix* result = matrix_create(a->rows, b->columns);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->columns; j++) {
            int index = i * a->columns + j;
            result->data[index] = a->data[index] - b->data[index];
        }
    }
    return result;
}

Matrix* matrix_transpose(const Matrix* m) {
    Matrix* result = matrix_create(m->columns, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->columns; j++) {
            result->data[j * result->columns + i] = m->data[i * m->columns + j];
        }
    }
    return result;
}

void matrix_map(Matrix* m, double (*func)(double)) {
    for (int i = 0; i < m->rows * m->columns; i++) {
        m->data[i] = func(m->data[i]);
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

void activate_sigmoid(Matrix* m) {
    matrix_map(m, sigmoid);
}

void derivative_sigmoid(Matrix* activated, Matrix* derivative) {
    for (int i = 0; i < activated->rows * activated-> columns; i++) {
        derivative->data[i] = sigmoid_derivative(activated->data[i]);
    }
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

void activate_relu(Matrix* activated) {
    matrix_map(activated, relu);
}

void derivative_relu(Matrix* activated, Matrix* derivative) {
    for (int i = 0; i < activated->rows * activated-> columns; i++) {
        derivative->data[i] = relu_derivative(activated->data[i]);
    }
}

void activate_softmax(Matrix* m) {
    double max_val = -DBL_MAX;
    for (int i = 0; i < m->rows * m->columns; i++) {
        if (m->data[i] > max_val) {
            max_val = m->data[i];
        }
    }

    double sum = 0;
    for (int i = 0; i < m->rows * m->columns; i++) {
        m->data[i] = exp(m->data[i] - max_val);
        sum += m->data[i];
    }

    for (int i = 0; i < m->rows * m->columns; i++) {
        m->data[i] = m->data[i] / sum;
    }
}

// Core network logic

Network* network_create(int* layer_sizes, int num_layers, ActivationType* activation_functions, double learning_rate) {
    srand(time(NULL));
    Network* network = malloc(sizeof(Network));
    network->num_layers = num_layers - 1; //input layer lacks weights/biases
    network->learning_rate = learning_rate;
    network->layers = malloc(sizeof(Layer*) * network->num_layers);

    for (int i = 0; i < network->num_layers; i++) {
        network->layers[i].weights = matrix_create(layer_sizes[i+1], layer_sizes[i]);
        network->layers[i].biases = matrix_create(layer_sizes[i+1], 1);
        matrix_randomize(network->layers[i].weights);
        matrix_randomize(network->layers[i].biases);

        switch (activation_functions[i]) {
            case SIGMOID:
                network->layers[i].activation = activate_sigmoid;
                network->layers[i].activation_derivative = derivative_relu;
                break;
            case RELU:
                network->layers[i].activation = activate_relu;
                network->layers[i].activation_derivative = derivative_relu;
                break;
            case SOFTMAX:
                network->layers[i].activation = activate_softmax;
                network->layers[i].activation_derivative = NULL;
                break;
        }
    }
    return network;
}

void network_free(Network* network) {
    if (network) {
        for (int i = 0; i < network->num_layers; i++) {
            matrix_free(network->layers[i].weights);
            matrix_free(network->layers[i].biases);
        }
        free(network->layers);
        free(network);
    }
}

Matrix* matrix_forward(Network* network, const Matrix* input) {
    Matrix* current_activation = matrix_create(input->rows, input->columns);
    for(int i=0; i < input->rows * input->columns; ++i) {
        current_activation->data[i] = input->data[i];
    }

    for (int i = 0; i < network->num_layers; i++) {
        Matrix* weighted_sum = matrix_multiply(network->layers[i].weights, current_activation);
        Matrix* with_biases = matrix_add(weighted_sum, network->layers[i].biases);

        network->layers[i].activation(with_biases);

        matrix_free(current_activation);
        matrix_free(weighted_sum);
        current_activation = with_biases;
    }
    return current_activation;
}

void network_backprop(Network* network, const Matrix* input, const Matrix* target) {
    Matrix** layer_outputs = malloc((network->num_layers + 1) * sizeof(Matrix*));
    layer_outputs[0] = matrix_create(input->rows, input->columns);
    for (int i = 0; i < input->rows * input->columns; ++i) {
        layer_outputs[0]->data[i] = input->data[i];
    }

    for (int i = 0; i < network->num_layers; i++) {
        Matrix* weighted_sum = matrix_multiply(network->layers[i].weights, layer_outputs[i]);
        Matrix* with_biases = matrix_add(weighted_sum, network->layers[i].biases);
        network->layers[i].activation(with_biases);
        matrix_free(weighted_sum);
    }

    //initial error calc
    Matrix* error = matrix_subtract(layer_outputs[network->num_layers], target);

    for (int i = network->num_layers; i >= 0; i--) {
        Matrix* gradient;

        //special case for Softmax (or any function where derivative is combined with loss)
        if (network->layers[i].activation_derivative == NULL) {
            gradient = matrix_create(error->rows, error->columns);
            for(int j=0; j < error->rows * error->columns; ++j) {
                gradient->data[j] = error->data[j];
            }
        } else {
            Matrix* output_derivative = matrix_create(layer_outputs[i+1]->rows, layer_outputs[i+1]->columns);
            network->layers[i].activation_derivative(layer_outputs[i+1], output_derivative);

            gradient = matrix_create(error->rows, error->columns);
            for(int j = 0; j < error->rows * error->columns; ++j) {
                gradient->data[j] = error->data[j] * output_derivative->data[j];
            }
            matrix_free(output_derivative);
        }

        Matrix* layer_output_transposed = matrix_transpose(layer_outputs[i]);
        Matrix* weight_deltas = matrix_multiply(gradient, layer_output_transposed);

        for(int j=0; j < network->layers[i].weights->rows * network->layers[i].weights->columns; ++j){
            network->layers[i].weights->data[j] -= weight_deltas->data[j] * network->learning_rate;
        }
        for(int j=0; j < network->layers[i].biases->rows * network->layers[i].biases->columns; ++j){
            network->layers[i].biases->data[j] -= gradient->data[j] * network->learning_rate;
        }

        Matrix* weights_transposed = matrix_transpose(network->layers[i].weights);
        Matrix* next_error = matrix_multiply(weights_transposed, error);

        matrix_free(error);
        error = next_error;

        matrix_free(gradient);
        matrix_free(layer_output_transposed);
        matrix_free(weight_deltas);
        matrix_free(weights_transposed);
    }

    matrix_free(error);
    for (int i = 0; i < network->num_layers; ++i) {
        matrix_free(network->layers[i].weights);
    }
    free(layer_outputs);

}

void network_train(Network* network, Matrix** inputs, Matrix** targets, int num_samples, int epochs) {
    for (int i = 0; i < epochs; i++) {
        double total_loss = 0;
        for (int j = 0; j < num_samples; j++) {
            Matrix* output = network_forward(network, inputs[j]);

            // Calculate Cross-Entropy Loss for Softmax
            for(int k=0; k < targets[j]->rows * targets[j]->columns; ++k) {
                if (targets[j]->data[k] == 1) { // One-hot encoding
                    total_loss -= log(output->data[k]);
                }
            }

            network_backprop(network, inputs[j], targets[j]);
            matrix_free(output);
        }
        if ((i + 1) % 1000 == 0) {
            printf("Epoch %d/%d, Loss: %f\n", i + 1, epochs, total_loss / num_samples);
        }
    }
}
