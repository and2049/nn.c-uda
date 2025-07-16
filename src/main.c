#include <stdio.h>
#include <time.h>
#include "nn.h"
#include "matrix.h"

void run_training_session(const char* mode_name, int epochs, NeuralNetwork* nn, const Matrix* x_train, const Matrix* y_train) {
    printf("\n--- Starting Training Session: %s ---\n", mode_name);

    clock_t start = clock();
    nn_train(nn, x_train, y_train, epochs, 0.01, 10);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nTraining complete.\n");
    printf("Time elapsed for %s training: %f seconds\n", mode_name, time_spent);
}

int main() {
    srand((unsigned int)time(NULL));

    int input_size = 512;
    int hidden_size = 256;
    int output_size = 64;
    int batch_size = 2048; // Number of training samples
    int epochs = 50;

    printf("Benchmark Config: Arch=[%d, %d, %d], Batch Size=%d, Epochs=%d\n",
           input_size, hidden_size, output_size, batch_size, epochs);

    Matrix* x_train = matrix_create(input_size, batch_size);
    Matrix* y_train = matrix_create(output_size, batch_size);
    matrix_fill_random(x_train, 1.0);
    matrix_fill_random(y_train, 1.0);

    int layer_sizes[] = {input_size, hidden_size, output_size};
    activation_func activations[] = {relu, tanh_activation};
    activation_func activation_derivatives[] = {relu_derivative, tanh_derivative};

    NeuralNetwork* nn = nn_create(3, layer_sizes, activations, activation_derivatives);

    NeuralNetwork* nn_copy = nn_create(3, layer_sizes, activations, activation_derivatives);
    for(int i=0; i < nn->num_layers - 1; ++i) {
        matrix_free(nn_copy->layers[i].weights);
        matrix_free(nn_copy->layers[i].biases);
        nn_copy->layers[i].weights = matrix_copy(nn->layers[i].weights);
        nn_copy->layers[i].biases  = matrix_copy(nn->layers[i].biases);
    }


    matrix_set_gpu_mode(1);
    run_training_session("GPU", epochs, nn, x_train, y_train);

    printf("\n----------------------------------------------------\n");

    matrix_set_gpu_mode(0);
    run_training_session("CPU-Only", epochs, nn_copy, x_train, y_train);

    printf("\n----------------------------------------------------\n");

    matrix_free(x_train);
    matrix_free(y_train);
    nn_free(nn);
    nn_free(nn_copy);

    return 0;
}
