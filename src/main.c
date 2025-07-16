#include <stdio.h>
#include <time.h>
#include "nn.h"
#include "matrix.h" // Include for matrix_set_gpu_mode

void run_training_session(const char* mode_name, const Matrix* x_train, const Matrix* y_train) {
    printf("\n--- Starting Training Session: %s ---\n", mode_name);

    // --- Network Architecture ---
    int layer_sizes[] = {2, 4, 1};
    activation_func activations[] = {relu, sigmoid};
    activation_func activation_derivatives[] = {relu_derivative, sigmoid_derivative};
    NeuralNetwork* nn = nn_create(3, layer_sizes, activations, activation_derivatives);

    // --- Timing and Training ---
    clock_t start = clock();
    nn_train(nn, x_train, y_train, 10000, 0.1);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nTraining complete.\n");
    printf("Time elapsed for %s training: %f seconds\n", mode_name, time_spent);

    // --- Verification ---
    printf("Final prediction for [1.0, 0.0]:\n");
    Matrix* test_input = matrix_create(2, 1);
    test_input->data[0] = 1.0;
    test_input->data[1] = 0.0;
    Matrix* prediction = forward_propagation(nn, test_input);
    printf("Input: [1.0, 0.0] -> Output: [%.6f], Expected: [1.0]\n", prediction->data[0]);

    // --- Cleanup ---
    matrix_free(test_input);
    matrix_free(prediction);
    nn_free(nn);
}

int main() {
    srand((unsigned int)time(NULL));

    printf("====================================================\n");
    printf("=   Neural Network Performance Comparison (GPU vs CPU)   =\n");
    printf("====================================================\n");

    // --- XOR Dataset ---
    double x_data[] = { 0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  1.0, 1.0 };
    double y_data[] = { 0.0,        1.0,        1.0,        0.0 };
    Matrix x_train_mat = {4, 2, x_data};
    Matrix y_train_mat = {4, 1, y_data};

    // --- Run 1: GPU Enabled ---
    matrix_set_gpu_mode(1);
    run_training_session("GPU (with CPU fallback)", &x_train_mat, &y_train_mat);

    printf("\n----------------------------------------------------\n");

    // --- Run 2: GPU Disabled (Force CPU) ---
    matrix_set_gpu_mode(0);
    run_training_session("CPU-Only", &x_train_mat, &y_train_mat);

    printf("\n====================================================\n");
    printf("=                  Benchmark Complete                =\n");
    printf("====================================================\n");

    return 0;
}
