#include <stdio.h>
#include <time.h>
#include "nn.h"

int main() {
    // Cast to unsigned int to prevent a common compiler warning on Windows.
    srand((unsigned int)time(NULL));

    printf("--- C Neural Network (CPU-Only) XOR Test ---\n");
    printf("--- Architecture: Input(2) -> ReLU(4) -> Sigmoid(1) ---\n\n");

    // --- XOR Dataset ---
    double x_data[] = { 0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  1.0, 1.0 };
    double y_data[] = { 0.0,        1.0,        1.0,        0.0 };
    Matrix x_train_mat = {4, 2, x_data};
    Matrix y_train_mat = {4, 1, y_data};

    // --- Network Architecture ---
    int layer_sizes[] = {2, 4, 1};
    int num_layers = 3;

    // --- Specify Activation Functions Per Layer ---
    // There are (num_layers - 1) layers with activations.
    // Layer 1 (Hidden): ReLU
    // Layer 2 (Output): Sigmoid (good for binary classification output between 0 and 1)
    activation_func activations[] = {relu, sigmoid};
    activation_func activation_derivatives[] = {relu_derivative, sigmoid_derivative};

    NeuralNetwork* nn = nn_create(num_layers, layer_sizes, activations, activation_derivatives);

    nn_train(nn, &x_train_mat, &y_train_mat, 10000, 0.1);
    printf("\nTraining complete.\n\n");

    printf("--- Final Predictions ---\n");
    for (int i = 0; i < x_train_mat.rows; i++) {
        Matrix* input_sample = matrix_create(2, 1);
        input_sample->data[0] = x_train_mat.data[i * 2 + 0];
        input_sample->data[1] = x_train_mat.data[i * 2 + 1];

        Matrix* prediction = forward_propagation(nn, input_sample);

        printf("Input: [%.1f, %.1f] -> Output: [%.6f], Expected: [%.1f]\n",
               input_sample->data[0],
               input_sample->data[1],
               prediction->data[0],
               y_train_mat.data[i]);

        matrix_free(input_sample);
        matrix_free(prediction);
    }

    // --- Cleanup ---
    nn_free(nn);

    return 0;
}
