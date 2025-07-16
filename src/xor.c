#include <stdio.h>
#include <time.h>
#include "nn.h"
#include "matrix.h"

int main() {
    srand((unsigned int)time(NULL));


    // --- XOR Dataset (Column-Major Format) ---
    // Each COLUMN is a sample: [feature1, feature2]'
    // Features are in rows, samples are in columns.
    double x_data[] = { 0.0, 0.0, 1.0, 1.0,  // Feature 1
                        0.0, 1.0, 0.0, 1.0 };// Feature 2
    double y_data[] = { 0.0, 1.0, 1.0, 0.0 };

    // Matrix dimensions: {rows, cols} -> {features, samples}
    Matrix x_train_mat = {2, 4, x_data};
    Matrix y_train_mat = {1, 4, y_data};

    // --- Network Architecture ---
    int layer_sizes[] = {2, 4, 1};
    activation_func activations[] = {relu, sigmoid};
    activation_func activation_derivatives[] = {relu_derivative, sigmoid_derivative};
    NeuralNetwork* nn = nn_create(3, layer_sizes, activations, activation_derivatives);

    // --- Training ---
    // Use the unified training function. We process all 4 samples as one batch.
    // GPU is disabled, unnecessary for small operation
    matrix_set_gpu_mode(0);
    nn_train(nn, &x_train_mat, &y_train_mat, 15000, 0.1, 1000);

    printf("\n--- Final Predictions ---\n");
    // Test each of the 4 samples
    for (int i = 0; i < x_train_mat.cols; i++) {
        // Create a single-column input vector for prediction
        Matrix* input_sample = matrix_create(2, 1);
        input_sample->data[0] = x_train_mat.data[i];          // Feature 1 from column i
        input_sample->data[1] = x_train_mat.data[i + x_train_mat.cols]; // Feature 2 from column i

        Matrix* prediction = forward_propagation(nn, input_sample);

        printf("Input: [%.1f, %.1f] -> Output: [%.6f], Expected: [%.1f]\n",
               input_sample->data[0],
               input_sample->data[1],
               prediction->data[0],
               y_train_mat.data[i]);

        matrix_free(input_sample);
        matrix_free(prediction);
    }

    nn_free(nn);

    return 0;
}
