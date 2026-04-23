#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "true_labels.h"

#define INPUT_SIZE 10752
#define OUTPUT_SIZE 10

static int load_input(const char* path, float* buf) {
    FILE* f = fopen(path, "r");
    if (!f) {
        perror(path);
        return -1;
    }
    for (int i = 0; i < INPUT_SIZE; ++i) {
        if (fscanf(f, "%f", &buf[i]) != 1) {
            fprintf(stderr, "%s: expected %d floats, got %d\n", path, INPUT_SIZE, i);
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    return 0;
}

static int argmax(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

int main(void) {
    float input[INPUT_SIZE];
    float output[OUTPUT_SIZE];
    int correct = 0;

    for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
        char path[256];
        snprintf(path, sizeof(path), "inputs/test_input_%d.txt", i);
        if (load_input(path, input) != 0) {
            return 1;
        }
        model_forward(input, output);
        int pred = argmax(output, OUTPUT_SIZE);
        int truth = TEST_TRUE_LABELS[i];
        if (pred == truth) {
            correct++;
        }
        printf("sample %2d: predicted=%d true=%d %s\n", i, pred, truth,
               pred == truth ? "OK" : "MISS");
    }
    printf("\naccuracy: %d/%d (%.2f%%)\n", correct, NUM_TEST_SAMPLES,
           100.0 * (double)correct / (double)NUM_TEST_SAMPLES);
    return 0;
}
