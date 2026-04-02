#include <stdio.h>

#include "model.h"
#include "test_inputs.h"

int main(void) {
    float output[TEST_OUTPUT_SIZE];

    // Output contract: one line per sample, format:
    // sample_<idx> <logit_0> <logit_1> ... <logit_N-1>
    for (int sample_idx = 0; sample_idx < NUM_TEST_SAMPLES; ++sample_idx) {
        model_forward(TEST_INPUTS[sample_idx], output);
        printf("sample_%d", sample_idx);
        for (int j = 0; j < TEST_OUTPUT_SIZE; ++j) {
            printf(" %.9g", output[j]);
        }
        printf("\n");
    }

    return 0;
}
