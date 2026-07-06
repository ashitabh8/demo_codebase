/*
 * ACIDS DeepSense C inference runner.
 *
 * Input file format:
 *   raw float32 binary, flattened [C=3][segments=7][T=25] @ 1600 Hz per segment.
 *   Channel 0 is used for inference (channels 1-2 are ignored).
 *
 * Run with no arguments to execute all packaged samples.
 * Run with one or more paths to execute explicit raw binary samples.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "acids_class_names.h"
#include "acids_mel_preprocess.h"
#include "acids_samples.h"
#include "model.h"

#define RAW_INPUT_SIZE (ACIDS_RAW_AUDIO_CHANNELS * ACIDS_NUM_SEGMENTS * ACIDS_RAW_SEG_SAMPLES)
#define MODEL_INPUT_SIZE ACIDS_OUTPUT_CHW_SIZE

static int argmax(const float *values, int n)
{
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

static int run_one_file(const char *path, const char *sample_name, int expected_id, const char *expected_name)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "failed to open input: %s\n", path);
        return 1;
    }

    float raw_input[RAW_INPUT_SIZE];
    float model_input[MODEL_INPUT_SIZE];
    float logits[ACIDS_NUM_CLASSES];

    size_t nread = fread(raw_input, sizeof(float), RAW_INPUT_SIZE, f);
    fclose(f);
    if (nread != RAW_INPUT_SIZE) {
        fprintf(stderr, "expected %d floats, read %zu\n", RAW_INPUT_SIZE, nread);
        return 1;
    }

    if (acids_audio_preprocess_ch0_segments_chw(raw_input, model_input) != 0) {
        fprintf(stderr, "preprocessing failed\n");
        return 1;
    }

    model_forward(model_input, logits);

    int pred = argmax(logits, ACIDS_NUM_CLASSES);
    printf("sample=%s\n", sample_name);
    printf("input=%s\n", path);
    if (expected_id >= 0) {
        printf("expected_id=%d\n", expected_id);
        printf("expected_name=%s\n", expected_name);
    }
    printf("pred_id=%d\n", pred);
    printf("pred_name=%s\n", ACIDS_CLASS_NAMES[pred]);
    printf("logits=");
    for (int i = 0; i < ACIDS_NUM_CLASSES; ++i) {
        printf("%s%.9g", i == 0 ? "" : ",", logits[i]);
    }
    printf("\n");
    if (expected_id >= 0) {
        printf("match=%s\n", pred == expected_id ? "yes" : "no");
    }
    printf("\n");
    return 0;
}

int main(int argc, char **argv)
{
    int failures = 0;
    if (argc == 1) {
        int matches = 0;
        for (int i = 0; i < ACIDS_NUM_PACKAGED_SAMPLES; ++i) {
            int rc = run_one_file(
                ACIDS_SAMPLE_PATHS[i],
                ACIDS_SAMPLE_NAMES[i],
                ACIDS_SAMPLE_LABEL_IDS[i],
                ACIDS_SAMPLE_LABEL_NAMES[i]
            );
            if (rc != 0) {
                failures += 1;
                continue;
            }
            /* Recompute pred cheaply is not worth duplicating; match line is for humans. */
        }
        (void)matches;
        return failures == 0 ? 0 : 1;
    }

    for (int i = 1; i < argc; ++i) {
        int rc = run_one_file(argv[i], argv[i], -1, "");
        if (rc != 0) {
            failures += 1;
        }
    }
    return failures == 0 ? 0 : 1;
}
