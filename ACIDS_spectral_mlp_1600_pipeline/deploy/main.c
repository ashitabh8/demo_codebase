/*
 * ACIDS Spectral MLP int8 C inference runner.
 *
 * Input file format:
 *   raw float32 binary, 1600 samples @ 1600 Hz (ch0 continuous truncate).
 *
 * Run with no arguments to execute all packaged samples.
 * Run with one or more paths to execute explicit raw binary samples.
 *
 * Pipeline: 1600 audio -> Welch spectral features (83) -> StandardScaler -> W8 MLP.
 */
#include <stdio.h>
#include <stdlib.h>

#include "acids_class_names.h"
#include "acids_samples.h"
#include "acids_spectral_preprocess.h"
#include "model.h"
#include "spectral_scaler.h"

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

static int run_one_file(
    const char *path,
    const char *sample_name,
    int expected_id,
    const char *expected_name
)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "failed to open input: %s\n", path);
        return 1;
    }

    float audio[ACIDS_SPECTRAL_INPUT_SAMPLES];
    float raw[ACIDS_SPECTRAL_FEATURE_DIM];
    float scaled[ACIDS_SPECTRAL_FEATURE_DIM];
    float logits[ACIDS_NUM_CLASSES];

    size_t nread = fread(audio, sizeof(float), ACIDS_SPECTRAL_INPUT_SAMPLES, f);
    fclose(f);
    if (nread != (size_t)ACIDS_SPECTRAL_INPUT_SAMPLES) {
        fprintf(
            stderr,
            "expected %d floats, read %zu\n",
            ACIDS_SPECTRAL_INPUT_SAMPLES,
            nread
        );
        return 1;
    }

    if (acids_spectral_preprocess(audio, raw) != 0) {
        fprintf(stderr, "spectral preprocessing failed\n");
        return 1;
    }

    spectral_apply_standard_scaler(raw, scaled, SPECTRAL_FEATURE_DIM);
    model_forward(scaled, logits);

    int pred = argmax(logits, ACIDS_NUM_CLASSES);
    if (sample_name != NULL) {
        printf("sample=%s\n", sample_name);
    }
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
        for (int i = 0; i < ACIDS_NUM_PACKAGED_SAMPLES; ++i) {
            int rc = run_one_file(
                ACIDS_SAMPLE_PATHS[i],
                ACIDS_SAMPLE_NAMES[i],
                ACIDS_SAMPLE_LABEL_IDS[i],
                ACIDS_SAMPLE_LABEL_NAMES[i]
            );
            if (rc != 0) {
                failures += 1;
            }
        }
        return failures == 0 ? 0 : 1;
    }

    for (int i = 1; i < argc; ++i) {
        if (run_one_file(argv[i], NULL, -1, "") != 0) {
            failures += 1;
        }
    }
    return failures == 0 ? 0 : 1;
}
