#include "acids_mel_preprocess.h"

#include <math.h>
#include <stddef.h>

#include "acids_mel_tables.h"
#include "acids_rfft_tables.h"

static void acids_rfft_160(const float *time_in, float *power_out)
{
    const int n = ACIDS_N_FFT;
    const int n_bins = ACIDS_N_FFT_BINS;

    for (int k = 0; k < n_bins; ++k) {
        float real = 0.0f;
        float imag = 0.0f;
        for (int t = 0; t < n; ++t) {
            const float x = time_in[t];
            real += x * ACIDS_RFFT_REAL[k][t];
            imag += x * ACIDS_RFFT_IMAG[k][t];
        }
        power_out[k] = real * real + imag * imag;
    }
}

static void acids_segment_mel(const float *segment, float *mel_out)
{
    float time_buf[ACIDS_N_FFT];
    float power[ACIDS_N_FFT_BINS];

    for (int t = 0; t < ACIDS_N_FFT; ++t) {
        time_buf[t] = segment[t];
    }

    acids_rfft_160(time_buf, power);

    for (int m = 0; m < ACIDS_MEL_BINS; ++m) {
        float mel_val = 0.0f;
        for (int k = 0; k < ACIDS_N_FFT_BINS; ++k) {
            mel_val += power[k] * ACIDS_MEL_FILTERBANK[m][k];
        }
        mel_out[m] = logf(mel_val + ACIDS_MEL_LOG_EPS);
    }
}

int acids_mel_preprocess_chw(const float *input_1600, float *out_chw)
{
    if (input_1600 == NULL || out_chw == NULL) {
        return -1;
    }

    for (int seg = 0; seg < ACIDS_NUM_SEGMENTS; ++seg) {
        const float *segment = input_1600 + seg * ACIDS_SEG_SAMPLES;
        float *mel_row = out_chw + seg * ACIDS_MEL_BINS;
        acids_segment_mel(segment, mel_row);
    }

    return 0;
}

int acids_mel_preprocess_hwc(const float *input_1600, float *out_hwc)
{
    float chw[ACIDS_OUTPUT_CHW_SIZE];

    if (input_1600 == NULL || out_hwc == NULL) {
        return -1;
    }

    if (acids_mel_preprocess_chw(input_1600, chw) != 0) {
        return -1;
    }

    for (int seg = 0; seg < ACIDS_NUM_SEGMENTS; ++seg) {
        for (int mel = 0; mel < ACIDS_MEL_BINS; ++mel) {
            const int chw_idx = seg * ACIDS_MEL_BINS + mel;
            const int hwc_idx = seg * ACIDS_MEL_BINS + mel;
            out_hwc[hwc_idx] = chw[chw_idx];
        }
    }

    return 0;
}
