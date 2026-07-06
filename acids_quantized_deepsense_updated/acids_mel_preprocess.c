#include "acids_mel_preprocess.h"

#include <math.h>
#include <stddef.h>

#include "acids_mel_tables.h"
#include "acids_rfft_tables.h"

static void acids_rfft(const float *time_in, float *power_out)
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

static void acids_segment_mel_len(const float *segment, int segment_len, float *mel_out)
{
    float time_buf[ACIDS_N_FFT];
    float power[ACIDS_N_FFT_BINS];

    for (int t = 0; t < ACIDS_N_FFT; ++t) {
        if (t < segment_len) {
            time_buf[t] = segment[t];
        } else {
            time_buf[t] = 0.0f;
        }
    }

    acids_rfft(time_buf, power);

    for (int m = 0; m < ACIDS_MEL_BINS; ++m) {
        float mel_val = 0.0f;
        for (int k = 0; k < ACIDS_N_FFT_BINS; ++k) {
            mel_val += power[k] * ACIDS_MEL_FILTERBANK[m][k];
        }
        mel_out[m] = logf(mel_val + ACIDS_MEL_LOG_EPS);
    }
}

static void acids_segment_mel(const float *segment, float *mel_out)
{
    acids_segment_mel_len(segment, ACIDS_N_FFT, mel_out);
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

#if 0
/* Legacy 16 kHz -> 1600 Hz decimator (no longer used; input is already @ 1600 Hz). */
static float acids_conv1d_same_at(const float *segment, int trimmed_len, int out_idx)
{
    float acc = 0.0f;
    for (int tap = 0; tap < ACIDS_DECIMATE_TAPS; ++tap) {
        const int in_idx = out_idx + tap - ACIDS_DECIMATE_PAD;
        if (in_idx >= 0 && in_idx < trimmed_len) {
            acc += segment[in_idx] * ACIDS_DECIMATE_FIR[tap];
        }
    }
    return acc;
}

static void acids_decimate_segment_256(const float *segment, float *out_decimated)
{
    const int trimmed_len =
        ACIDS_LEGACY_RAW_SEG_SAMPLES - (ACIDS_LEGACY_RAW_SEG_SAMPLES % ACIDS_DECIMATE_FACTOR);

    for (int i = 0; i < ACIDS_DECIMATED_SEG_SAMPLES; ++i) {
        const int filtered_idx = i * ACIDS_DECIMATE_FACTOR;
        out_decimated[i] = acids_conv1d_same_at(segment, trimmed_len, filtered_idx);
    }
}
#endif

int acids_audio_preprocess_ch0_segments_chw(const float *input_chw, float *out_chw)
{
    if (input_chw == NULL || out_chw == NULL) {
        return -1;
    }

    for (int seg = 0; seg < ACIDS_NUM_SEGMENTS; ++seg) {
        const float *raw_segment = input_chw + seg * ACIDS_RAW_SEG_SAMPLES;
        float *mel_row = out_chw + seg * ACIDS_MEL_BINS;
        acids_segment_mel_len(raw_segment, ACIDS_RAW_SEG_SAMPLES, mel_row);
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
