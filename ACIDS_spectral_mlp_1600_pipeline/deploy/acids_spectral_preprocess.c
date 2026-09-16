#include "acids_spectral_preprocess.h"

#include <stddef.h>

#include "kissfft/kiss_fftr.h"

#define ACIDS_SPECTRAL_FFTR_MEM_BYTES 1888u
#define ACIDS_SPECTRAL_FREQ_HZ_PER_BIN \
    ((float)ACIDS_SPECTRAL_SAMPLE_RATE_HZ / (float)ACIDS_SPECTRAL_N_FFT)

static kiss_fftr_cfg spectral_fftr_cfg(void)
{
    static unsigned char mem[ACIDS_SPECTRAL_FFTR_MEM_BYTES];
    static kiss_fftr_cfg cfg = NULL;

    if (cfg == NULL) {
        size_t len = ACIDS_SPECTRAL_FFTR_MEM_BYTES;
        cfg = kiss_fftr_alloc(ACIDS_SPECTRAL_N_FFT, 0, mem, &len);
    }
    return cfg;
}

static void spectral_rfft_power(const float *time_in, double *power_out)
{
    const int n_bins = ACIDS_SPECTRAL_N_BINS;
    kiss_fft_cpx freq[ACIDS_SPECTRAL_N_BINS];
    kiss_fft_scalar timedata[ACIDS_SPECTRAL_N_FFT];
    int t;

    for (t = 0; t < ACIDS_SPECTRAL_N_FFT; ++t) {
        timedata[t] = (kiss_fft_scalar)time_in[t];
    }

    kiss_fftr(spectral_fftr_cfg(), timedata, freq);

    for (t = 0; t < n_bins; ++t) {
        const double real = (double)freq[t].r;
        const double imag = (double)freq[t].i;
        power_out[t] = real * real + imag * imag;
    }
}

int acids_spectral_preprocess(const float *input_1600, float *out_features_83)
{
    float time_buf[ACIDS_SPECTRAL_N_FFT];
    double frame_power[ACIDS_SPECTRAL_N_BINS];
    double avg_power[ACIDS_SPECTRAL_N_BINS];
    double power_sum_raw = 0.0;
    double weighted_freq_sum = 0.0;
    int k;
    int start;
    int n_frames = 0;
    const int last_start = ACIDS_SPECTRAL_INPUT_SAMPLES - ACIDS_SPECTRAL_N_FFT;

    if (input_1600 == NULL || out_features_83 == NULL) {
        return -1;
    }

    if (spectral_fftr_cfg() == NULL) {
        return -1;
    }

    for (k = 0; k < ACIDS_SPECTRAL_N_BINS; ++k) {
        avg_power[k] = 0.0;
    }

    /* Welch: average periodogram over hop windows; always include final aligned frame. */
    for (start = 0; start <= last_start; start += ACIDS_SPECTRAL_HOP) {
        for (k = 0; k < ACIDS_SPECTRAL_N_FFT; ++k) {
            time_buf[k] = input_1600[start + k];
        }
        spectral_rfft_power(time_buf, frame_power);
        for (k = 0; k < ACIDS_SPECTRAL_N_BINS; ++k) {
            avg_power[k] += frame_power[k];
        }
        n_frames += 1;
    }
    if (n_frames > 0) {
        const int last_aligned = (n_frames - 1) * ACIDS_SPECTRAL_HOP;
        if (last_aligned != last_start) {
            for (k = 0; k < ACIDS_SPECTRAL_N_FFT; ++k) {
                time_buf[k] = input_1600[last_start + k];
            }
            spectral_rfft_power(time_buf, frame_power);
            for (k = 0; k < ACIDS_SPECTRAL_N_BINS; ++k) {
                avg_power[k] += frame_power[k];
            }
            n_frames += 1;
        }
    }

    if (n_frames <= 0) {
        return -1;
    }

    for (k = 0; k < ACIDS_SPECTRAL_N_BINS; ++k) {
        avg_power[k] /= (double)n_frames;
    }

    power_sum_raw = 0.0;
    weighted_freq_sum = 0.0;
    for (k = 0; k < ACIDS_SPECTRAL_N_BINS; ++k) {
        power_sum_raw += avg_power[k];
        weighted_freq_sum += avg_power[k] * (double)((float)k * ACIDS_SPECTRAL_FREQ_HZ_PER_BIN);
    }

    {
        const double power_sum = power_sum_raw + (double)ACIDS_SPECTRAL_EPS;
        out_features_83[0] = (float)(weighted_freq_sum / power_sum);
        out_features_83[1] = (float)(power_sum_raw / (double)ACIDS_SPECTRAL_N_BINS);

        for (k = 0; k < ACIDS_SPECTRAL_N_BINS; ++k) {
            out_features_83[2 + k] = (float)(avg_power[k] / power_sum);
        }
    }

    return 0;
}
