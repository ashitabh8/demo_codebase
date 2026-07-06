#ifndef ACIDS_MEL_PREPROCESS_H
#define ACIDS_MEL_PREPROCESS_H

#include <stddef.h>

#define ACIDS_INPUT_SAMPLES   1600
#define ACIDS_NUM_SEGMENTS    7
#define ACIDS_SEG_SAMPLES     228
#define ACIDS_USED_SAMPLES    (ACIDS_NUM_SEGMENTS * ACIDS_SEG_SAMPLES)
#define ACIDS_RAW_AUDIO_CHANNELS 3
#define ACIDS_RAW_SAMPLE_RATE_HZ 1600
#define ACIDS_RAW_SEG_SAMPLES   25
#define ACIDS_MEL_BINS        80
#define ACIDS_N_FFT           25
#define ACIDS_N_FFT_BINS      (ACIDS_N_FFT / 2 + 1)
#define ACIDS_OUTPUT_CHW_SIZE (1 * ACIDS_NUM_SEGMENTS * ACIDS_MEL_BINS)

#define ACIDS_MEL_LOG_EPS     1e-6f

/**
 * Log-mel preprocessing for ACIDS 1-ch inference.
 *
 * Input: 1600 float32 samples @ 1600 Hz (1 s, mic ch0, already decimated).
 * The last (ACIDS_INPUT_SAMPLES - ACIDS_USED_SAMPLES) samples are ignored.
 *
 * Output layouts:
 *   CHW: [C=1][H=7 segments][W=80 mel] — PyTorch NCHW / model input.
 *   HWC: [H=7][W=80][C=1] — embedded channel-last tiles.
 */
int acids_mel_preprocess_chw(const float *input_1600, float *out_chw);
int acids_mel_preprocess_hwc(const float *input_1600, float *out_hwc);

/**
 * Dataset-shaped ACIDS audio preprocessing.
 *
 * Input: [C=3][segments=7][T=25] flattened in CHW order @ 1600 Hz per segment.
 * Processing: select channel 0, apply log-mel per segment (n_fft=25, no padding).
 *
 * Output CHW: [C=1][H=7][W=80] flattened.
 */
int acids_audio_preprocess_ch0_segments_chw(const float *input_chw, float *out_chw);

#endif /* ACIDS_MEL_PREPROCESS_H */
