#ifndef ACIDS_SPECTRAL_PREPROCESS_H
#define ACIDS_SPECTRAL_PREPROCESS_H

#define ACIDS_SPECTRAL_INPUT_SAMPLES  1600
#define ACIDS_SPECTRAL_N_FFT          160
#define ACIDS_SPECTRAL_N_BINS         81
#define ACIDS_SPECTRAL_FEATURE_DIM    83
#define ACIDS_SPECTRAL_SAMPLE_RATE_HZ 1600
#define ACIDS_SPECTRAL_HOP            80
#define ACIDS_SPECTRAL_EPS            1e-12f

/**
 * Welch spectral feature extraction for ch0 continuous path (no mel).
 *
 * Input:  exactly 1600 float32 samples @ 1600 Hz.
 *         Overlapping RFFT windows (n_fft=160, hop=80) cover the full stream;
 *         frame power spectra are averaged, then centroid / mean energy / PSD.
 * Output: 83 unscaled features [centroid, mean_energy, psd[0..80]].
 */
int acids_spectral_preprocess(const float *input_1600, float *out_features_83);

#endif /* ACIDS_SPECTRAL_PREPROCESS_H */
