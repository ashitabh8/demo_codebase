// Minimal int8 ops for spectral MLP deploy.
#ifndef NN_OPS_INT8_H
#define NN_OPS_INT8_H

#include <stdint.h>

static inline void dense_float_input_int8_weight_per_channel(
    const float *x,
    int in_features,
    const int8_t *W,
    const float *b,
    int out_features,
    const float *weight_scales,
    float *y)
{
    for (int o = 0; o < out_features; ++o) {
        float acc = (b != NULL) ? b[o] : 0.0f;
        float scale_o = weight_scales[o];
        const int8_t *w_col = W + o;
        for (int i = 0; i < in_features; ++i) {
            acc += x[i] * ((float)w_col[i * out_features] * scale_o);
        }
        y[o] = acc;
    }
}

#endif /* NN_OPS_INT8_H */
