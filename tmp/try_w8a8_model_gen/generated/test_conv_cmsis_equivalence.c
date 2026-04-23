#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nn_ops_int8.h"
#include "nn_ops_int8_cmsis.h"

#undef conv2d_nhwc_int8
#undef conv2d_nhwc_int8_per_channel
#undef depthwise_conv2d_nhwc_int8
#undef depthwise_conv2d_nhwc_int8_to_float
#undef conv2d_nhwc_float_input_int8_weight_per_channel

static uint32_t lcg_state = 987654321u;

static uint32_t lcg_next(void) {
    lcg_state = (1664525u * lcg_state + 1013904223u);
    return lcg_state;
}

static int8_t rand_i8(void) {
    return (int8_t)((int)(lcg_next() % 255u) - 127);
}

static float rand_f32(float scale) {
    int32_t v = (int32_t)(lcg_next() % 20001u) - 10000;
    return ((float)v / 10000.0f) * scale;
}

static int compare_i8(const int8_t* a, const int8_t* b, int n, const char* tag) {
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            printf("[FAIL] %s mismatch at %d: %d vs %d\n", tag, i, (int)a[i], (int)b[i]);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    {
        const int in_h = 5, in_w = 6, in_c = 12;
        const int k_h = 3, k_w = 3, out_c = 10;
        const int stride_h = 1, stride_w = 2, pad_h = 1, pad_w = 1;
        const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

        int8_t in[in_h * in_w * in_c];
        int8_t filt[k_h * k_w * in_c * out_c];
        float bias[out_c];
        int8_t out_ref[out_h * out_w * out_c];
        int8_t out_cmsis[out_h * out_w * out_c];

        for (int i = 0; i < in_h * in_w * in_c; ++i) in[i] = rand_i8();
        for (int i = 0; i < k_h * k_w * in_c * out_c; ++i) filt[i] = rand_i8();
        for (int i = 0; i < out_c; ++i) bias[i] = rand_f32(0.2f);

        conv2d_nhwc_int8(
            in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.02f, 0.03f, 0.05f, 0, out_ref);
        conv2d_nhwc_int8_cmsis(
            in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.02f, 0.03f, 0.05f, 0, out_cmsis);

        if (compare_i8(out_ref, out_cmsis, out_h * out_w * out_c, "conv2d_int8")) {
            return 1;
        }
    }

    {
        const int in_h = 6, in_w = 6, channels = 15;
        const int k_h = 3, k_w = 3;
        const int stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
        const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

        int8_t in[in_h * in_w * channels];
        int8_t filt[k_h * k_w * channels];
        float bias[channels];
        float w_scales[channels];
        int8_t out_ref[out_h * out_w * channels];
        int8_t out_cmsis[out_h * out_w * channels];

        for (int i = 0; i < in_h * in_w * channels; ++i) in[i] = rand_i8();
        for (int i = 0; i < k_h * k_w * channels; ++i) filt[i] = rand_i8();
        for (int i = 0; i < channels; ++i) {
            bias[i] = rand_f32(0.15f);
            w_scales[i] = 0.003f + (float)(i + 1) * 0.0004f;
        }

        depthwise_conv2d_nhwc_int8(
            in, in_h, in_w, channels, filt, k_h, k_w, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.03f, w_scales, 0.07f, 0, out_ref);
        depthwise_conv2d_nhwc_int8_cmsis(
            in, in_h, in_w, channels, filt, k_h, k_w, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.03f, w_scales, 0.07f, 0, out_cmsis);

        if (compare_i8(out_ref, out_cmsis, out_h * out_w * channels, "depthwise_int8")) {
            return 1;
        }
    }

    printf("[PASS] CMSIS conv wrappers match naive conv implementations.\n");
#if !NN_OPS_CMSIS_AVAILABLE
    printf("[INFO] CMSIS-NN not enabled/available in this build; fallback parity verified.\n");
#endif
    return 0;
}
