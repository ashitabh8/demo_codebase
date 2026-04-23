#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nn_ops_int8.h"
#include "nn_ops_int8_arm.h"

#undef conv2d_nhwc_int8
#undef conv2d_nhwc_int8_per_channel
#undef depthwise_conv2d_nhwc_int8
#undef depthwise_conv2d_nhwc_int8_to_float
#undef conv2d_nhwc_float_input_int8_weight_per_channel

static uint32_t lcg_state = 123456789u;

static uint32_t lcg_next(void) {
    lcg_state = (1103515245u * lcg_state + 12345u);
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

static int compare_f32(const float* a, const float* b, int n, float tol, const char* tag) {
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tol) {
            printf("[FAIL] %s mismatch at %d: %.9g vs %.9g (diff=%.9g)\n",
                   tag, i, a[i], b[i], diff);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    {
        const int in_h = 5, in_w = 6, in_c = 13;
        const int k_h = 3, k_w = 3, out_c = 9;
        const int stride_h = 1, stride_w = 2, pad_h = 1, pad_w = 1;
        const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

        int8_t in[in_h * in_w * in_c];
        int8_t filt[k_h * k_w * in_c * out_c];
        float bias[out_c];
        int8_t out_ref[out_h * out_w * out_c];
        int8_t out_arm[out_h * out_w * out_c];

        for (int i = 0; i < in_h * in_w * in_c; ++i) in[i] = rand_i8();
        for (int i = 0; i < k_h * k_w * in_c * out_c; ++i) filt[i] = rand_i8();
        for (int i = 0; i < out_c; ++i) bias[i] = rand_f32(0.2f);

        conv2d_nhwc_int8(
            in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.02f, 0.03f, 0.05f, 0, out_ref);
        conv2d_nhwc_int8_arm(
            in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.02f, 0.03f, 0.05f, 0, out_arm);

        if (compare_i8(out_ref, out_arm, out_h * out_w * out_c, "conv2d_nhwc_int8")) {
            return 1;
        }
    }

    {
        const int in_h = 4, in_w = 4, in_c = 11;
        const int k_h = 3, k_w = 1, out_c = 16;
        const int stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 0;
        const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

        int8_t in[in_h * in_w * in_c];
        int8_t filt[k_h * k_w * in_c * out_c];
        float bias[out_c];
        float w_scales[out_c];
        int8_t out_ref[out_h * out_w * out_c];
        int8_t out_arm[out_h * out_w * out_c];

        for (int i = 0; i < in_h * in_w * in_c; ++i) in[i] = rand_i8();
        for (int i = 0; i < k_h * k_w * in_c * out_c; ++i) filt[i] = rand_i8();
        for (int i = 0; i < out_c; ++i) {
            bias[i] = rand_f32(0.1f);
            w_scales[i] = 0.005f + (float)(i + 1) * 0.0005f;
        }

        conv2d_nhwc_int8_per_channel(
            in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.04f, w_scales, 0.08f, 0, out_ref);
        conv2d_nhwc_int8_per_channel_arm(
            in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.04f, w_scales, 0.08f, 0, out_arm);

        if (compare_i8(out_ref, out_arm, out_h * out_w * out_c, "conv2d_per_channel")) {
            return 1;
        }
    }

    {
        const int in_h = 6, in_w = 5, channels = 17;
        const int k_h = 3, k_w = 3;
        const int stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
        const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

        int8_t in[in_h * in_w * channels];
        int8_t filt[k_h * k_w * channels];
        float bias[channels];
        float w_scales[channels];
        int8_t out_ref[out_h * out_w * channels];
        int8_t out_arm[out_h * out_w * channels];
        float out_ref_f[out_h * out_w * channels];
        float out_arm_f[out_h * out_w * channels];

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
        depthwise_conv2d_nhwc_int8_arm(
            in, in_h, in_w, channels, filt, k_h, k_w, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.03f, w_scales, 0.07f, 0, out_arm);

        if (compare_i8(out_ref, out_arm, out_h * out_w * channels, "depthwise_int8")) {
            return 1;
        }

        depthwise_conv2d_nhwc_int8_to_float(
            in, in_h, in_w, channels, filt, k_h, k_w, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.03f, w_scales, out_ref_f);
        depthwise_conv2d_nhwc_int8_to_float_arm(
            in, in_h, in_w, channels, filt, k_h, k_w, bias,
            stride_h, stride_w, pad_h, pad_w,
            0.03f, w_scales, out_arm_f);

        if (compare_f32(out_ref_f, out_arm_f, out_h * out_w * channels, 1e-6f, "depthwise_to_float")) {
            return 1;
        }
    }

    printf("[PASS] ARM conv wrappers match naive conv implementations.\n");
#if !(defined(__ARM_NEON) || defined(__ARM_NEON__))
    printf("[INFO] Built without ARM NEON; this verified fallback parity on current CPU.\n");
#endif
    return 0;
}
