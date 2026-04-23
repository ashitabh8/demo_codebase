/*
 * Quantized Neural Network Operations - int8 (ARM-optimized conv variants)
 *
 * This header includes all operators from nn_ops_int8.h and remaps only the
 * convolution entry points to ARM NEON accelerated kernels when available.
 * Non-ARM targets fall back to the naive reference implementations.
 */

#ifndef NN_OPS_INT8_ARM_H_
#define NN_OPS_INT8_ARM_H_

#include "nn_ops_int8.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

static inline void conv2d_nhwc_int8_arm(
    const int8_t* in, int in_h, int in_w, int in_c,
    const int8_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    float weight_scale,
    float output_scale,
    int offset,
    int8_t* out)
{
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    const float combined_scale = input_scale * weight_scale;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            int oc = 0;

            for (; oc + 7 < out_c; oc += 8) {
                int32x4_t acc_lo = vdupq_n_s32(0);
                int32x4_t acc_hi = vdupq_n_s32(0);

                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }

                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }

                        const int8_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int8_t* f_base = filt + (((kh * k_w + kw) * in_c) * out_c + oc);

                        for (int ic = 0; ic < in_c; ++ic) {
                            const int8_t x = in_px[ic];
                            const int8_t* w_ptr = f_base + ic * out_c;
                            const int8x8_t w_vec = vld1_s8(w_ptr);
                            const int16x8_t prod = vmull_n_s8(w_vec, x);
                            acc_lo = vaddw_s16(acc_lo, vget_low_s16(prod));
                            acc_hi = vaddw_s16(acc_hi, vget_high_s16(prod));
                        }
                    }
                }

                int32_t acc_buf[8];
                vst1q_s32(acc_buf, acc_lo);
                vst1q_s32(acc_buf + 4, acc_hi);
                for (int lane = 0; lane < 8; ++lane) {
                    float result = (float)acc_buf[lane] * combined_scale;
                    if (bias != NULL) {
                        result += bias[oc + lane];
                    }
                    out[((oh * out_w + ow) * out_c) + (oc + lane)] =
                        quantize_scalar_int8(result, output_scale, offset);
                }
            }

            for (; oc < out_c; ++oc) {
                int32_t acc = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }
                        const int8_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int8_t* f_base = filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        for (int ic = 0; ic < in_c; ++ic) {
                            acc += (int32_t)in_px[ic] * (int32_t)f_base[ic * out_c];
                        }
                    }
                }
                float result = (float)acc * combined_scale;
                if (bias != NULL) {
                    result += bias[oc];
                }
                out[((oh * out_w + ow) * out_c) + oc] =
                    quantize_scalar_int8(result, output_scale, offset);
            }
        }
    }
#else
    conv2d_nhwc_int8(
        in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
        stride_h, stride_w, pad_h, pad_w, input_scale, weight_scale,
        output_scale, offset, out
    );
#endif
}

static inline void conv2d_nhwc_int8_per_channel_arm(
    const int8_t* in, int in_h, int in_w, int in_c,
    const int8_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int offset,
    int8_t* out)
{
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            int oc = 0;

            for (; oc + 7 < out_c; oc += 8) {
                int32x4_t acc_lo = vdupq_n_s32(0);
                int32x4_t acc_hi = vdupq_n_s32(0);

                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }
                        const int8_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int8_t* f_base = filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        for (int ic = 0; ic < in_c; ++ic) {
                            const int8_t x = in_px[ic];
                            const int8_t* w_ptr = f_base + ic * out_c;
                            const int8x8_t w_vec = vld1_s8(w_ptr);
                            const int16x8_t prod = vmull_n_s8(w_vec, x);
                            acc_lo = vaddw_s16(acc_lo, vget_low_s16(prod));
                            acc_hi = vaddw_s16(acc_hi, vget_high_s16(prod));
                        }
                    }
                }

                int32_t acc_buf[8];
                vst1q_s32(acc_buf, acc_lo);
                vst1q_s32(acc_buf + 4, acc_hi);
                for (int lane = 0; lane < 8; ++lane) {
                    const int out_idx = oc + lane;
                    float result = (float)acc_buf[lane] * input_scale * weight_scales[out_idx];
                    if (bias != NULL) {
                        result += bias[out_idx];
                    }
                    out[((oh * out_w + ow) * out_c) + out_idx] =
                        quantize_scalar_int8(result, output_scale, offset);
                }
            }

            for (; oc < out_c; ++oc) {
                int32_t acc = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }
                        const int8_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int8_t* f_base = filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        for (int ic = 0; ic < in_c; ++ic) {
                            acc += (int32_t)in_px[ic] * (int32_t)f_base[ic * out_c];
                        }
                    }
                }
                float result = (float)acc * input_scale * weight_scales[oc];
                if (bias != NULL) {
                    result += bias[oc];
                }
                out[((oh * out_w + ow) * out_c) + oc] =
                    quantize_scalar_int8(result, output_scale, offset);
            }
        }
    }
#else
    conv2d_nhwc_int8_per_channel(
        in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
        stride_h, stride_w, pad_h, pad_w, input_scale, weight_scales,
        output_scale, offset, out
    );
#endif
}

static inline void depthwise_conv2d_nhwc_int8_arm(
    const int8_t* in,
    int in_h,
    int in_w,
    int channels,
    const int8_t* filt,
    int k_h,
    int k_w,
    const float* bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int offset,
    int8_t* out)
{
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            int c = 0;

            for (; c + 15 < channels; c += 16) {
                int32x4_t acc0 = vdupq_n_s32(0);
                int32x4_t acc1 = vdupq_n_s32(0);
                int32x4_t acc2 = vdupq_n_s32(0);
                int32x4_t acc3 = vdupq_n_s32(0);

                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }

                        const int8_t* x_ptr = in + ((ih * in_w + iw) * channels) + c;
                        const int8_t* w_ptr = filt + ((kh * k_w + kw) * channels) + c;
                        const int8x16_t x_vec = vld1q_s8(x_ptr);
                        const int8x16_t w_vec = vld1q_s8(w_ptr);

                        const int16x8_t p0 = vmull_s8(vget_low_s8(x_vec), vget_low_s8(w_vec));
                        const int16x8_t p1 = vmull_s8(vget_high_s8(x_vec), vget_high_s8(w_vec));

                        acc0 = vaddw_s16(acc0, vget_low_s16(p0));
                        acc1 = vaddw_s16(acc1, vget_high_s16(p0));
                        acc2 = vaddw_s16(acc2, vget_low_s16(p1));
                        acc3 = vaddw_s16(acc3, vget_high_s16(p1));
                    }
                }

                int32_t acc_buf[16];
                vst1q_s32(acc_buf, acc0);
                vst1q_s32(acc_buf + 4, acc1);
                vst1q_s32(acc_buf + 8, acc2);
                vst1q_s32(acc_buf + 12, acc3);

                for (int lane = 0; lane < 16; ++lane) {
                    const int ch = c + lane;
                    float result = (float)acc_buf[lane] * input_scale * weight_scales[ch];
                    if (bias != NULL) {
                        result += bias[ch];
                    }
                    out[((oh * out_w + ow) * channels) + ch] =
                        quantize_scalar_int8(result, output_scale, offset);
                }
            }

            for (; c < channels; ++c) {
                int32_t acc = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }
                        const int8_t x = in[((ih * in_w + iw) * channels) + c];
                        const int8_t w = filt[((kh * k_w + kw) * channels) + c];
                        acc += (int32_t)x * (int32_t)w;
                    }
                }
                float result = (float)acc * input_scale * weight_scales[c];
                if (bias != NULL) {
                    result += bias[c];
                }
                out[((oh * out_w + ow) * channels) + c] =
                    quantize_scalar_int8(result, output_scale, offset);
            }
        }
    }
#else
    depthwise_conv2d_nhwc_int8(
        in, in_h, in_w, channels, filt, k_h, k_w, bias, stride_h, stride_w,
        pad_h, pad_w, input_scale, weight_scales, output_scale, offset, out
    );
#endif
}

static inline void depthwise_conv2d_nhwc_int8_to_float_arm(
    const int8_t* in,
    int in_h,
    int in_w,
    int channels,
    const int8_t* filt,
    int k_h,
    int k_w,
    const float* bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    float input_scale,
    const float* weight_scales,
    float* out)
{
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            int c = 0;

            for (; c + 15 < channels; c += 16) {
                int32x4_t acc0 = vdupq_n_s32(0);
                int32x4_t acc1 = vdupq_n_s32(0);
                int32x4_t acc2 = vdupq_n_s32(0);
                int32x4_t acc3 = vdupq_n_s32(0);

                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }

                        const int8_t* x_ptr = in + ((ih * in_w + iw) * channels) + c;
                        const int8_t* w_ptr = filt + ((kh * k_w + kw) * channels) + c;
                        const int8x16_t x_vec = vld1q_s8(x_ptr);
                        const int8x16_t w_vec = vld1q_s8(w_ptr);

                        const int16x8_t p0 = vmull_s8(vget_low_s8(x_vec), vget_low_s8(w_vec));
                        const int16x8_t p1 = vmull_s8(vget_high_s8(x_vec), vget_high_s8(w_vec));

                        acc0 = vaddw_s16(acc0, vget_low_s16(p0));
                        acc1 = vaddw_s16(acc1, vget_high_s16(p0));
                        acc2 = vaddw_s16(acc2, vget_low_s16(p1));
                        acc3 = vaddw_s16(acc3, vget_high_s16(p1));
                    }
                }

                int32_t acc_buf[16];
                vst1q_s32(acc_buf, acc0);
                vst1q_s32(acc_buf + 4, acc1);
                vst1q_s32(acc_buf + 8, acc2);
                vst1q_s32(acc_buf + 12, acc3);

                for (int lane = 0; lane < 16; ++lane) {
                    const int ch = c + lane;
                    float result = (float)acc_buf[lane] * input_scale * weight_scales[ch];
                    if (bias != NULL) {
                        result += bias[ch];
                    }
                    out[((oh * out_w + ow) * channels) + ch] = result;
                }
            }

            for (; c < channels; ++c) {
                int32_t acc = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    const int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        const int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }
                        const int8_t x = in[((ih * in_w + iw) * channels) + c];
                        const int8_t w = filt[((kh * k_w + kw) * channels) + c];
                        acc += (int32_t)x * (int32_t)w;
                    }
                }
                float result = (float)acc * input_scale * weight_scales[c];
                if (bias != NULL) {
                    result += bias[c];
                }
                out[((oh * out_w + ow) * channels) + c] = result;
            }
        }
    }
#else
    depthwise_conv2d_nhwc_int8_to_float(
        in, in_h, in_w, channels, filt, k_h, k_w, bias, stride_h, stride_w,
        pad_h, pad_w, input_scale, weight_scales, out
    );
#endif
}

static inline void conv2d_nhwc_float_input_int8_weight_per_channel_arm(
    const float* in, int in_h, int in_w, int in_c,
    const int8_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    const float* weight_scales,
    float* out)
{
    conv2d_nhwc_float_input_int8_weight_per_channel(
        in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
        stride_h, stride_w, pad_h, pad_w, weight_scales, out
    );
}

#ifndef NN_OPS_INT8_ARM_DISABLE_REMAP
#define conv2d_nhwc_int8 conv2d_nhwc_int8_arm
#define conv2d_nhwc_int8_per_channel conv2d_nhwc_int8_per_channel_arm
#define depthwise_conv2d_nhwc_int8 depthwise_conv2d_nhwc_int8_arm
#define depthwise_conv2d_nhwc_int8_to_float depthwise_conv2d_nhwc_int8_to_float_arm
#define conv2d_nhwc_float_input_int8_weight_per_channel \
    conv2d_nhwc_float_input_int8_weight_per_channel_arm
#endif

#endif /* NN_OPS_INT8_ARM_H_ */
