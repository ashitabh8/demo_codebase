/*
 * Quantized Neural Network Operations - int8 (CMSIS-NN conv backend)
 *
 * Includes all operators from nn_ops_int8.h and remaps conv entry points to
 * CMSIS-NN kernels when NN_OPS_USE_CMSIS_NN is enabled and CMSIS headers are
 * available. Otherwise, falls back to the naive reference implementations.
 *
 * Target: Cortex-M (e.g., STM32H747 on Arduino GIGA R1)
 */

#ifndef NN_OPS_INT8_CMSIS_H_
#define NN_OPS_INT8_CMSIS_H_

#include "nn_ops_int8.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(NN_OPS_USE_CMSIS_NN)
#if defined(__has_include)
#if __has_include("cmsis_nn.h")
#include "cmsis_nn.h"
#define NN_OPS_CMSIS_AVAILABLE 1
#endif
#endif
#endif

#ifndef NN_OPS_CMSIS_AVAILABLE
#define NN_OPS_CMSIS_AVAILABLE 0
#endif

#if NN_OPS_CMSIS_AVAILABLE
static inline void nn_ops_quantize_multiplier(
    float real_multiplier,
    int32_t* quantized_multiplier,
    int32_t* shift)
{
    if (real_multiplier <= 0.0f) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }

    int exp = 0;
    double q = frexp((double)real_multiplier, &exp);
    int64_t q_fixed = (int64_t)llround(q * (double)(1ll << 31));

    if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        ++exp;
    }

    *quantized_multiplier = (int32_t)q_fixed;
    *shift = (int32_t)exp;
}
#endif

static inline void conv2d_nhwc_int8_cmsis(
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
#if NN_OPS_CMSIS_AVAILABLE
    if (output_scale > 0.0f) {
        const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
        const float real_multiplier = (input_scale * weight_scale) / output_scale;

        int32_t multiplier = 0;
        int32_t shift = 0;
        nn_ops_quantize_multiplier(real_multiplier, &multiplier, &shift);

        int32_t* bias_q = (int32_t*)malloc((size_t)out_c * sizeof(int32_t));
        if (bias_q != NULL) {
            const float bias_scale = input_scale * weight_scale;
            if (bias != NULL && bias_scale > 0.0f) {
                for (int oc = 0; oc < out_c; ++oc) {
                    bias_q[oc] = (int32_t)llround((double)bias[oc] / (double)bias_scale);
                }
            } else {
                memset(bias_q, 0, (size_t)out_c * sizeof(int32_t));
            }

            int32_t* mult_arr = (int32_t*)malloc((size_t)out_c * sizeof(int32_t));
            int32_t* shift_arr = (int32_t*)malloc((size_t)out_c * sizeof(int32_t));
            if (mult_arr != NULL && shift_arr != NULL) {
                for (int oc = 0; oc < out_c; ++oc) {
                    mult_arr[oc] = multiplier;
                    shift_arr[oc] = shift;
                }

                cmsis_nn_context ctx;
                ctx.buf = NULL;
                ctx.size = 0;

                cmsis_nn_conv_params conv_params;
                conv_params.input_offset = 0;
                conv_params.output_offset = offset;
                conv_params.stride.w = stride_w;
                conv_params.stride.h = stride_h;
                conv_params.padding.w = pad_w;
                conv_params.padding.h = pad_h;
                conv_params.dilation.w = 1;
                conv_params.dilation.h = 1;
                conv_params.activation.min = -128;
                conv_params.activation.max = 127;

                cmsis_nn_per_channel_quant_params quant_params;
                quant_params.multiplier = mult_arr;
                quant_params.shift = shift_arr;

                cmsis_nn_dims input_dims;
                input_dims.n = 1;
                input_dims.h = in_h;
                input_dims.w = in_w;
                input_dims.c = in_c;

                cmsis_nn_dims filter_dims;
                filter_dims.n = out_c;
                filter_dims.h = k_h;
                filter_dims.w = k_w;
                filter_dims.c = in_c;

                cmsis_nn_dims bias_dims;
                bias_dims.n = 1;
                bias_dims.h = 1;
                bias_dims.w = 1;
                bias_dims.c = out_c;

                cmsis_nn_dims output_dims;
                output_dims.n = 1;
                output_dims.h = out_h;
                output_dims.w = out_w;
                output_dims.c = out_c;

                const arm_cmsis_nn_status status = arm_convolve_s8(
                    &ctx,
                    &conv_params,
                    &quant_params,
                    &input_dims,
                    in,
                    &filter_dims,
                    filt,
                    &bias_dims,
                    bias_q,
                    &output_dims,
                    out);

                free(mult_arr);
                free(shift_arr);
                free(bias_q);

                if (status == ARM_CMSIS_NN_SUCCESS) {
                    return;
                }
            } else {
                free(mult_arr);
                free(shift_arr);
            }
            free(bias_q);
        }
    }
#endif

    conv2d_nhwc_int8(
        in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
        stride_h, stride_w, pad_h, pad_w, input_scale, weight_scale,
        output_scale, offset, out
    );
}

static inline void conv2d_nhwc_int8_per_channel_cmsis(
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
#if NN_OPS_CMSIS_AVAILABLE
    if (output_scale > 0.0f && weight_scales != NULL) {
        const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

        int32_t* bias_q = (int32_t*)malloc((size_t)out_c * sizeof(int32_t));
        int32_t* mult_arr = (int32_t*)malloc((size_t)out_c * sizeof(int32_t));
        int32_t* shift_arr = (int32_t*)malloc((size_t)out_c * sizeof(int32_t));

        if (bias_q != NULL && mult_arr != NULL && shift_arr != NULL) {
            for (int oc = 0; oc < out_c; ++oc) {
                const float w_scale = weight_scales[oc];
                const float bias_scale = input_scale * w_scale;
                if (bias != NULL && bias_scale > 0.0f) {
                    bias_q[oc] = (int32_t)llround((double)bias[oc] / (double)bias_scale);
                } else {
                    bias_q[oc] = 0;
                }
                nn_ops_quantize_multiplier((input_scale * w_scale) / output_scale, &mult_arr[oc], &shift_arr[oc]);
            }

            cmsis_nn_context ctx;
            ctx.buf = NULL;
            ctx.size = 0;

            cmsis_nn_conv_params conv_params;
            conv_params.input_offset = 0;
            conv_params.output_offset = offset;
            conv_params.stride.w = stride_w;
            conv_params.stride.h = stride_h;
            conv_params.padding.w = pad_w;
            conv_params.padding.h = pad_h;
            conv_params.dilation.w = 1;
            conv_params.dilation.h = 1;
            conv_params.activation.min = -128;
            conv_params.activation.max = 127;

            cmsis_nn_per_channel_quant_params quant_params;
            quant_params.multiplier = mult_arr;
            quant_params.shift = shift_arr;

            cmsis_nn_dims input_dims;
            input_dims.n = 1;
            input_dims.h = in_h;
            input_dims.w = in_w;
            input_dims.c = in_c;

            cmsis_nn_dims filter_dims;
            filter_dims.n = out_c;
            filter_dims.h = k_h;
            filter_dims.w = k_w;
            filter_dims.c = in_c;

            cmsis_nn_dims bias_dims;
            bias_dims.n = 1;
            bias_dims.h = 1;
            bias_dims.w = 1;
            bias_dims.c = out_c;

            cmsis_nn_dims output_dims;
            output_dims.n = 1;
            output_dims.h = out_h;
            output_dims.w = out_w;
            output_dims.c = out_c;

            const arm_cmsis_nn_status status = arm_convolve_s8(
                &ctx,
                &conv_params,
                &quant_params,
                &input_dims,
                in,
                &filter_dims,
                filt,
                &bias_dims,
                bias_q,
                &output_dims,
                out);

            free(bias_q);
            free(mult_arr);
            free(shift_arr);

            if (status == ARM_CMSIS_NN_SUCCESS) {
                return;
            }
        } else {
            free(bias_q);
            free(mult_arr);
            free(shift_arr);
        }
    }
#endif

    conv2d_nhwc_int8_per_channel(
        in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
        stride_h, stride_w, pad_h, pad_w, input_scale, weight_scales,
        output_scale, offset, out
    );
}

static inline void depthwise_conv2d_nhwc_int8_cmsis(
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
#if NN_OPS_CMSIS_AVAILABLE
    if (output_scale > 0.0f && weight_scales != NULL) {
        const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

        int32_t* bias_q = (int32_t*)malloc((size_t)channels * sizeof(int32_t));
        int32_t* mult_arr = (int32_t*)malloc((size_t)channels * sizeof(int32_t));
        int32_t* shift_arr = (int32_t*)malloc((size_t)channels * sizeof(int32_t));

        if (bias_q != NULL && mult_arr != NULL && shift_arr != NULL) {
            for (int c = 0; c < channels; ++c) {
                const float w_scale = weight_scales[c];
                const float bias_scale = input_scale * w_scale;
                if (bias != NULL && bias_scale > 0.0f) {
                    bias_q[c] = (int32_t)llround((double)bias[c] / (double)bias_scale);
                } else {
                    bias_q[c] = 0;
                }
                nn_ops_quantize_multiplier((input_scale * w_scale) / output_scale, &mult_arr[c], &shift_arr[c]);
            }

            cmsis_nn_context ctx;
            ctx.buf = NULL;
            ctx.size = 0;

            cmsis_nn_dw_conv_params dw_params;
            dw_params.input_offset = 0;
            dw_params.output_offset = offset;
            dw_params.ch_mult = 1;
            dw_params.stride.w = stride_w;
            dw_params.stride.h = stride_h;
            dw_params.padding.w = pad_w;
            dw_params.padding.h = pad_h;
            dw_params.dilation.w = 1;
            dw_params.dilation.h = 1;
            dw_params.activation.min = -128;
            dw_params.activation.max = 127;

            cmsis_nn_per_channel_quant_params quant_params;
            quant_params.multiplier = mult_arr;
            quant_params.shift = shift_arr;

            cmsis_nn_dims input_dims;
            input_dims.n = 1;
            input_dims.h = in_h;
            input_dims.w = in_w;
            input_dims.c = channels;

            cmsis_nn_dims filter_dims;
            filter_dims.n = 1;
            filter_dims.h = k_h;
            filter_dims.w = k_w;
            filter_dims.c = channels;

            cmsis_nn_dims bias_dims;
            bias_dims.n = 1;
            bias_dims.h = 1;
            bias_dims.w = 1;
            bias_dims.c = channels;

            cmsis_nn_dims output_dims;
            output_dims.n = 1;
            output_dims.h = out_h;
            output_dims.w = out_w;
            output_dims.c = channels;

            const arm_cmsis_nn_status status = arm_depthwise_conv_s8(
                &ctx,
                &dw_params,
                &quant_params,
                &input_dims,
                in,
                &filter_dims,
                filt,
                &bias_dims,
                bias_q,
                &output_dims,
                out);

            free(bias_q);
            free(mult_arr);
            free(shift_arr);

            if (status == ARM_CMSIS_NN_SUCCESS) {
                return;
            }
        } else {
            free(bias_q);
            free(mult_arr);
            free(shift_arr);
        }
    }
#endif

    depthwise_conv2d_nhwc_int8(
        in, in_h, in_w, channels, filt, k_h, k_w, bias, stride_h, stride_w,
        pad_h, pad_w, input_scale, weight_scales, output_scale, offset, out
    );
}

/*
 * CMSIS-NN kernels produce int8 outputs. This float-output variant keeps the
 * original semantics by using the reference implementation.
 */
static inline void depthwise_conv2d_nhwc_int8_to_float_cmsis(
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
    depthwise_conv2d_nhwc_int8_to_float(
        in, in_h, in_w, channels, filt, k_h, k_w, bias, stride_h, stride_w,
        pad_h, pad_w, input_scale, weight_scales, out
    );
}

/*
 * CMSIS-NN conv expects int8 input. This float-input variant preserves the
 * current behavior by keeping the reference path.
 */
static inline void conv2d_nhwc_float_input_int8_weight_per_channel_cmsis(
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

#ifndef NN_OPS_INT8_CMSIS_DISABLE_REMAP
#define conv2d_nhwc_int8 conv2d_nhwc_int8_cmsis
#define conv2d_nhwc_int8_per_channel conv2d_nhwc_int8_per_channel_cmsis
#define depthwise_conv2d_nhwc_int8 depthwise_conv2d_nhwc_int8_cmsis
#define depthwise_conv2d_nhwc_int8_to_float depthwise_conv2d_nhwc_int8_to_float_cmsis
#define conv2d_nhwc_float_input_int8_weight_per_channel \
    conv2d_nhwc_float_input_int8_weight_per_channel_cmsis
#endif

#endif /* NN_OPS_INT8_CMSIS_H_ */
