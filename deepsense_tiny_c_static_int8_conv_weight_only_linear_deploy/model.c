// Auto-generated model implementation
// DO NOT EDIT

// Input shape  (NCHW): [1, 1, 7, 80]
// Output shape: [1, 5]

#include "model.h"
#include "weights.h"
#include "nn_ops_float.h"
#include "nn_ops_int8.h"

#include <string.h>

void model_forward(const float* input, float* output) {
    static int8_t slot_0[5852];
    static int8_t slot_1[5852];
    static float slot_2[5852];
    static float slot_3[5852];

    // wrapped_backbone_freq_stack_0_depthwise_input_q [quantize]
    quantize_float_to_int8(input, 560, 0.1536287322757751f, 0, slot_0);
    // wrapped_backbone_freq_stack_0_depthwise [conv2d]
    conv2d_nhwc_int8_per_channel(slot_0, 7, 80, 1, wrapped_backbone_freq_stack_0_depthwise_weight, 1, 5, 1, NULL, 1, 2, 0, 0, 0.1536287322757751f, wrapped_backbone_freq_stack_0_depthwise_weight_per_channel_scales, 0.09890554082675243f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_0_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 266, 0.09890554082675243f, 0, slot_2);
    // wrapped_backbone_freq_stack_0_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 266, 0.09890554082675243f, 0, slot_0);
    // wrapped_backbone_freq_stack_0_pointwise [conv2d]
    conv2d_nhwc_int8_per_channel(slot_0, 7, 38, 1, wrapped_backbone_freq_stack_0_pointwise_weight, 1, 1, 22, wrapped_backbone_freq_stack_0_pointwise_bias, 1, 1, 0, 0, 0.09890554082675243f, wrapped_backbone_freq_stack_0_pointwise_weight_per_channel_scales, 0.09051995765505813f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_0_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 5852, 0.09051995765505813f, 0, slot_2);
    // wrapped_backbone_freq_stack_0_bn [batchnorm]
    batchnorm2d_nhwc(slot_2, 7, 38, 22, wrapped_backbone_freq_stack_0_bn_gamma, wrapped_backbone_freq_stack_0_bn_beta, wrapped_backbone_freq_stack_0_bn_mean, wrapped_backbone_freq_stack_0_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_freq_stack_0_act [gelu]
    gelu(slot_3, 5852);
    // wrapped_backbone_freq_stack_1_depthwise_input_q [quantize]
    quantize_float_to_int8(slot_3, 5852, 0.016170725109070305f, 0, slot_0);
    // wrapped_backbone_freq_stack_1_depthwise [conv2d]
    depthwise_conv2d_nhwc_int8_per_channel(slot_0, 7, 38, 22, wrapped_backbone_freq_stack_1_depthwise_weight, 1, 3, NULL, 1, 2, 0, 0, 0.016170725109070305f, wrapped_backbone_freq_stack_1_depthwise_weight_per_channel_scales, 0.012607383915758509f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_1_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 2772, 0.012607383915758509f, 0, slot_2);
    // wrapped_backbone_freq_stack_1_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 2772, 0.012607383915758509f, 0, slot_0);
    // wrapped_backbone_freq_stack_1_pointwise [conv2d]
    conv2d_nhwc_int8_per_channel(slot_0, 7, 18, 22, wrapped_backbone_freq_stack_1_pointwise_weight, 1, 1, 32, wrapped_backbone_freq_stack_1_pointwise_bias, 1, 1, 0, 0, 0.012607383915758509f, wrapped_backbone_freq_stack_1_pointwise_weight_per_channel_scales, 0.009925036918459914f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_1_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 4032, 0.009925036918459914f, 0, slot_2);
    // wrapped_backbone_freq_stack_1_bn [batchnorm]
    batchnorm2d_nhwc(slot_2, 7, 18, 32, wrapped_backbone_freq_stack_1_bn_gamma, wrapped_backbone_freq_stack_1_bn_beta, wrapped_backbone_freq_stack_1_bn_mean, wrapped_backbone_freq_stack_1_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_freq_stack_1_act [gelu]
    gelu(slot_3, 4032);
    // wrapped_backbone_freq_stack_2_depthwise_input_q [quantize]
    quantize_float_to_int8(slot_3, 4032, 0.03413985094686193f, 0, slot_0);
    // wrapped_backbone_freq_stack_2_depthwise [conv2d]
    depthwise_conv2d_nhwc_int8_per_channel(slot_0, 7, 18, 32, wrapped_backbone_freq_stack_2_depthwise_weight, 1, 3, NULL, 1, 1, 0, 1, 0.03413985094686193f, wrapped_backbone_freq_stack_2_depthwise_weight_per_channel_scales, 0.019685075039000022f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_2_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 4032, 0.019685075039000022f, 0, slot_2);
    // wrapped_backbone_freq_stack_2_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 4032, 0.019685075039000022f, 0, slot_0);
    // wrapped_backbone_freq_stack_2_pointwise [conv2d]
    conv2d_nhwc_int8_per_channel(slot_0, 7, 18, 32, wrapped_backbone_freq_stack_2_pointwise_weight, 1, 1, 43, wrapped_backbone_freq_stack_2_pointwise_bias, 1, 1, 0, 0, 0.019685075039000022f, wrapped_backbone_freq_stack_2_pointwise_weight_per_channel_scales, 0.024614135111410786f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_2_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 5418, 0.024614135111410786f, 0, slot_2);
    // wrapped_backbone_freq_stack_2_bn [batchnorm]
    batchnorm2d_nhwc(slot_2, 7, 18, 43, wrapped_backbone_freq_stack_2_bn_gamma, wrapped_backbone_freq_stack_2_bn_beta, wrapped_backbone_freq_stack_2_bn_mean, wrapped_backbone_freq_stack_2_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_freq_stack_2_act [gelu]
    gelu(slot_3, 5418);
    // wrapped_backbone_freq_stack_3_depthwise_input_q [quantize]
    quantize_float_to_int8(slot_3, 5418, 0.062444401538278176f, 0, slot_0);
    // wrapped_backbone_freq_stack_3_depthwise [conv2d]
    depthwise_conv2d_nhwc_int8_per_channel(slot_0, 7, 18, 43, wrapped_backbone_freq_stack_3_depthwise_weight, 1, 3, NULL, 1, 1, 0, 1, 0.062444401538278176f, wrapped_backbone_freq_stack_3_depthwise_weight_per_channel_scales, 0.03750483445295199f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_3_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 5418, 0.03750483445295199f, 0, slot_2);
    // wrapped_backbone_freq_stack_3_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 5418, 0.03750483445295199f, 0, slot_0);
    // wrapped_backbone_freq_stack_3_pointwise [conv2d]
    conv2d_nhwc_int8_per_channel(slot_0, 7, 18, 43, wrapped_backbone_freq_stack_3_pointwise_weight, 1, 1, 43, wrapped_backbone_freq_stack_3_pointwise_bias, 1, 1, 0, 0, 0.03750483445295199f, wrapped_backbone_freq_stack_3_pointwise_weight_per_channel_scales, 0.03388325999102255f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_3_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 5418, 0.03388325999102255f, 0, slot_2);
    // wrapped_backbone_freq_stack_3_bn [batchnorm]
    batchnorm2d_nhwc(slot_2, 7, 18, 43, wrapped_backbone_freq_stack_3_bn_gamma, wrapped_backbone_freq_stack_3_bn_beta, wrapped_backbone_freq_stack_3_bn_mean, wrapped_backbone_freq_stack_3_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_freq_stack_3_act [gelu]
    gelu(slot_3, 5418);
    // wrapped_backbone_freq_stack_4_depthwise_input_q [quantize]
    quantize_float_to_int8(slot_3, 5418, 0.09287617901178795f, 0, slot_0);
    // wrapped_backbone_freq_stack_4_depthwise [conv2d]
    depthwise_conv2d_nhwc_int8_per_channel(slot_0, 7, 18, 43, wrapped_backbone_freq_stack_4_depthwise_weight, 1, 3, NULL, 1, 1, 0, 1, 0.09287617901178795f, wrapped_backbone_freq_stack_4_depthwise_weight_per_channel_scales, 0.06608971272866557f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_4_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 5418, 0.06608971272866557f, 0, slot_2);
    // wrapped_backbone_freq_stack_4_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 5418, 0.06608971272866557f, 0, slot_0);
    // wrapped_backbone_freq_stack_4_pointwise [conv2d]
    conv2d_nhwc_int8_per_channel(slot_0, 7, 18, 43, wrapped_backbone_freq_stack_4_pointwise_weight, 1, 1, 43, wrapped_backbone_freq_stack_4_pointwise_bias, 1, 1, 0, 0, 0.06608971272866557f, wrapped_backbone_freq_stack_4_pointwise_weight_per_channel_scales, 0.055625784115528494f, 0, 0, 0, slot_1);
    // wrapped_backbone_freq_stack_4_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 5418, 0.055625784115528494f, 0, slot_2);
    // wrapped_backbone_freq_stack_4_bn [batchnorm]
    batchnorm2d_nhwc(slot_2, 7, 18, 43, wrapped_backbone_freq_stack_4_bn_gamma, wrapped_backbone_freq_stack_4_bn_beta, wrapped_backbone_freq_stack_4_bn_mean, wrapped_backbone_freq_stack_4_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_freq_stack_4_act [gelu]
    gelu(slot_3, 5418);
    // getattr_1 [method_getattr]
    // getitem [method_getitem]
    // getitem_1 [method_getitem]
    // getitem_2 [method_getitem]
    // getitem_3 [method_getitem]
    // permute [method_permute]
    /* permute(0,2,1,3): source NHWC [H,W,C] -> [H,C,W] */
    for (int hh = 0; hh < 7; ++hh) {
        for (int cc = 0; cc < 43; ++cc) {
            for (int ww = 0; ww < 18; ++ww) {
                slot_2[((hh * 43 + cc) * 18) + ww] = slot_3[((hh * 18 + ww) * 43) + cc];
            }
        }
    }
    // mul [mul]
    // reshape [method_reshape]
    memcpy(slot_3, slot_2, 5418 * sizeof(float));
    // wrapped_backbone_spectrum_proj [linear]
    for (int r = 0; r < 7; ++r) {
        dense_float_input_int8_weight_per_channel(slot_3 + r * 774, 774, wrapped_backbone_spectrum_proj_weight, wrapped_backbone_spectrum_proj_bias, 42, wrapped_backbone_spectrum_proj_weight_per_channel_scales, slot_2 + r * 42);
    }
    // permute_1 [method_permute]
    /* permute(0,2,1) after linear: row-major == NLC, memcpy */
    memcpy(slot_3, slot_2, 294 * sizeof(float));
    // wrapped_backbone_temporal_stack_0_depthwise_input_q [quantize]
    quantize_float_to_int8(slot_3, 294, 0.19965655409444974f, 0, slot_0);
    // wrapped_backbone_temporal_stack_0_depthwise [conv1d]
    depthwise_conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_0_depthwise_weight, 1, 3, NULL, 1, 1, 0, 1, 0.19965655409444974f, wrapped_backbone_temporal_stack_0_depthwise_weight_per_channel_scales, 0.10309379307303841f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_0_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.10309379307303841f, 0, slot_2);
    // wrapped_backbone_temporal_stack_0_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 294, 0.10309379307303841f, 0, slot_0);
    // wrapped_backbone_temporal_stack_0_pointwise [conv1d]
    conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_0_pointwise_weight, 1, 1, 42, wrapped_backbone_temporal_stack_0_pointwise_bias, 1, 1, 0, 0, 0.10309379307303841f, wrapped_backbone_temporal_stack_0_pointwise_weight_per_channel_scales, 0.07196937020369402f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_0_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.07196937020369402f, 0, slot_2);
    // wrapped_backbone_temporal_stack_0_bn [batchnorm1d]
    batchnorm2d_nhwc(slot_2, 1, 7, 42, wrapped_backbone_temporal_stack_0_bn_gamma, wrapped_backbone_temporal_stack_0_bn_beta, wrapped_backbone_temporal_stack_0_bn_mean, wrapped_backbone_temporal_stack_0_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_temporal_stack_0_act [gelu]
    gelu(slot_3, 294);
    // wrapped_backbone_temporal_stack_1_depthwise_input_q [quantize]
    quantize_float_to_int8(slot_3, 294, 0.07889597434697189f, 0, slot_0);
    // wrapped_backbone_temporal_stack_1_depthwise [conv1d]
    depthwise_conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_1_depthwise_weight, 1, 3, NULL, 1, 1, 0, 1, 0.07889597434697189f, wrapped_backbone_temporal_stack_1_depthwise_weight_per_channel_scales, 0.04882806868065061f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_1_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.04882806868065061f, 0, slot_2);
    // wrapped_backbone_temporal_stack_1_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 294, 0.04882806868065061f, 0, slot_0);
    // wrapped_backbone_temporal_stack_1_pointwise [conv1d]
    conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_1_pointwise_weight, 1, 1, 42, wrapped_backbone_temporal_stack_1_pointwise_bias, 1, 1, 0, 0, 0.04882806868065061f, wrapped_backbone_temporal_stack_1_pointwise_weight_per_channel_scales, 0.02187255987032192f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_1_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.02187255987032192f, 0, slot_2);
    // wrapped_backbone_temporal_stack_1_bn [batchnorm1d]
    batchnorm2d_nhwc(slot_2, 1, 7, 42, wrapped_backbone_temporal_stack_1_bn_gamma, wrapped_backbone_temporal_stack_1_bn_beta, wrapped_backbone_temporal_stack_1_bn_mean, wrapped_backbone_temporal_stack_1_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_temporal_stack_1_act [gelu]
    gelu(slot_3, 294);
    // wrapped_backbone_temporal_stack_2_depthwise_input_q [quantize]
    quantize_float_to_int8(slot_3, 294, 0.043157994277833955f, 0, slot_0);
    // wrapped_backbone_temporal_stack_2_depthwise [conv1d]
    depthwise_conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_2_depthwise_weight, 1, 3, NULL, 1, 1, 0, 1, 0.043157994277833955f, wrapped_backbone_temporal_stack_2_depthwise_weight_per_channel_scales, 0.03603355527862789f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_2_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.03603355527862789f, 0, slot_2);
    // wrapped_backbone_temporal_stack_2_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 294, 0.03603355527862789f, 0, slot_0);
    // wrapped_backbone_temporal_stack_2_pointwise [conv1d]
    conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_2_pointwise_weight, 1, 1, 42, wrapped_backbone_temporal_stack_2_pointwise_bias, 1, 1, 0, 0, 0.03603355527862789f, wrapped_backbone_temporal_stack_2_pointwise_weight_per_channel_scales, 0.015576885441156823f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_2_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.015576885441156823f, 0, slot_2);
    // wrapped_backbone_temporal_stack_2_bn [batchnorm1d]
    batchnorm2d_nhwc(slot_2, 1, 7, 42, wrapped_backbone_temporal_stack_2_bn_gamma, wrapped_backbone_temporal_stack_2_bn_beta, wrapped_backbone_temporal_stack_2_bn_mean, wrapped_backbone_temporal_stack_2_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_temporal_stack_2_act [gelu]
    gelu(slot_3, 294);
    // wrapped_backbone_temporal_stack_3_depthwise_input_q [quantize]
    quantize_float_to_int8(slot_3, 294, 0.0383245513195128f, 0, slot_0);
    // wrapped_backbone_temporal_stack_3_depthwise [conv1d]
    depthwise_conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_3_depthwise_weight, 1, 3, NULL, 1, 1, 0, 1, 0.0383245513195128f, wrapped_backbone_temporal_stack_3_depthwise_weight_per_channel_scales, 0.0465993843679353f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_3_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.0465993843679353f, 0, slot_2);
    // wrapped_backbone_temporal_stack_3_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 294, 0.0465993843679353f, 0, slot_0);
    // wrapped_backbone_temporal_stack_3_pointwise [conv1d]
    conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_3_pointwise_weight, 1, 1, 42, wrapped_backbone_temporal_stack_3_pointwise_bias, 1, 1, 0, 0, 0.0465993843679353f, wrapped_backbone_temporal_stack_3_pointwise_weight_per_channel_scales, 0.01857761322982668f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_3_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.01857761322982668f, 0, slot_2);
    // wrapped_backbone_temporal_stack_3_bn [batchnorm1d]
    batchnorm2d_nhwc(slot_2, 1, 7, 42, wrapped_backbone_temporal_stack_3_bn_gamma, wrapped_backbone_temporal_stack_3_bn_beta, wrapped_backbone_temporal_stack_3_bn_mean, wrapped_backbone_temporal_stack_3_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_temporal_stack_3_act [gelu]
    gelu(slot_3, 294);
    // wrapped_backbone_temporal_stack_4_depthwise_input_q [quantize]
    quantize_float_to_int8(slot_3, 294, 0.039280790043628125f, 0, slot_0);
    // wrapped_backbone_temporal_stack_4_depthwise [conv1d]
    depthwise_conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_4_depthwise_weight, 1, 3, NULL, 1, 1, 0, 1, 0.039280790043628125f, wrapped_backbone_temporal_stack_4_depthwise_weight_per_channel_scales, 0.038378009645957646f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_4_depthwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.038378009645957646f, 0, slot_2);
    // wrapped_backbone_temporal_stack_4_pointwise_input_q [quantize]
    quantize_float_to_int8(slot_2, 294, 0.038378009645957646f, 0, slot_0);
    // wrapped_backbone_temporal_stack_4_pointwise [conv1d]
    conv2d_nhwc_int8_per_channel(slot_0, 1, 7, 42, wrapped_backbone_temporal_stack_4_pointwise_weight, 1, 1, 42, wrapped_backbone_temporal_stack_4_pointwise_bias, 1, 1, 0, 0, 0.038378009645957646f, wrapped_backbone_temporal_stack_4_pointwise_weight_per_channel_scales, 0.03435989439956785f, 0, 0, 0, slot_1);
    // wrapped_backbone_temporal_stack_4_pointwise_output_dq [dequantize]
    dequantize_int8_to_float(slot_1, 294, 0.03435989439956785f, 0, slot_2);
    // wrapped_backbone_temporal_stack_4_bn [batchnorm1d]
    batchnorm2d_nhwc(slot_2, 1, 7, 42, wrapped_backbone_temporal_stack_4_bn_gamma, wrapped_backbone_temporal_stack_4_bn_beta, wrapped_backbone_temporal_stack_4_bn_mean, wrapped_backbone_temporal_stack_4_bn_var, 1e-05f, slot_3);
    // wrapped_backbone_temporal_stack_4_act [gelu]
    gelu(slot_3, 294);
    // mean [method_mean]
    /* Mean over last dimension (NCL -> NLC in C) */
    mean_hwc(slot_3, 1, 7, 42, slot_2);
    // wrapped_backbone_sample_embd_layer_0 [linear]
    dense_float_input_int8_weight_per_channel(slot_2, 42, wrapped_backbone_sample_embd_layer_0_weight, wrapped_backbone_sample_embd_layer_0_bias, 42, wrapped_backbone_sample_embd_layer_0_weight_per_channel_scales, slot_3);
    // wrapped_backbone_sample_embd_layer_1 [relu]
    relu(slot_3, 42);
    // wrapped_backbone_class_layer [linear]
    dense_float_input_int8_weight_per_channel(slot_3, 42, wrapped_backbone_class_layer_weight, wrapped_backbone_class_layer_bias, 5, wrapped_backbone_class_layer_weight_per_channel_scales, slot_2);
    memcpy(output, slot_2, 5 * sizeof(float));
}
