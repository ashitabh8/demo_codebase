// Auto-generated model implementation
// DO NOT EDIT

// Input shape  (NCHW): [1, 6, 7, 256]
// Output shape: [1, 10]

#include "model.h"
#include "weights.h"
#include "nn_ops_float.h"

#include <string.h>

void model_forward(const float* input, float* output) {
    static float slot_0[7168];
    static float slot_1[7168];

    // freq_stack_0_depthwise [conv2d]
    depthwise_conv2d_nhwc(input, 7, 256, 6, freq_stack_0_depthwise_weight, 1, 8, NULL, 1, 8, 0, 0, slot_0);
    // freq_stack_0_pointwise [conv2d]
    conv2d_nhwc(slot_0, 7, 32, 6, freq_stack_0_pointwise_weight, 1, 1, 32, freq_stack_0_pointwise_bias, 1, 1, 0, 0, slot_1);
    // freq_stack_0_bn [batchnorm]
    batchnorm2d_nhwc(slot_1, 7, 32, 32, freq_stack_0_bn_gamma, freq_stack_0_bn_beta, freq_stack_0_bn_mean, freq_stack_0_bn_var, 1e-05f, slot_0);
    // freq_stack_0_act [relu]
    relu(slot_0, 7168);
    // freq_stack_1_depthwise [conv2d]
    depthwise_conv2d_nhwc(slot_0, 7, 32, 32, freq_stack_1_depthwise_weight, 1, 5, NULL, 1, 4, 0, 0, slot_1);
    // freq_stack_1_pointwise [conv2d]
    conv2d_nhwc(slot_1, 7, 7, 32, freq_stack_1_pointwise_weight, 1, 1, 64, freq_stack_1_pointwise_bias, 1, 1, 0, 0, slot_0);
    // freq_stack_1_bn [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 7, 64, freq_stack_1_bn_gamma, freq_stack_1_bn_beta, freq_stack_1_bn_mean, freq_stack_1_bn_var, 1e-05f, slot_1);
    // freq_stack_1_act [relu]
    relu(slot_1, 3136);
    // freq_stack_2_depthwise [conv2d]
    depthwise_conv2d_nhwc(slot_1, 7, 7, 64, freq_stack_2_depthwise_weight, 1, 3, NULL, 1, 2, 0, 0, slot_0);
    // freq_stack_2_pointwise [conv2d]
    conv2d_nhwc(slot_0, 7, 3, 64, freq_stack_2_pointwise_weight, 1, 1, 96, freq_stack_2_pointwise_bias, 1, 1, 0, 0, slot_1);
    // freq_stack_2_bn [batchnorm]
    batchnorm2d_nhwc(slot_1, 7, 3, 96, freq_stack_2_bn_gamma, freq_stack_2_bn_beta, freq_stack_2_bn_mean, freq_stack_2_bn_var, 1e-05f, slot_0);
    // freq_stack_2_act [relu]
    relu(slot_0, 2016);
    // getattr_1 [method_getattr]
    // getitem [method_getitem]
    // getitem_1 [method_getitem]
    // getitem_2 [method_getitem]
    // getitem_3 [method_getitem]
    // permute [method_permute]
    /* permute(0,2,1,3): source NHWC [H,W,C] -> [H,C,W] */
    for (int hh = 0; hh < 7; ++hh) {
        for (int cc = 0; cc < 96; ++cc) {
            for (int ww = 0; ww < 3; ++ww) {
                slot_1[((hh * 96 + cc) * 3) + ww] = slot_0[((hh * 3 + ww) * 96) + cc];
            }
        }
    }
    // mul [mul]
    // reshape [method_reshape]
    memcpy(slot_0, slot_1, 2016 * sizeof(float));
    // spectrum_proj [linear]
    for (int r = 0; r < 7; ++r) {
        dense(slot_0 + r * 288, 288, spectrum_proj_weight, spectrum_proj_bias, 64, slot_1 + r * 64);
    }
    // permute_1 [method_permute]
    permute_4d(slot_1, 7, 64, 1, 1, 1, 0, 2, 3, slot_0);
    // unsqueeze [method_unsqueeze]
    /* unsqueeze(-1): [C, I] -> NHWC [I, 1, C] */
    for (int ii = 0; ii < 7; ++ii) {
        for (int cc = 0; cc < 64; ++cc) {
            slot_1[(ii * 64) + cc] = slot_0[(cc * 7) + ii];
        }
    }
    // temporal_stack_0_depthwise [conv2d]
    depthwise_conv2d_nhwc(slot_1, 7, 1, 64, temporal_stack_0_depthwise_weight, 3, 1, NULL, 1, 1, 1, 0, slot_0);
    // temporal_stack_0_pointwise [conv2d]
    conv2d_nhwc(slot_0, 7, 1, 64, temporal_stack_0_pointwise_weight, 1, 1, 64, temporal_stack_0_pointwise_bias, 1, 1, 0, 0, slot_1);
    // temporal_stack_0_bn [batchnorm]
    batchnorm2d_nhwc(slot_1, 7, 1, 64, temporal_stack_0_bn_gamma, temporal_stack_0_bn_beta, temporal_stack_0_bn_mean, temporal_stack_0_bn_var, 1e-05f, slot_0);
    // temporal_stack_0_act [relu]
    relu(slot_0, 448);
    // squeeze [method_squeeze]
    /* squeeze(-1): NHWC [I, 1, C] -> [C, I] */
    for (int cc = 0; cc < 64; ++cc) {
        for (int ii = 0; ii < 7; ++ii) {
            slot_1[(cc * 7) + ii] = slot_0[(ii * 64) + cc];
        }
    }
    // unsqueeze_1 [method_unsqueeze]
    /* unsqueeze(-1): [C, I] -> NHWC [I, 1, C] */
    for (int ii = 0; ii < 7; ++ii) {
        for (int cc = 0; cc < 64; ++cc) {
            slot_0[(ii * 64) + cc] = slot_1[(cc * 7) + ii];
        }
    }
    // temporal_stack_1_depthwise [conv2d]
    depthwise_conv2d_nhwc(slot_0, 7, 1, 64, temporal_stack_1_depthwise_weight, 3, 1, NULL, 1, 1, 1, 0, slot_1);
    // temporal_stack_1_pointwise [conv2d]
    conv2d_nhwc(slot_1, 7, 1, 64, temporal_stack_1_pointwise_weight, 1, 1, 64, temporal_stack_1_pointwise_bias, 1, 1, 0, 0, slot_0);
    // temporal_stack_1_bn [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 1, 64, temporal_stack_1_bn_gamma, temporal_stack_1_bn_beta, temporal_stack_1_bn_mean, temporal_stack_1_bn_var, 1e-05f, slot_1);
    // temporal_stack_1_act [relu]
    relu(slot_1, 448);
    // squeeze_1 [method_squeeze]
    /* squeeze(-1): NHWC [I, 1, C] -> [C, I] */
    for (int cc = 0; cc < 64; ++cc) {
        for (int ii = 0; ii < 7; ++ii) {
            slot_0[(cc * 7) + ii] = slot_1[(ii * 64) + cc];
        }
    }
    // unsqueeze_2 [method_unsqueeze]
    /* unsqueeze(-1): [C, I] -> NHWC [I, 1, C] */
    for (int ii = 0; ii < 7; ++ii) {
        for (int cc = 0; cc < 64; ++cc) {
            slot_1[(ii * 64) + cc] = slot_0[(cc * 7) + ii];
        }
    }
    // temporal_stack_2_depthwise [conv2d]
    depthwise_conv2d_nhwc(slot_1, 7, 1, 64, temporal_stack_2_depthwise_weight, 3, 1, NULL, 1, 1, 1, 0, slot_0);
    // temporal_stack_2_pointwise [conv2d]
    conv2d_nhwc(slot_0, 7, 1, 64, temporal_stack_2_pointwise_weight, 1, 1, 64, temporal_stack_2_pointwise_bias, 1, 1, 0, 0, slot_1);
    // temporal_stack_2_bn [batchnorm]
    batchnorm2d_nhwc(slot_1, 7, 1, 64, temporal_stack_2_bn_gamma, temporal_stack_2_bn_beta, temporal_stack_2_bn_mean, temporal_stack_2_bn_var, 1e-05f, slot_0);
    // temporal_stack_2_act [relu]
    relu(slot_0, 448);
    // squeeze_2 [method_squeeze]
    /* squeeze(-1): NHWC [I, 1, C] -> [C, I] */
    for (int cc = 0; cc < 64; ++cc) {
        for (int ii = 0; ii < 7; ++ii) {
            slot_1[(cc * 7) + ii] = slot_0[(ii * 64) + cc];
        }
    }
    // unsqueeze_3 [method_unsqueeze]
    /* unsqueeze(-1): [C, I] -> NHWC [I, 1, C] */
    for (int ii = 0; ii < 7; ++ii) {
        for (int cc = 0; cc < 64; ++cc) {
            slot_0[(ii * 64) + cc] = slot_1[(cc * 7) + ii];
        }
    }
    // temporal_stack_3_depthwise [conv2d]
    depthwise_conv2d_nhwc(slot_0, 7, 1, 64, temporal_stack_3_depthwise_weight, 3, 1, NULL, 1, 1, 1, 0, slot_1);
    // temporal_stack_3_pointwise [conv2d]
    conv2d_nhwc(slot_1, 7, 1, 64, temporal_stack_3_pointwise_weight, 1, 1, 64, temporal_stack_3_pointwise_bias, 1, 1, 0, 0, slot_0);
    // temporal_stack_3_bn [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 1, 64, temporal_stack_3_bn_gamma, temporal_stack_3_bn_beta, temporal_stack_3_bn_mean, temporal_stack_3_bn_var, 1e-05f, slot_1);
    // temporal_stack_3_act [relu]
    relu(slot_1, 448);
    // squeeze_3 [method_squeeze]
    /* squeeze(-1): NHWC [I, 1, C] -> [C, I] */
    for (int cc = 0; cc < 64; ++cc) {
        for (int ii = 0; ii < 7; ++ii) {
            slot_0[(cc * 7) + ii] = slot_1[(ii * 64) + cc];
        }
    }
    // mean [method_mean]
    /* Mean over last dimension */
    mean_last_dim(slot_0, 64, 7, slot_1);
    // fc1 [linear]
    dense(slot_1, 64, fc1_weight, fc1_bias, 64, slot_0);
    // fc1_relu [relu]
    relu(slot_0, 64);
    // fc2 [linear]
    dense(slot_0, 64, fc2_weight, fc2_bias, 10, slot_1);
    memcpy(output, slot_1, 10 * sizeof(float));
}
