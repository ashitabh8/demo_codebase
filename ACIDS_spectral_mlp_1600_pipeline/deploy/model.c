// Auto-generated model implementation
// DO NOT EDIT

// Input shape  (NCHW): [1, 83]
// Output shape: [1, 10]

#include "model.h"
#include "weights.h"
#include "nn_ops_float.h"
#include "nn_ops_int8.h"

#include <string.h>

void model_forward(const float* input, float* output) {
    static float slot_0[128];
    static float slot_1[128];

    // net_0 [linear]
    dense_float_input_int8_weight_per_channel(input, 83, net_0_weight, net_0_bias, 128, net_0_weight_per_channel_scales, slot_0);
    // net_1 [relu]
    relu(slot_0, 128);
    // net_3 [linear]
    dense_float_input_int8_weight_per_channel(slot_0, 128, net_3_weight, net_3_bias, 128, net_3_weight_per_channel_scales, slot_1);
    // net_4 [relu]
    relu(slot_1, 128);
    // net_6 [linear]
    dense_float_input_int8_weight_per_channel(slot_1, 128, net_6_weight, net_6_bias, 10, net_6_weight_per_channel_scales, slot_0);
    memcpy(output, slot_0, 10 * sizeof(float));
}
