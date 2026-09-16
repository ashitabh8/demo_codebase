#include "spectral_scaler.h"

void spectral_apply_standard_scaler(
    const float *raw,
    float *scaled,
    int dim
)
{
    for (int i = 0; i < dim; ++i) {
        scaled[i] = (raw[i] - SPECTRAL_SCALER_MEAN[i]) / SPECTRAL_SCALER_SCALE[i];
    }
}
