#ifndef MODEL_FLASH_H_
#define MODEL_FLASH_H_

#include "QSPIFBlockDevice.h"

// Flash addresses from flash_uploader.py memory map output
#define CONV1_WEIGHT_ADDR  0x00010000UL
#define CONV1_BIAS_ADDR    0x00010900UL
#define BN1_GAMMA_ADDR     0x00010A00UL
#define BN1_BETA_ADDR      0x00010B00UL
#define BN1_MEAN_ADDR      0x00010C00UL
#define BN1_VAR_ADDR       0x00010D00UL

#define CONV2_WEIGHT_ADDR  0x00010E00UL
#define CONV2_BIAS_ADDR    0x00058E00UL
#define BN2_GAMMA_ADDR     0x00059000UL
#define BN2_BETA_ADDR      0x00059200UL
#define BN2_MEAN_ADDR      0x00059400UL
#define BN2_VAR_ADDR       0x00059600UL

#define CONV3_WEIGHT_ADDR  0x00059800UL
#define CONV3_BIAS_ADDR    0x00179800UL
#define BN3_GAMMA_ADDR     0x00179C00UL
#define BN3_BETA_ADDR      0x0017A000UL
#define BN3_MEAN_ADDR      0x0017A400UL
#define BN3_VAR_ADDR       0x0017A800UL

#define CONV4_WEIGHT_ADDR  0x0017AC00UL
#define CONV4_BIAS_ADDR    0x004DAC00UL
#define BN4_GAMMA_ADDR     0x004DB200UL
#define BN4_BETA_ADDR      0x004DB800UL
#define BN4_MEAN_ADDR      0x004DBE00UL
#define BN4_VAR_ADDR       0x004DC400UL

#define FC_WEIGHT_ADDR     0x004DCA00UL
#define FC_BIAS_ADDR       0x004E0600UL

// Same signature as model_forward() — drop-in replacement.
// flash: an already-initialised QSPIFBlockDevice.
void model_forward_flash(const float* input, float* output, QSPIFBlockDevice& flash);

#endif // MODEL_FLASH_H_