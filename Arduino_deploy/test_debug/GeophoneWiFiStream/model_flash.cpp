/*
 * model_flash.cpp
 * ===============
 * Drop-in replacement for model.c that reads all weights from NOR flash
 * at runtime instead of embedding them in program storage.
 *
 * WHY: The original weights.h puts ~5 MB of float32 weights into program
 * flash, which overflows the Giga R1's 1.97 MB limit (264% usage).
 * Weights were already uploaded to NOR flash by flash_uploader.py, so we
 * just read them from there.
 *
 * HOW (streaming convolution):
 * Weights are stored in HWIO order [kH, kW, Cin, Cout] in flash, meaning
 * the innermost (fastest-varying) dimension is Cout. This lets us read the
 * full weight tensor exactly ONCE, sequentially, while accumulating into
 * the output buffer:
 *
 *   for kh, kw, ic:          <- sequential flash read of Cout floats
 *     for oh, ow:            <- loop over output spatial positions
 *       out[oh,ow,:] += in[oh*s+kh, ow*s+kw, ic] * wbuf[:]
 *
 * Peak extra RAM per conv: max(Cout) = 384 floats = 1.5 KB (weight slice)
 *                        + max(C)    = 384 floats = 1.5 KB (batchnorm bufs)
 * Activation slots: 2 × 50176 floats = ~392 KB (same as before, static)
 * Total stays comfortably within the 523 KB SRAM limit.
 */

#include "model_flash.h"
#include "nn_ops_float.h"
#include <string.h>
#include <math.h>

// ── Scratch buffers (static = goes into BSS, not stack) ──────────────────────

// Activation double-buffer — same sizes as the original model.c
static float slot_0[50176];   // largest: 28×28×64  = 50176
static float slot_1[50176];

// Weight slice buffer: holds one (kh,kw,ic) → all Cout weights at a time.
// Sized for the largest Cout in the model (conv4: 384).
static float wbuf[384];

// BatchNorm parameter buffers — sized for largest channel count (384).
static float bn_gamma[384];
static float bn_beta[384];
static float bn_mean[384];
static float bn_var[384];

// FC weight buffer: 384 × 10 = 3840 floats = 15 KB — small enough to read at once.
static float fc_weight_buf[3840];
static float fc_bias_buf[10];

// ── Helpers ───────────────────────────────────────────────────────────────────

static inline void fread(QSPIFBlockDevice& flash, uint32_t addr, float* buf, int n_floats) {
    flash.read(buf, addr, (size_t)n_floats * sizeof(float));
}

// ── Streaming conv2d ──────────────────────────────────────────────────────────
/*
 * Reads weight tensor from flash exactly once in HWIO sequential order.
 * Bias is read first (small, fits in wbuf).
 * Output is initialised to bias, then weights are streamed and accumulated.
 */
static void conv2d_flash(
    QSPIFBlockDevice& flash,
    const float* in,    int in_h,  int in_w,  int in_c,
    uint32_t filt_addr, int k_h,   int k_w,   int out_c,
    uint32_t bias_addr,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    // ── 1. Initialise output to bias ─────────────────────────────────────────
    // Bias is small (≤ 384 floats), reuse wbuf temporarily.
    fread(flash, bias_addr, wbuf, out_c);
    for (int oh = 0; oh < out_h; oh++)
        for (int ow = 0; ow < out_w; ow++)
            for (int oc = 0; oc < out_c; oc++)
                out[(oh * out_w + ow) * out_c + oc] = wbuf[oc];

    // ── 2. Stream weights and accumulate ────────────────────────────────────
    // HWIO layout: outermost = kh, then kw, then ic, innermost = oc.
    // Each read fetches out_c floats (one full "oc strip") sequentially.
    uint32_t addr = filt_addr;

    for (int kh = 0; kh < k_h; kh++) {
        for (int kw = 0; kw < k_w; kw++) {
            for (int ic = 0; ic < in_c; ic++) {
                // Read the strip filt[kh, kw, ic, :] — out_c floats, contiguous
                fread(flash, addr, wbuf, out_c);
                addr += (uint32_t)out_c * sizeof(float);

                // Accumulate into every valid output pixel
                for (int oh = 0; oh < out_h; oh++) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;

                    for (int ow = 0; ow < out_w; ow++) {
                        int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;

                        float in_val = in[(ih * in_w + iw) * in_c + ic];
                        float* out_px = out + (oh * out_w + ow) * out_c;

                        for (int oc = 0; oc < out_c; oc++)
                            out_px[oc] += in_val * wbuf[oc];
                    }
                }
            }
        }
    }
}

// ── BatchNorm (reads parameters from flash) ───────────────────────────────────
static void batchnorm_flash(
    QSPIFBlockDevice& flash,
    float* inout, int h, int w, int c,
    uint32_t gamma_addr, uint32_t beta_addr,
    uint32_t mean_addr,  uint32_t var_addr,
    float eps)
{
    fread(flash, gamma_addr, bn_gamma, c);
    fread(flash, beta_addr,  bn_beta,  c);
    fread(flash, mean_addr,  bn_mean,  c);
    fread(flash, var_addr,   bn_var,   c);

    for (int ih = 0; ih < h; ih++) {
        for (int iw = 0; iw < w; iw++) {
            float* px = inout + (ih * w + iw) * c;
            for (int ch = 0; ch < c; ch++) {
                float norm = (px[ch] - bn_mean[ch]) / sqrtf(bn_var[ch] + eps);
                px[ch] = bn_gamma[ch] * norm + bn_beta[ch];
            }
        }
    }
}

// ── Dense (reads weights from flash) ─────────────────────────────────────────
static void dense_flash(
    QSPIFBlockDevice& flash,
    const float* x,    int in_features,
    uint32_t W_addr,   uint32_t b_addr,
    int out_features,
    float* y)
{
    // FC is small enough to read entirely at once
    fread(flash, W_addr, fc_weight_buf, in_features * out_features);
    fread(flash, b_addr, fc_bias_buf,   out_features);
    dense(x, in_features, fc_weight_buf, fc_bias_buf, out_features, y);
}

// ── Public entry point ────────────────────────────────────────────────────────
void model_forward_flash(const float* input, float* output, QSPIFBlockDevice& flash) {

    // conv1: 28×28×1 → 28×28×64
    conv2d_flash(flash, input, 28, 28, 1,
                 CONV1_WEIGHT_ADDR, 3, 3, 64, CONV1_BIAS_ADDR,
                 1, 1, 1, 1, slot_0);
    batchnorm_flash(flash, slot_0, 28, 28, 64,
                    BN1_GAMMA_ADDR, BN1_BETA_ADDR, BN1_MEAN_ADDR, BN1_VAR_ADDR, 1e-5f);
    relu(slot_0, 50176);

    // conv2: 28×28×64 → 14×14×128  (stride 2)
    conv2d_flash(flash, slot_0, 28, 28, 64,
                 CONV2_WEIGHT_ADDR, 3, 3, 128, CONV2_BIAS_ADDR,
                 2, 2, 1, 1, slot_1);
    batchnorm_flash(flash, slot_1, 14, 14, 128,
                    BN2_GAMMA_ADDR, BN2_BETA_ADDR, BN2_MEAN_ADDR, BN2_VAR_ADDR, 1e-5f);
    relu(slot_1, 25088);

    // conv3: 14×14×128 → 7×7×256  (stride 2)
    conv2d_flash(flash, slot_1, 14, 14, 128,
                 CONV3_WEIGHT_ADDR, 3, 3, 256, CONV3_BIAS_ADDR,
                 2, 2, 1, 1, slot_0);
    batchnorm_flash(flash, slot_0, 7, 7, 256,
                    BN3_GAMMA_ADDR, BN3_BETA_ADDR, BN3_MEAN_ADDR, BN3_VAR_ADDR, 1e-5f);
    relu(slot_0, 12544);

    // conv4: 7×7×256 → 4×4×384  (stride 2)
    conv2d_flash(flash, slot_0, 7, 7, 256,
                 CONV4_WEIGHT_ADDR, 3, 3, 384, CONV4_BIAS_ADDR,
                 2, 2, 1, 1, slot_1);
    batchnorm_flash(flash, slot_1, 4, 4, 384,
                    BN4_GAMMA_ADDR, BN4_BETA_ADDR, BN4_MEAN_ADDR, BN4_VAR_ADDR, 1e-5f);
    relu(slot_1, 6144);

    // global average pool: 4×4×384 → 384
    mean_hwc(slot_1, 4, 4, 384, slot_0);

    // fc: 384 → 10
    dense_flash(flash, slot_0, 384, FC_WEIGHT_ADDR, FC_BIAS_ADDR, 10, slot_1);

    memcpy(output, slot_1, 10 * sizeof(float));
}