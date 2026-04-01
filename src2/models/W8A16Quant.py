"""
W8A16 Quantization — INT8 weights + INT16 activations.

Activations are clamped to the symmetric INT16 range [-32768, 32767]
during fake-quantization, giving 65536 discrete levels per tensor.
This preserves much more activation fidelity than W8A8 while keeping
weights compressed to INT8.

Use this module when:
  - Activation quantization noise from INT8 hurts accuracy noticeably.
  - The target hardware supports INT16 accumulators or mixed precision.

All calibration, export, and logging helpers are re-exported from
QuantBase so callers only need to import from this one file.

Typical workflow
----------------
1. Build a model with w8a16=True in the YAML / factory.
2. Call calibrate_w8a8(model, loader, device) to set activation scales.
   (The same function works for W8A16 — it detects act_bits from each layer.)
3. Train with QAT — fake-quant is active throughout.
4. Export: export_quantized_checkpoint() saves int8 weights + float32
   activation scales. Your compiler reads act_scale to requantize at runtime.
"""

from models.QuantBase import (
    # Constants
    INT8_MAX,
    INT8_MIN,
    INT16_MAX,
    INT16_MIN,
    # Primitives
    _fake_quant_symmetric,
    _per_channel_weight_scale,
    # Base layer classes
    QuantDWConv2d as _QuantDWConv2dBase,
    QuantDWConv1d as _QuantDWConv1dBase,
    # Helpers
    calibrate_w8a8,
    freeze_w8a8,
    disable_w8a8,
    has_w8a8_layers,
    export_quantized_checkpoint,
    log_w8a8_scales,
)


class QuantDWConv2d(_QuantDWConv2dBase):
    """
    W8A16 depthwise-separable 2D conv.

    Identical to QuantBase.QuantDWConv2d but act_bits is locked to 16.
    Activations are fake-quantized to INT16 [-32768, 32767].
    Weights are still fake-quantized to INT8 per output channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        dropout_ratio: float = 0.0,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            dropout_ratio=dropout_ratio, act_bits=16,
        )


class QuantDWConv1d(_QuantDWConv1dBase):
    """
    W8A16 depthwise-separable 1D conv.

    Identical to QuantBase.QuantDWConv1d but act_bits is locked to 16.
    Activations are fake-quantized to INT16 [-32768, 32767].
    Weights are still fake-quantized to INT8 per output channel.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout_ratio: float = 0.0,
    ):
        super().__init__(channels, kernel_size, dropout_ratio=dropout_ratio, act_bits=16)
