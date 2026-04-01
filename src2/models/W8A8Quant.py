"""
W8A8 Quantization — INT8 weights + INT8 activations.

Activations are clamped to the symmetric INT8 range [-128, 127] during
fake-quantization, giving 256 discrete levels per tensor.

Use this module when targeting hardware that natively supports INT8 MAC
operations (e.g. CMSIS-NN, TensorFlow Lite INT8, Arduino Nano 33 BLE).

All calibration, export, and logging helpers are re-exported from
QuantBase so callers only need to import from this one file.

Typical workflow
----------------
1. Build a model with w8a8=True in the YAML / factory.
2. Call calibrate_w8a8(model, loader, device) to set activation scales.
3. Train with QAT — fake-quant is active throughout.
4. Export: export_quantized_checkpoint() saves int8 weights + scales.
"""

from models.QuantBase import (
    # Constants
    INT8_MAX,
    INT8_MIN,
    INT16_MAX,
    INT16_MIN,
    # Primitives (re-exported for callers that need them directly)
    _fake_quant_symmetric,
    _per_channel_weight_scale,
    # Base layer classes (used internally as parents)
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
    W8A8 depthwise-separable 2D conv.

    Identical to QuantBase.QuantDWConv2d but act_bits is locked to 8.
    Activations are fake-quantized to INT8 [-128, 127].
    Weights are fake-quantized to INT8 per output channel.
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
            dropout_ratio=dropout_ratio, act_bits=8,
        )


class QuantDWConv1d(_QuantDWConv1dBase):
    """
    W8A8 depthwise-separable 1D conv.

    Identical to QuantBase.QuantDWConv1d but act_bits is locked to 8.
    Activations are fake-quantized to INT8 [-128, 127].
    Weights are fake-quantized to INT8 per output channel.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout_ratio: float = 0.0,
    ):
        super().__init__(channels, kernel_size, dropout_ratio=dropout_ratio, act_bits=8)
