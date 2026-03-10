"""
Weight-only Quantization-Aware Training (QAT) modules.

Simple symmetric uniform fake quantization with STE (straight-through estimator)
for 2-bit and 4-bit weight-only QAT. Activations stay full precision.

Design for compiler compatibility:
- WeightOnlyConv2d subclasses nn.Conv2d and WeightOnlyLinear subclasses nn.Linear
  so the Tiny-NN-in-C compiler (isinstance(module, nn.Conv2d/nn.Linear)) recognizes
  them and lowers to the same C ops (conv2d_nhwc, dense). State dict keys and
  parameter names match standard Conv2d/Linear (weight, bias).
- Whenever nbit < 32 we fake-quantize in forward (train and eval), so validation
  accuracy is true QAT (quantized-weight) accuracy. The compiler still exports
  using module.weight (the stored float parameter) when lowering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quant_weight_symmetric(weight: torch.Tensor, nbit: int) -> torch.Tensor:
    """
    Symmetric uniform fake quantization for weights (STE).
    Quantization range: [-2^(nbit-1), 2^(nbit-1) - 1] for signed.
    Scale is per-tensor (max abs of weight).
    """
    if nbit >= 32:
        return weight
    qmin = -(2 ** (nbit - 1))
    qmax = (2 ** (nbit - 1)) - 1
    scale = weight.abs().max().clamp(min=1e-8)
    w = weight / scale
    w = torch.clamp(torch.round(w * qmax) / qmax, -1.0, 1.0)
    w_quant = w * scale
    return weight + (w_quant - weight).detach()


class WeightOnlyConv2d(nn.Conv2d):
    """
    Conv2d with weight-only fake quantization (symmetric, STE).
    Subclasses nn.Conv2d so the compiler recognizes it and lowers to conv2d_nhwc.
    Forward always uses fake-quantized weights when nbit < 32 so validation reports QAT accuracy.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        nbit: int = 4,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.nbit = int(nbit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nbit < 32:
            w = fake_quant_weight_symmetric(self.weight, self.nbit)
            return F.conv2d(
                x, w, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            )
        return super().forward(x)


class WeightOnlyLinear(nn.Linear):
    """
    Linear with weight-only fake quantization (symmetric, STE).
    Subclasses nn.Linear so the compiler recognizes it and lowers to dense.
    Forward always uses fake-quantized weights when nbit < 32 so validation reports QAT accuracy.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nbit: int = 4,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.nbit = int(nbit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nbit < 32:
            w = fake_quant_weight_symmetric(self.weight, self.nbit)
            return F.linear(x, w, self.bias)
        return super().forward(x)
