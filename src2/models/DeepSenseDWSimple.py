# src2/models/DeepSenseDWSimple.py
"""
Compiler-friendly DeepSense backbone using depthwise-separable convolutions.

Stripped to ops supported by the Tiny-NN-in-C source-to-source compiler:
    Conv2d, Linear, BatchNorm2d, ReLU, AdaptiveAvgPool2d / mean, Reshape.

Differences from DeepSenseDepthwise.py
---------------------------------------
1. GELU → ReLU everywhere.
2. Dropout / Dropout2d removed.
3. Temporal layers use Conv2d with a (kernel, 1) kernel instead of Conv1d,
   so that only Conv2d and BatchNorm2d appear in the compiled graph.
4. No BiGRU, pretrain mode, projection head, or output_dims MLP.
5. Backbone forward: Tensor in → Tensor out (no dict, single execution path).

Classes
-------
DSDWConvLayerSimple     : depthwise Conv2d + pointwise Conv2d + BN2d + ReLU
DSTemporalDWLayerSimple : depthwise Conv2d (k,1) + pointwise Conv2d + BN2d + ReLU
DeepSenseDWSimpleBackbone : freq stack → spectrum proj → temporal stack → FC head
SingleModalDeepSenseDWSimple : thin dict-unpacking wrapper (mirrors SingleModalSimpleResNet)
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_freq_out_dw(in_spectrum_len: int, kernel_sizes, strides) -> int:
    """Compute freq dimension size after the depthwise freq conv stack."""
    freq = in_spectrum_len
    for k, s in zip(kernel_sizes, strides):
        freq_k = k[1]
        freq_s = s[1]
        if freq_s > 1:  # "valid" padding
            freq = (freq - freq_k) // freq_s + 1
        # else "same": freq unchanged
    return freq


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DSDWConvLayerSimple(nn.Module):
    """
    Frequency-domain 2D depthwise-separable conv layer (compiler-friendly).

    Depthwise:  Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch)
    Pointwise:  Conv2d(in_ch, out_ch, (1,1))
    Norm/Act:   BatchNorm2d(out_ch) → ReLU

    Padding: max(stride)==1 → "same"; max(stride)>1 → "valid".

    Args:
        in_channels:  input channels
        out_channels: output channels (pointwise projection)
        kernel_size:  (time_k, freq_k)
        stride:       (time_s, freq_s)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        kernel_size = tuple(kernel_size)
        stride = tuple(stride)
        padding = "same" if max(stride) == 1 else "valid"

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, padding_mode="zeros",
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 1), stride=1,
            padding=0, bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class DSTemporalDWLayerSimple(nn.Module):
    """
    Temporal depthwise-separable conv layer using Conv2d with width=1.

    Input [B, C, I] is unsqueezed to [B, C, I, 1], processed with a
    (kernel_size, 1) depthwise Conv2d (same padding along I), then squeezed
    back. This avoids Conv1d and BatchNorm1d, which are not in the compiler's
    supported op set.

    Input:  [B, C, I]
    Output: [B, C, I]

    Args:
        channels:    number of channels (input == output)
        kernel_size: temporal kernel (odd; 3 or 5 typical)
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=channels, bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=True)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, I]
        x = x.unsqueeze(-1)                            # [B, C, I, 1]
        x = self.act(self.bn(self.pointwise(self.depthwise(x))))  # [B, C, I, 1]
        return x.squeeze(-1)                            # [B, C, I]


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class DeepSenseDWSimpleBackbone(nn.Module):
    """
    Compiler-friendly DeepSense depthwise-separable backbone.

    Architecture
    ------------
    DW freq stack  (len(channels_freq) layers of DSDWConvLayerSimple):
        [B, in_channels, I, spectrum] → [B, channels_freq[-1], I, freq_out]

    Spectrum projection:
        permute + reshape → [B, I, channels_freq[-1] * freq_out]
        Linear            → [B, I, temporal_channels]
        permute           → [B, temporal_channels, I]

    DW temporal stack (num_temporal_layers × DSTemporalDWLayerSimple):
        [B, temporal_channels, I] → [B, temporal_channels, I]

    Global mean pool:
        mean(dim=-1) → [B, temporal_channels]

    FC head:
        Linear(temporal_channels, fc_dim) → ReLU → Linear(fc_dim, num_classes)
        → [B, num_classes]

    Args
    ----
    in_channels:         input channels
    in_spectrum_len:     input spectrum length (used to compute freq_out)
    num_classes:         number of output classes
    channels_freq:       list of output channels per freq conv layer
    kernel_sizes_freq:   list of (time_k, freq_k) per freq conv layer
    strides_freq:        list of (time_s, freq_s) per freq conv layer
    temporal_channels:   channel dim for temporal conv stack and spectrum projection
    num_temporal_layers: number of DSTemporalDWLayerSimple layers
    temporal_kernel:     kernel size for temporal DW conv (odd)
    fc_dim:              feature embedding dimension

    Returns
    -------
    torch.Tensor: logits [B, num_classes]
    """

    def __init__(
        self,
        in_channels: int,
        in_spectrum_len: int,
        num_classes: int,
        channels_freq,
        kernel_sizes_freq,
        strides_freq,
        temporal_channels: int,
        num_temporal_layers: int,
        temporal_kernel: int,
        fc_dim: int,
    ):
        super().__init__()

        if not (len(channels_freq) == len(kernel_sizes_freq) == len(strides_freq)):
            raise ValueError(
                f"channels_freq, kernel_sizes_freq, strides_freq must have equal length; "
                f"got {len(channels_freq)}, {len(kernel_sizes_freq)}, {len(strides_freq)}"
            )
        if num_temporal_layers < 1:
            raise ValueError(f"num_temporal_layers must be >= 1, got {num_temporal_layers}")

        # Frequency DW-sep conv stack
        freq_layers = []
        in_ch = in_channels
        for out_ch, k, s in zip(channels_freq, kernel_sizes_freq, strides_freq):
            freq_layers.append(DSDWConvLayerSimple(in_ch, out_ch, k, s))
            in_ch = out_ch
        self.freq_stack = nn.ModuleList(freq_layers)

        # Spectrum projection: channels_freq[-1] * freq_out → temporal_channels
        freq_out = _compute_freq_out_dw(in_spectrum_len, kernel_sizes_freq, strides_freq)
        self.spectrum_proj = nn.Linear(channels_freq[-1] * freq_out, temporal_channels)

        # Temporal DW-sep conv stack (Conv2d-based)
        self.temporal_stack = nn.ModuleList([
            DSTemporalDWLayerSimple(temporal_channels, temporal_kernel)
            for _ in range(num_temporal_layers)
        ])

        # FC head
        self.fc1 = nn.Linear(temporal_channels, fc_dim)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, intervals, spectrum]
        Returns:
            logits: [B, num_classes]
        """
        # Freq DW-sep stack
        for layer in self.freq_stack:
            x = layer(x)
        # x: [B, channels_freq[-1], I, freq_out]

        # Spectrum projection
        B, C, I, S = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, I, C * S)  # [B, I, C*freq_out]
        x = self.spectrum_proj(x)                         # [B, I, temporal_channels]
        x = x.permute(0, 2, 1)                           # [B, temporal_channels, I]

        # Temporal DW-sep stack
        for layer in self.temporal_stack:
            x = layer(x)
        # x: [B, temporal_channels, I]

        # Global mean pool + head
        x = x.mean(dim=-1)       # [B, temporal_channels]
        x = self.fc1_relu(self.fc1(x))  # [B, fc_dim]
        x = self.fc2(x)          # [B, num_classes]
        return x


# ---------------------------------------------------------------------------
# Single-modality wrapper
# ---------------------------------------------------------------------------

class SingleModalDeepSenseDWSimple(nn.Module):
    """
    Thin wrapper around DeepSenseDWSimpleBackbone — mirrors SingleModalSimpleResNet.

    Unpacks inputs[location_name][modality_name] and forwards the tensor
    to the backbone. The compiler only sees the backbone.

    Input:  inputs[location_name][modality_name] = [B, C, intervals, spectrum]
    Output: [B, num_classes]
    """

    def __init__(self, location_name: str, modality_name: str, backbone: nn.Module):
        super().__init__()
        self.location_name = location_name
        self.modality_name = modality_name
        self.backbone = backbone

    def forward(self, inputs: dict) -> torch.Tensor:
        x = inputs[self.location_name][self.modality_name]
        return self.backbone(x)
