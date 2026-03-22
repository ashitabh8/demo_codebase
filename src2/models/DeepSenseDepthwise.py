# src2/models/DeepSenseDepthwise.py
"""
DeepSense backbone with depthwise-separable convolutions for CMSIS-NN deployment.

Architecture
------------
Frequency 2D depthwise-separable conv stack
  → spectrum projection (flatten freq → Linear → temporal_channels)
  → Temporal 1D depthwise-separable conv stack
  → Global average pool
  → FC head

Key differences from DeepSenseLatest.py
----------------------------------------
1. DSDWConvLayer: depthwise Conv2d + pointwise Conv2d instead of full Conv2d.
   Reduces parameters ~(in_ch / out_ch)× vs standard conv at equal channels.
2. No GRU: replaced by DSTemporalDWLayer stack over the interval dimension.
   CMSIS-NN has arm_depthwise_conv_s8 but no GRU primitive; temporal DW-sep
   conv avoids hidden state and maps directly to ARM kernels.
3. temporal_channels is independent of channels_freq[-1]: spectrum_proj maps
   from channels_freq[-1] * freq_out → temporal_channels, giving separate
   control over freq depth and temporal depth.

Classes
-------
DSDWConvLayer              : depthwise 2D + pointwise 2D + BN + GELU + Dropout2d
DSTemporalDWLayer          : depthwise 1D + pointwise 1D + BN1d + GELU + Dropout
DeepSenseDepthwiseBackbone : configurable DW freq stack → proj → DW temporal stack → FC
SingleModalDeepSenseDW     : thin wrapper unpacking freq_x[location][modality]
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DSDWConvLayer(nn.Module):
    """
    Frequency-domain 2D depthwise-separable conv layer.

    Depthwise:  Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch)
    Pointwise:  Conv2d(in_ch, out_ch, (1,1))
    Norm/Act:   BatchNorm2d(out_ch) → GELU → Dropout2d

    Padding convention (same as DSConvLayer in DeepSenseLatest.py):
        max(stride) == 1  →  "same"   on the depthwise conv
        max(stride) >  1  →  "valid"  (no padding; dims shrink)
    The pointwise 1×1 conv never needs padding.

    Args:
        in_channels:   input channels
        out_channels:  output channels (pointwise projection)
        kernel_size:   (time_k, freq_k) — time_k is almost always 1
        stride:        (time_s, freq_s)
        dropout_ratio: Dropout2d probability
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_ratio=0.0):
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
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p=dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.pointwise(self.depthwise(x)))))


class DSTemporalDWLayer(nn.Module):
    """
    Temporal 1D depthwise-separable conv layer over the interval dimension.

    Input:  [B, C, I]
    Depthwise:  Conv1d(C, C, kernel, padding=kernel//2, groups=C)  — "same" always
    Pointwise:  Conv1d(C, C, 1)
    Norm/Act:   BatchNorm1d(C) → GELU → Dropout

    Args:
        channels:      number of channels (input == output, depthwise)
        kernel_size:   temporal kernel (odd recommended; 3 or 5 typical)
        dropout_ratio: Dropout probability
    """

    def __init__(self, channels, kernel_size=3, dropout_ratio=0.0):
        super().__init__()
        padding = kernel_size // 2  # "same" for odd kernels

        self.depthwise = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size, padding=padding,
            groups=channels, bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, I]
        return self.drop(self.act(self.bn(self.pointwise(self.depthwise(x)))))


def _compute_freq_out_dw(in_spectrum_len: int, kernel_sizes, strides) -> int:
    """Compute freq dimension size after the depthwise freq conv stack."""
    freq = in_spectrum_len
    for k, s in zip(kernel_sizes, strides):
        freq_k = k[1]   # kernel along freq dimension
        freq_s = s[1]   # stride along freq dimension
        if freq_s > 1:  # "valid" padding
            freq = (freq - freq_k) // freq_s + 1
        # else "same": freq unchanged
    return freq


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class DeepSenseDepthwiseBackbone(nn.Module):
    """
    Configurable DeepSense backbone using depthwise-separable convolutions.

    Architecture
    ------------
    DW freq stack  (N = len(channels_freq) layers):
        Input:   [B, in_channels, intervals, spectrum]
        Output:  [B, channels_freq[-1], intervals, freq_out]

    Spectrum projection:
        [B, channels_freq[-1], I, freq_out]
        → reshape → [B, I, channels_freq[-1] * freq_out]
        → Linear  → [B, I, temporal_channels]
        → permute → [B, temporal_channels, I]

    DW temporal stack (num_temporal_layers layers, channels=temporal_channels):
        → [B, temporal_channels, I]

    Global avg pool:
        → [B, temporal_channels]

    sample_embd_layer (Linear + ReLU):
        → [B, fc_dim]                   ← 'features'

    class_layer (Linear):
        → [B, num_classes]              ← 'logits'

    Args
    ----
    in_channels:         input channels from data
    in_spectrum_len:     input spectrum length (used to compute freq_out)
    num_classes:         number of output classes
    channels_freq:       list of output channels per freq conv layer, e.g. [128, 256, 512, 512]
    kernel_sizes_freq:   list of (time_k, freq_k) per freq conv layer
    strides_freq:        list of (time_s, freq_s) per freq conv layer
                         len must equal len(channels_freq) == len(kernel_sizes_freq)
    temporal_channels:   channel dimension for temporal conv stack and spectrum projection output
                         (independent of channels_freq[-1])
    num_temporal_layers: number of DSTemporalDWLayer layers
    temporal_kernel:     kernel size for temporal DW conv (odd, e.g. 3)
    fc_dim:              feature embedding dimension
    dropout_ratio:       dropout probability (applied in both conv stacks)

    Returns
    -------
    dict: {'logits': [B, num_classes], 'features': [B, fc_dim]}
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
        dropout_ratio: float = 0.0,
    ):
        super().__init__()

        if not (len(channels_freq) == len(kernel_sizes_freq) == len(strides_freq)):
            raise ValueError(
                f"channels_freq, kernel_sizes_freq, strides_freq must all have equal length; "
                f"got {len(channels_freq)}, {len(kernel_sizes_freq)}, {len(strides_freq)}"
            )
        if num_temporal_layers < 1:
            raise ValueError(f"num_temporal_layers must be >= 1, got {num_temporal_layers}")

        # Frequency DW-sep conv stack
        freq_layers = []
        in_ch = in_channels
        for out_ch, k, s in zip(channels_freq, kernel_sizes_freq, strides_freq):
            freq_layers.append(DSDWConvLayer(in_ch, out_ch, k, s, dropout_ratio))
            in_ch = out_ch
        self.freq_stack = nn.ModuleList(freq_layers)

        # Spectrum projection: channels_freq[-1] * freq_out → temporal_channels
        freq_out = _compute_freq_out_dw(in_spectrum_len, kernel_sizes_freq, strides_freq)
        self.spectrum_proj = nn.Linear(channels_freq[-1] * freq_out, temporal_channels)

        # Temporal DW-sep conv stack
        temporal_layers = []
        for _ in range(num_temporal_layers):
            temporal_layers.append(
                DSTemporalDWLayer(temporal_channels, temporal_kernel, dropout_ratio)
            )
        self.temporal_stack = nn.ModuleList(temporal_layers)

        # Embedding projection
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(temporal_channels, fc_dim),
            nn.ReLU(),
        )

        # Classifier
        self.class_layer = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, in_channels, intervals, spectrum]
        Returns:
            {'logits': [B, num_classes], 'features': [B, fc_dim]}
        """
        # Freq DW-sep stack: [B, C_in, I, S] → [B, channels_freq[-1], I, freq_out]
        for layer in self.freq_stack:
            x = layer(x)

        # Spectrum projection: [B, C, I, S] → [B, I, C*S] → [B, I, T] → [B, T, I]
        B, C, I, S = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, I, C * S)
        x = self.spectrum_proj(x)          # [B, I, temporal_channels]
        x = x.permute(0, 2, 1)            # [B, temporal_channels, I]

        # Temporal DW-sep stack: [B, T, I] → [B, T, I]
        for layer in self.temporal_stack:
            x = layer(x)

        # Global avg pool over intervals: [B, T, I] → [B, T]
        x = x.mean(dim=-1)

        # Embedding + classify
        features = self.sample_embd_layer(x)
        logits = self.class_layer(features)

        return {'logits': logits, 'features': features}


# ---------------------------------------------------------------------------
# Single-modality wrapper
# ---------------------------------------------------------------------------

class SingleModalDeepSenseDW(nn.Module):
    """
    Thin wrapper around DeepSenseDepthwiseBackbone — mirrors SingleModalDeepSense.

    Unpacks freq_x[location_name][modality_name] and forwards the tensor
    to the backbone.

    Input:  freq_x[location_name][modality_name] = [B, C, intervals, spectrum]
    Output: {'logits': [B, num_classes], 'features': [B, fc_dim]}
    """

    def __init__(
        self,
        modality_name: str,
        location_name: str,
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
        dropout_ratio: float = 0.0,
    ):
        super().__init__()
        self.modality_name = modality_name
        self.location_name = location_name

        self.backbone = DeepSenseDepthwiseBackbone(
            in_channels=in_channels,
            in_spectrum_len=in_spectrum_len,
            num_classes=num_classes,
            channels_freq=channels_freq,
            kernel_sizes_freq=kernel_sizes_freq,
            strides_freq=strides_freq,
            temporal_channels=temporal_channels,
            num_temporal_layers=num_temporal_layers,
            temporal_kernel=temporal_kernel,
            fc_dim=fc_dim,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, freq_x: dict) -> dict:
        x = freq_x[self.location_name][self.modality_name]
        return self.backbone(x)
