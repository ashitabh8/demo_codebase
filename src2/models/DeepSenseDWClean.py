# src2/models/DeepSenseDWClean.py
"""
DeepSense backbone with depthwise-separable convolutions — simplified variant.

Stripped version of DeepSenseDepthwise.py:
  - No w8a8 / w8a16 quantization branches
  - GELU → ReLU activations
  - No BiGRU / recurrent support
  - Conv1d temporal stack (kept)
  - Dropout support (kept)
  - pretrain_mode + projection_head (kept)
  - output_dims MLP path (kept)

Architecture
------------
Frequency 2D depthwise-separable conv stack
  → spectrum projection (flatten freq → Linear → temporal_channels)
  → Temporal 1D depthwise-separable conv stack
  → Global average pool
  → FC head

Classes
-------
DSDWConvLayerClean         : depthwise Conv2d + pointwise Conv2d + BN + ReLU + Dropout2d
DSTemporalDWLayerClean     : depthwise Conv1d + pointwise Conv1d + BN1d + ReLU + Dropout
DeepSenseDWCleanBackbone   : configurable DW freq stack → proj → DW temporal stack → FC
SingleModalDeepSenseDWClean: thin wrapper unpacking freq_x[location][modality]
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DSDWConvLayerClean(nn.Module):
    """
    Frequency-domain 2D depthwise-separable conv layer.

    Depthwise:  Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch)
    Pointwise:  Conv2d(in_ch, out_ch, (1,1))
    Norm/Act:   BatchNorm2d(out_ch) → ReLU → Dropout2d

    Padding convention:
        max(stride) == 1  →  "same"   on the depthwise conv
        max(stride) >  1  →  "valid"  (no padding; dims shrink)

    Args:
        in_channels:   input channels
        out_channels:  output channels (pointwise projection)
        kernel_size:   (time_k, freq_k)
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
        self.act = nn.ReLU()
        self.drop = nn.Dropout2d(p=dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.pointwise(self.depthwise(x)))))


class DSTemporalDWLayerClean(nn.Module):
    """
    Temporal 1D depthwise-separable conv layer over the interval dimension.

    Input:  [B, C, I]
    Depthwise:  Conv1d(C, C, kernel, padding=kernel//2, groups=C)  — "same" always
    Pointwise:  Conv1d(C, C, 1)
    Norm/Act:   BatchNorm1d(C) → ReLU → Dropout

    Args:
        channels:      number of channels (input == output, depthwise)
        kernel_size:   temporal kernel (odd recommended; 3 or 5 typical)
        dropout_ratio: Dropout probability
    """

    def __init__(self, channels, kernel_size=3, dropout_ratio=0.0):
        super().__init__()
        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size, padding=padding,
            groups=channels, bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, I]
        return self.drop(self.act(self.bn(self.pointwise(self.depthwise(x)))))


def _compute_freq_out_dw(in_spectrum_len: int, kernel_sizes, strides) -> int:
    """Compute freq dimension size after the depthwise freq conv stack."""
    freq = in_spectrum_len
    for k, s in zip(kernel_sizes, strides):
        freq_k = k[1]
        freq_s = s[1]
        if freq_s > 1:  # "valid" padding
            freq = (freq - freq_k) // freq_s + 1
    return freq


def _build_output_dims_mlp(
    in_dim: int,
    output_dims,
    dropout_ratio: float,
) -> nn.Sequential:
    """
    Stack of Linear layers with ReLU between (no ReLU after last layer).
    output_dims length = number of linear layers; final dim is embedding size.
    """
    if not output_dims:
        raise ValueError("output_dims must be a non-empty list of positive ints")
    parts = []
    d_in = in_dim
    n = len(output_dims)
    for i, d_out in enumerate(output_dims):
        if d_out < 1:
            raise ValueError(f"output_dims entries must be >= 1, got {d_out}")
        parts.append(nn.Linear(d_in, d_out))
        if i < n - 1:
            parts.append(nn.ReLU())
            if dropout_ratio > 0:
                parts.append(nn.Dropout(p=dropout_ratio))
        d_in = d_out
    return nn.Sequential(*parts)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class DeepSenseDWCleanBackbone(nn.Module):
    """
    Simplified DeepSense backbone using depthwise-separable convolutions.

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

    Global average pool over intervals:
        → [B, temporal_channels]

    Head: either output_dims MLP or sample_embd_layer
    (Linear + ReLU) to fc_dim  → 'features'

    class_layer (Linear):
        → [B, num_classes]  ← 'logits'

    Args
    ----
    in_channels:         input channels from data
    in_spectrum_len:     input spectrum length (used to compute freq_out)
    num_classes:         number of output classes
    channels_freq:       list of output channels per freq conv layer
    kernel_sizes_freq:   list of (time_k, freq_k) per freq conv layer
    strides_freq:        list of (time_s, freq_s) per freq conv layer
                         len must equal len(channels_freq) == len(kernel_sizes_freq)
    temporal_channels:   channel dimension for temporal conv stack and spectrum projection output
    num_temporal_layers: number of DSTemporalDWLayerClean layers
    temporal_kernel:     kernel size for temporal DW conv (odd, e.g. 3)
    fc_dim:              feature embedding dimension
    dropout_ratio:       dropout probability (applied in both conv stacks)
    pretrain_mode:       if True, adds projection_head for contrastive learning
    proj_hidden_dim:     hidden dim for projection_head
    proj_out_dim:        output dim for projection_head
    output_dims:         optional list of dims for MLP (if None, use sample_embd_layer)

    Returns
    -------
    dict: {'logits': [B, num_classes], 'features': [B, embed_dim]}
          embed_dim is output_dims[-1] if output_dims set, else fc_dim.
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
        pretrain_mode: bool = False,
        proj_hidden_dim: int = 256,
        proj_out_dim: int = 128,
        output_dims=None,
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
            freq_layers.append(DSDWConvLayerClean(in_ch, out_ch, k, s, dropout_ratio))
            in_ch = out_ch
        self.freq_stack = nn.ModuleList(freq_layers)

        # Spectrum projection: channels_freq[-1] * freq_out → temporal_channels
        freq_out = _compute_freq_out_dw(in_spectrum_len, kernel_sizes_freq, strides_freq)
        self.spectrum_proj = nn.Linear(channels_freq[-1] * freq_out, temporal_channels)

        # Temporal DW-sep conv stack
        temporal_layers = []
        for _ in range(num_temporal_layers):
            temporal_layers.append(
                DSTemporalDWLayerClean(temporal_channels, temporal_kernel, dropout_ratio)
            )
        self.temporal_stack = nn.ModuleList(temporal_layers)

        self.output_dims_mlp = None
        self.sample_embd_layer = None

        if output_dims is not None:
            self.output_dims_mlp = _build_output_dims_mlp(
                temporal_channels, output_dims, dropout_ratio
            )
            embed_dim = output_dims[-1]
        else:
            self.sample_embd_layer = nn.Sequential(
                nn.Linear(temporal_channels, fc_dim),
                nn.ReLU(),
            )
            embed_dim = fc_dim

        self.class_layer = nn.Linear(embed_dim, num_classes)

        self.pretrain_mode = pretrain_mode
        if self.pretrain_mode:
            self.projection_head = nn.Sequential(
                nn.Linear(embed_dim, proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_out_dim),
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, in_channels, intervals, spectrum]
        Returns:
            {'logits': [B, num_classes], 'features': [B, embed_dim]}
            If pretrain_mode: also includes 'projection'
        """
        # Freq DW-sep stack: [B, C_in, I, S] → [B, channels_freq[-1], I, freq_out]
        for layer in self.freq_stack:
            x = layer(x)

        # Spectrum projection: [B, C, I, S] → [B, I, C*S] → [B, I, T] → [B, T, I]
        B, C, I, S = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, I, C * S)
        x = self.spectrum_proj(x)           # [B, I, temporal_channels]
        x = x.permute(0, 2, 1)              # [B, temporal_channels, I]

        # Temporal DW-sep stack: [B, T, I] → [B, T, I]
        for layer in self.temporal_stack:
            x = layer(x)

        # Global average pool
        x = x.mean(dim=-1)                  # [B, temporal_channels]

        if self.output_dims_mlp is not None:
            features = self.output_dims_mlp(x)
        else:
            features = self.sample_embd_layer(x)

        if self.pretrain_mode:
            projection = self.projection_head(features)
            logits = self.class_layer(features)
            return {"features": features, "projection": projection, "logits": logits}

        logits = self.class_layer(features)
        return {'logits': logits, 'features': features}


# ---------------------------------------------------------------------------
# Single-modality wrapper
# ---------------------------------------------------------------------------

class SingleModalDeepSenseDWClean(nn.Module):
    """
    Thin wrapper around DeepSenseDWCleanBackbone.

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
        pretrain_mode: bool = False,
        proj_hidden_dim: int = 256,
        proj_out_dim: int = 128,
        output_dims=None,
    ):
        super().__init__()
        self.modality_name = modality_name
        self.location_name = location_name

        self.backbone = DeepSenseDWCleanBackbone(
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
            pretrain_mode=pretrain_mode,
            proj_hidden_dim=proj_hidden_dim,
            proj_out_dim=proj_out_dim,
            output_dims=output_dims,
        )

    def forward(self, freq_x: dict) -> dict:
        x = freq_x[self.location_name][self.modality_name]
        return self.backbone(x)
