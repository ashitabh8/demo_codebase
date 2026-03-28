# src2/models/DeepSenseLatest.py
"""
DeepSense backbone for single-modality classification.

Classes
-------
DSConvLayer          : single Conv2d + BN + GELU + Dropout2d building block
DeepSenseBackbone    : configurable conv stack → freq projection → GRU → linear head
SingleModalDeepSense : thin wrapper unpacking freq_x[location][modality]
"""
import torch
import torch.nn as nn

from models.RecurrentModule import RecurrentBlock


class DSConvLayer(nn.Module):
    """
    Single 2-D conv layer: Conv2d → BatchNorm2d → GELU → Dropout2d.

    Padding convention (same as ConvLayer2D in ConvModules.py):
        max(stride) == 1  →  "same"   (output spatial dims unchanged)
        max(stride) >  1  →  "valid"  (no padding; dims shrink by kernel/stride)

    Args:
        in_channels:   input channels
        out_channels:  output channels
        kernel_size:   (freq_kernel, time_kernel)
        stride:        (freq_stride, time_stride)
        dropout_ratio: Dropout2d probability
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_ratio=0.0):
        super().__init__()
        kernel_size = tuple(kernel_size)
        stride = tuple(stride)
        padding = "same" if max(stride) == 1 else "valid"
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, padding_mode="zeros", bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p=dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


def _compute_freq_out(in_spectrum_len: int, kernel_sizes, strides) -> int:
    """Compute the freq dimension size after the conv stack."""
    freq = in_spectrum_len
    for k, s in zip(kernel_sizes, strides):
        freq_k = k[1]  # kernel size along freq dimension
        freq_s = s[1]  # stride along freq dimension
        if freq_s > 1:  # "valid" padding branch
            freq = (freq - freq_k) // freq_s + 1
        # else "same" padding: freq unchanged
    return freq


class DeepSenseBackbone(nn.Module):
    """
    Configurable DeepSense backbone for a single input tensor.

    Architecture
    ------------
    Conv stack (N = len(channels) layers):
        Layer i: DSConvLayer(in=channels[i-1], out=channels[i])
                 Layer 0 input is in_channels (from data).
        Input:  [B, in_channels, intervals, spectrum]
        Output: [B, channels[-1], intervals', spectrum']

    Spectrum projection (flatten freq into features + Linear):
        [B, channels[-1], intervals', freq_out]
        → reshape → [B, intervals', channels[-1] * freq_out]
        → spectrum_proj → [B, intervals', channels[-1]]
        → permute → [B, channels[-1], intervals']

    RecurrentBlock (bidirectional GRU):
        → [B, recurrent_dim * 2]

    sample_embd_layer (Linear + ReLU):
        → [B, fc_dim]   ← 'features'

    class_layer (Linear):
        → [B, num_classes]  ← 'logits'

    Args
    ----
    in_channels:      input channels (from dataset config)
    in_spectrum_len:  input spectrum length (used to compute freq_out after conv stack)
    num_classes:      number of output classes
    channels:         list of output channels per conv layer, e.g. [64, 128, 128]
    kernel_sizes:     list of (freq_k, time_k) per conv layer
    strides:          list of (freq_s, time_s) per conv layer
                      len(channels) == len(kernel_sizes) == len(strides) required
    recurrent_dim:    GRU hidden size (output dim is recurrent_dim * 2, bidirectional)
    recurrent_layers: number of GRU layers
    fc_dim:           feature embedding dimension
    dropout_ratio:    dropout probability for conv layers and GRU

    Returns
    -------
    dict: {'logits': [B, num_classes], 'features': [B, fc_dim]}
    """

    def __init__(
        self,
        in_channels: int,
        in_spectrum_len: int,
        num_classes: int,
        channels,
        kernel_sizes,
        strides,
        recurrent_dim: int,
        recurrent_layers: int,
        fc_dim: int,
        dropout_ratio: float = 0.0,
        pretrain_mode: bool = False,
        proj_hidden_dim: int = 256,
        proj_out_dim: int = 128,
    ):
        super().__init__()

        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(
                f"channels, kernel_sizes, and strides must all have the same length; "
                f"got {len(channels)}, {len(kernel_sizes)}, {len(strides)}"
            )

        # Build conv stack: channels[i] is output for layer i
        conv_layers = []
        in_ch = in_channels
        for out_ch, k, s in zip(channels, kernel_sizes, strides):
            conv_layers.append(DSConvLayer(in_ch, out_ch, k, s, dropout_ratio))
            in_ch = out_ch
        self.conv_stack = nn.ModuleList(conv_layers)

        # Project flattened [channels[-1] * freq_out] → channels[-1] per time step,
        # preserving all frequency information instead of global average pooling.
        freq_out = _compute_freq_out(in_spectrum_len, kernel_sizes, strides)
        self.spectrum_proj = nn.Linear(channels[-1] * freq_out, channels[-1])

        # Bidirectional GRU; input dim = channels[-1]
        self.recurrent_layer = RecurrentBlock(
            in_channel=channels[-1],
            out_channel=recurrent_dim,
            num_layers=recurrent_layers,
            dropout_ratio=dropout_ratio,
        )

        # Embedding projection
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(recurrent_dim * 2, fc_dim),  # *2 for bidirectional GRU
            nn.ReLU(),
        )

        # Classifier
        self.class_layer = nn.Linear(fc_dim, num_classes)

        self.pretrain_mode = pretrain_mode
        if self.pretrain_mode:
            self.projection_head = nn.Sequential(
                nn.Linear(fc_dim, proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_out_dim),
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, in_channels, intervals, spectrum]
        Returns:
            {'logits': [B, num_classes], 'features': [B, fc_dim]}
        """
        # Conv stack: [B, C, I, S] → [B, channels[-1], I', S']
        for layer in self.conv_stack:
            x = layer(x)

        # Flatten freq into features and project: preserves spectral information.
        # [B, C, I, S] → [B, I, C*S] → spectrum_proj → [B, I, C] → [B, C, I]
        B, C, I, S = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, I, C * S)
        x = self.spectrum_proj(x)
        x = x.permute(0, 2, 1)

        # GRU: → [B, recurrent_dim * 2]
        recurrent_out, _ = self.recurrent_layer(x)

        # Embed: → [B, fc_dim]
        features = self.sample_embd_layer(recurrent_out)

        if self.pretrain_mode:
            projection = self.projection_head(features)
            return {"features": features, "projection": projection}

        # Classify: → [B, num_classes]
        logits = self.class_layer(features)

        return {'logits': logits, 'features': features}


class SingleModalDeepSense(nn.Module):
    """
    Thin wrapper around DeepSenseBackbone — mirrors SingleModalResNet.

    Unpacks freq_x[location_name][modality_name] and forwards the tensor
    to the backbone. No architecture logic lives here.

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
        channels,
        kernel_sizes,
        strides,
        recurrent_dim: int,
        recurrent_layers: int,
        fc_dim: int,
        dropout_ratio: float = 0.0,
        pretrain_mode: bool = False,
        proj_hidden_dim: int = 256,
        proj_out_dim: int = 128,
    ):
        super().__init__()
        self.modality_name = modality_name
        self.location_name = location_name

        self.backbone = DeepSenseBackbone(
            in_channels=in_channels,
            in_spectrum_len=in_spectrum_len,
            num_classes=num_classes,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            recurrent_dim=recurrent_dim,
            recurrent_layers=recurrent_layers,
            fc_dim=fc_dim,
            dropout_ratio=dropout_ratio,
            pretrain_mode=pretrain_mode,
            proj_hidden_dim=proj_hidden_dim,
            proj_out_dim=proj_out_dim,
        )

    def forward(self, freq_x: dict) -> dict:
        x = freq_x[self.location_name][self.modality_name]
        return self.backbone(x)
