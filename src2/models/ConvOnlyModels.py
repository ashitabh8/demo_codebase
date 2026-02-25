"""
Conv-Only Models for Distillation Student Networks

Pure convolutional networks without residual connections, designed for
ultra-lightweight student models in knowledge distillation pipelines.

Features:
- Fully configurable: filter sizes, kernel sizes, strides, blocks per stage
- No residual connections (pure Conv-BN-ReLU stacks)
- Early exit support with GAP-based or Conv-based exit heads
- Dict-format input handling (same interface as MultiModalResNet)
- Single-modality (unimodal) design for distillation students

Quick Usage:
    
    # Single-modality ConvOnly with early exits
    from src2.models.ConvOnlyModels import SingleModalConvOnly
    
    model = SingleModalConvOnly(
        modality_name='audio',
        location_name='shake',
        in_channels=6,
        num_classes=10,
        num_blocks=[2, 2, 2, 2],
        filter_sizes=[16, 32, 48, 96],
        fc_dim=64,
        early_exit_layers=[1, 2],
    )
    
    inputs = {'shake': {'audio': torch.randn(4, 6, 128, 128)}}
    out = model(inputs)
    # out['logits'].shape == [4, 10]
    # len(out['exits']) == 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ResNet import EarlyExitBranch


class ConvBlock(nn.Module):
    """
    Single Conv -> BatchNorm -> ReLU block.
    
    The fundamental building block for ConvOnly networks.
    No residual connection.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size (default: 3)
        stride: Convolution stride (default: 1)
        dropout_ratio: Dropout probability (default: 0)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_ratio=0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
    
    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))


class ConvOnlyNet(nn.Module):
    """
    Pure convolutional network without residual connections.
    
    Architecture:
        Stem: Conv-BN-ReLU
        N stages, each containing M Conv-BN-ReLU blocks
        Global Average Pooling -> FC -> ReLU -> Classifier
    
    The first conv in each stage (except stage 0) uses stride > 1 for downsampling.
    Early exits can be placed after any stage using GAP-based or Conv-based heads.
    
    Args:
        num_blocks: List of ints, number of ConvBlocks per stage. e.g., [2, 2, 2, 2]
        filter_sizes: List of ints, output channels per stage. e.g., [16, 32, 48, 96]
        in_channels: Number of input channels
        num_classes: Number of output classes
        fc_dim: Hidden dimension for final classifier embedding
        kernel_sizes: List of ints, kernel size per stage (default: [3, 3, ...])
        strides: List of ints, stride for first conv in each stage (default: [1, 2, 2, ...])
        stem_channels: Channels after stem conv (default: filter_sizes[0])
        stem_kernel: Stem conv kernel size (default: 3)
        stem_stride: Stem conv stride (default: 1)
        dropout_ratio: Dropout probability
        early_exit_layers: List of stage indices for early exits
        early_exit_type: 'gap_linear' (GAP + Linear) or 'conv_gap' (1x1 Conv + GAP)
    
    Example:
        >>> model = ConvOnlyNet(
        ...     num_blocks=[2, 2, 2, 2],
        ...     filter_sizes=[16, 32, 48, 96],
        ...     in_channels=6, num_classes=10, fc_dim=64,
        ...     early_exit_layers=[1, 2],
        ...     early_exit_type='gap_linear'
        ... )
        >>> x = torch.randn(4, 6, 128, 128)
        >>> out = model(x)
        >>> out['logits'].shape   # [4, 10]
        >>> len(out['exits'])     # 2
    """
    
    def __init__(self, num_blocks, filter_sizes, in_channels=3, num_classes=10,
                 fc_dim=64, kernel_sizes=None, strides=None,
                 stem_channels=None, stem_kernel=3, stem_stride=1,
                 dropout_ratio=0, early_exit_layers=None,
                 early_exit_type="gap_linear"):
        super().__init__()
        
        num_stages = len(filter_sizes)
        assert len(num_blocks) == num_stages, \
            f"num_blocks ({len(num_blocks)}) and filter_sizes ({len(filter_sizes)}) must match"
        
        if kernel_sizes is None:
            kernel_sizes = [3] * num_stages
        if strides is None:
            strides = [1] + [2] * (num_stages - 1)
        if stem_channels is None:
            stem_channels = filter_sizes[0]
        
        assert len(kernel_sizes) == num_stages
        assert len(strides) == num_stages
        
        self.filter_sizes = filter_sizes
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.fc_dim = fc_dim
        self.early_exit_layers = sorted(early_exit_layers or [])
        self.early_exit_type = early_exit_type
        
        # Stem
        stem_padding = stem_kernel // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, stem_kernel,
                      stride=stem_stride, padding=stem_padding, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for i in range(num_stages):
            stage_blocks = []
            out_ch = filter_sizes[i]
            # First block may downsample
            stage_blocks.append(ConvBlock(in_ch, out_ch, kernel_sizes[i],
                                          strides[i], dropout_ratio))
            # Remaining blocks
            for _ in range(1, num_blocks[i]):
                stage_blocks.append(ConvBlock(out_ch, out_ch, kernel_sizes[i],
                                              1, dropout_ratio))
            self.stages.append(nn.Sequential(*stage_blocks))
            in_ch = out_ch
        
        # Feature dimension
        self.feature_dim = filter_sizes[-1]
        
        # Final head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = nn.Sequential(
            nn.Linear(self.feature_dim, fc_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(fc_dim, num_classes)
        
        # Early exit branches
        self.exit_branches = nn.ModuleDict()
        for layer_idx in self.early_exit_layers:
            assert 0 <= layer_idx < num_stages, \
                f"early_exit layer {layer_idx} out of range [0, {num_stages})"
            exit_ch = filter_sizes[layer_idx]
            if early_exit_type == "conv_gap":
                # 1x1 Conv -> GAP -> Flatten (CAM-style)
                self.exit_branches[str(layer_idx)] = nn.Sequential(
                    nn.Conv2d(exit_ch, num_classes, kernel_size=1, bias=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                )
            else:
                # GAP -> Linear (default, lightweight)
                self.exit_branches[str(layer_idx)] = EarlyExitBranch(exit_ch, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_stages(self, x):
        """
        Run input through stem and all stages, returning features after each stage.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            stage_features: List of tensors, one per stage.
        """
        x = self.stem(x)
        stage_features = []
        for stage in self.stages:
            x = stage(x)
            stage_features.append(x)
        return stage_features
    
    def forward(self, x):
        """
        Forward pass returning dict with logits, early exit outputs, and features.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            dict with keys:
                'logits': [B, num_classes] final classifier output
                'exits': list of [B, num_classes] early exit outputs (ordered by stage)
                'features': [B, fc_dim] final embedding before classifier
        """
        stage_features = self.forward_stages(x)
        
        # Early exits
        exits = []
        for layer_idx in self.early_exit_layers:
            exit_logits = self.exit_branches[str(layer_idx)](stage_features[layer_idx])
            exits.append(exit_logits)
        
        # Final head
        final_feat = stage_features[-1]
        final_feat = self.avgpool(final_feat)
        final_feat = torch.flatten(final_feat, 1)
        features = self.embed(final_feat)
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'exits': exits,
            'features': features,
        }
    
    def get_model_size_bytes(self):
        """Estimate model size in bytes (float32)."""
        return sum(p.numel() for p in self.parameters()) * 4


class SingleModalConvOnly(nn.Module):
    """
    Single-modality ConvOnly model with configurable architecture and early exits.
    
    Thin wrapper around ConvOnlyNet that handles the dict-format input
    used by the data pipeline. Extracts one modality from one location and
    passes it through the backbone.
    
    Input format: freq_x[location][modality] = tensor [B, C, H, W]
    
    Args:
        modality_name: Which modality to use (e.g., 'audio' or 'seismic')
        location_name: Which location to use (e.g., 'shake')
        in_channels: Number of input channels for the chosen modality
        num_classes: Number of output classes
        num_blocks: List of ConvBlock counts per stage (e.g., [2, 2, 2, 2])
        filter_sizes: List of channel counts per stage (e.g., [16, 32, 48, 96])
        fc_dim: Embedding dimension before classifier
        kernel_sizes: List of kernel sizes per stage (default: [3, 3, ...])
        strides: List of strides per stage (default: [1, 2, 2, ...])
        stem_channels: Channels after stem conv (default: filter_sizes[0])
        stem_kernel: Stem conv kernel size (default: 3)
        stem_stride: Stem conv stride (default: 1)
        dropout_ratio: Dropout probability
        early_exit_layers: List of stage indices for early exits (e.g., [1, 2])
        early_exit_type: 'gap_linear' (GAP + Linear) or 'conv_gap' (1x1 Conv + GAP)
    
    Example:
        >>> model = SingleModalConvOnly(
        ...     modality_name='audio', location_name='shake',
        ...     in_channels=6, num_classes=10,
        ...     num_blocks=[2,2,2,2], filter_sizes=[16,32,48,96],
        ...     fc_dim=64, early_exit_layers=[1, 2]
        ... )
        >>> inputs = {'shake': {'audio': torch.randn(4, 6, 128, 128)}}
        >>> out = model(inputs)
        >>> out['logits'].shape   # [4, 10]
        >>> len(out['exits'])     # 2
    """
    
    def __init__(self, modality_name, location_name, in_channels,
                 num_classes, num_blocks, filter_sizes, fc_dim=64,
                 kernel_sizes=None, strides=None,
                 stem_channels=None, stem_kernel=3, stem_stride=1,
                 dropout_ratio=0, early_exit_layers=None,
                 early_exit_type="gap_linear"):
        super().__init__()
        
        self.modality_name = modality_name
        self.location_name = location_name
        
        self.backbone = ConvOnlyNet(
            num_blocks=num_blocks,
            filter_sizes=filter_sizes,
            in_channels=in_channels,
            num_classes=num_classes,
            fc_dim=fc_dim,
            kernel_sizes=kernel_sizes,
            strides=strides,
            stem_channels=stem_channels,
            stem_kernel=stem_kernel,
            stem_stride=stem_stride,
            dropout_ratio=dropout_ratio,
            early_exit_layers=early_exit_layers,
            early_exit_type=early_exit_type,
        )
    
    def forward(self, freq_x):
        """
        Args:
            freq_x: Dict[location][modality] = tensor [B, C, H, W]
        
        Returns:
            dict: {'logits': [B, num_classes],
                   'exits': list of [B, num_classes],
                   'features': [B, fc_dim]}
        """
        x = freq_x[self.location_name][self.modality_name]
        return self.backbone(x)
    
    def get_model_size_bytes(self):
        """Estimate model size in bytes (float32)."""
        return sum(p.numel() for p in self.parameters()) * 4


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: ConvOnlyNet with early exits
    print("=" * 60)
    print("Example 1: ConvOnlyNet with Early Exits")
    print("=" * 60)
    
    model = ConvOnlyNet(
        num_blocks=[2, 2, 2, 2],
        filter_sizes=[16, 32, 48, 96],
        in_channels=6,
        num_classes=10,
        fc_dim=64,
        dropout_ratio=0.1,
        early_exit_layers=[1, 2],
        early_exit_type="gap_linear",
    )
    
    x = torch.randn(4, 6, 128, 128)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Final logits shape: {out['logits'].shape}")
    print(f"Number of early exits: {len(out['exits'])}")
    for i, exit_out in enumerate(out['exits']):
        print(f"  Exit {i} shape: {exit_out.shape}")
    print(f"Features shape: {out['features'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {model.get_model_size_bytes() / 1024:.1f} KB")
    
    # Example 2: ConvOnlyNet with conv_gap exits
    print("\n" + "=" * 60)
    print("Example 2: ConvOnlyNet with Conv-GAP Exits")
    print("=" * 60)
    
    model = ConvOnlyNet(
        num_blocks=[2, 2, 2, 2],
        filter_sizes=[16, 32, 48, 96],
        in_channels=6,
        num_classes=10,
        fc_dim=64,
        early_exit_layers=[1, 2],
        early_exit_type="conv_gap",
    )
    
    out = model(x)
    print(f"Final logits shape: {out['logits'].shape}")
    print(f"Exit 0 shape: {out['exits'][0].shape}")
    print(f"Model size: {model.get_model_size_bytes() / 1024:.1f} KB")
    
    # Example 3: SingleModalConvOnly - Audio (ACIDS)
    print("\n" + "=" * 60)
    print("Example 3: SingleModalConvOnly - Audio (ACIDS)")
    print("=" * 60)
    
    model = SingleModalConvOnly(
        modality_name='audio',
        location_name='shake',
        in_channels=6,
        num_classes=10,
        num_blocks=[2, 2, 2, 2],
        filter_sizes=[16, 32, 48, 96],
        fc_dim=64,
        early_exit_layers=[1, 2],
        dropout_ratio=0.1,
    )
    
    inputs = {'shake': {'audio': torch.randn(4, 6, 128, 128)}}
    out = model(inputs)
    
    print(f"Modality: audio")
    print(f"Final logits shape: {out['logits'].shape}")
    print(f"Early exits: {len(out['exits'])}")
    print(f"Model size: {model.get_model_size_bytes() / 1024:.1f} KB")
    
    # Example 4: SingleModalConvOnly - Seismic (ACIDS)
    print("\n" + "=" * 60)
    print("Example 4: SingleModalConvOnly - Seismic (ACIDS)")
    print("=" * 60)
    
    model = SingleModalConvOnly(
        modality_name='seismic',
        location_name='shake',
        in_channels=4,
        num_classes=10,
        num_blocks=[2, 2, 2, 2],
        filter_sizes=[16, 32, 48, 96],
        fc_dim=64,
        early_exit_layers=[1, 2],
    )
    
    inputs = {'shake': {'seismic': torch.randn(4, 4, 128, 128)}}
    out = model(inputs)
    
    print(f"Modality: seismic")
    print(f"Final logits shape: {out['logits'].shape}")
    print(f"Early exits: {len(out['exits'])}")
    print(f"Model size: {model.get_model_size_bytes() / 1024:.1f} KB")
    
    # Example 5: Tiny ConvOnly for size estimation
    print("\n" + "=" * 60)
    print("Example 5: Size Comparison - Various Configurations")
    print("=" * 60)
    
    configs = [
        ("Tiny",   [1, 1, 1, 1], [8, 16, 32, 64]),
        ("Small",  [2, 2, 2, 2], [16, 32, 48, 96]),
        ("Medium", [2, 2, 3, 3], [32, 64, 96, 128]),
        ("Large",  [3, 3, 4, 3], [64, 128, 192, 256]),
    ]
    
    for name, blocks, filters in configs:
        model = ConvOnlyNet(
            num_blocks=blocks,
            filter_sizes=filters,
            in_channels=6,
            num_classes=10,
            fc_dim=64,
        )
        size_kb = model.get_model_size_bytes() / 1024
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:8s}: blocks={blocks}, filters={filters} -> "
              f"{params:>8,} params, {size_kb:>8.1f} KB")
