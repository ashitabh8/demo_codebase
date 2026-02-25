"""
ResNet Implementation for CIFAR-style datasets (ResNet-20, 32, 44, 56, 110)

This module provides ResNet variants designed for smaller input sizes (32x32),
following the original "Deep Residual Learning for Image Recognition" paper
architecture for CIFAR-10/100.

Key differences from ImageNet ResNets:
- 3 stages instead of 4 (with 16, 32, 64 filters)
- First layer is 3x3 conv with stride 1 (no 7x7 conv or max pooling)
- Designed for 32x32 input images
- Uses BasicBlock only (no Bottleneck variants)

Features:
- Supports ResNet-20, ResNet-32, ResNet-44, ResNet-56, ResNet-110
- Custom Conv and BatchNorm layers for intermediate layers
- Multi-modal and multi-location setups with mean fusion
- Simple one-function-call API

Quick Usage:
    
    # Single-modal ResNet-20
    from ResNetCIFAR import build_resnet_cifar
    model = build_resnet_cifar('resnet20', in_channels=3, num_classes=10)
    
    # Multi-modal ResNet with custom layers
    from ResNetCIFAR import build_multimodal_resnet_cifar
    from models.QuantModules import QuanConv, CustomBatchNorm
    
    modality_in_channels = {
        'loc1': {'acoustic': 1, 'seismic': 1}
    }
    model = build_multimodal_resnet_cifar(
        model_name='resnet20',
        modality_names=['acoustic', 'seismic'],
        location_names=['loc1'],
        modality_in_channels=modality_in_channels,
        num_classes=10,
        Conv=QuanConv,
        BatchNorm=CustomBatchNorm
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFusionBlock(nn.Module):
    """
    Simple mean fusion block that averages features across a specified dimension
    """
    def __init__(self):
        super(MeanFusionBlock, self).__init__()
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, num_items, features]
        Returns:
            Averaged tensor of shape [batch, features]
        """
        return torch.mean(x, dim=1)


class BasicBlockCIFAR(nn.Module):
    """
    Basic residual block for CIFAR-style ResNets
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution
        Conv: Custom convolution class (must be child of nn.Module)
        BatchNorm: Custom batch normalization class (must be child of nn.Module)
        dropout_ratio: Dropout probability (default: 0)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, Conv=None, BatchNorm=None, dropout_ratio=0):
        super(BasicBlockCIFAR, self).__init__()
        
        # Use provided Conv and BatchNorm classes
        if Conv is None:
            Conv = nn.Conv2d
        if BatchNorm is None:
            BatchNorm = nn.BatchNorm2d
            
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, 
                         stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm(out_channels)
        
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BatchNorm(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.shortcut(identity)
        out += identity
        out = F.relu(out)
        
        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet Architecture for CIFAR-style datasets (32x32 images)
    
    Architecture:
        - Initial 3x3 conv (no stride, no pooling)
        - Stage 1: n blocks with 16 filters
        - Stage 2: n blocks with 32 filters (first block has stride 2)
        - Stage 3: n blocks with 64 filters (first block has stride 2)
        - Global average pooling
        - Fully connected layer
    
    For ResNet-20: n=3, for ResNet-32: n=5, for ResNet-56: n=9, etc.
    Total layers = 6n + 2
    
    Args:
        num_blocks: Number of blocks per stage (n)
        in_channels: Number of input channels (default: 3 for RGB images)
        num_classes: Number of output classes (default: 10)
        Conv: Custom convolution class for intermediate layers (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class for intermediate layers (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Whether to use standard nn.Conv2d for first layer (default: True)
        initial_channels: Number of channels after first conv (default: 16)
    """
    
    def __init__(self, num_blocks, in_channels=3, num_classes=10, 
                 Conv=None, BatchNorm=None, dropout_ratio=0, 
                 use_standard_first_layer=True, initial_channels=16):
        super(ResNetCIFAR, self).__init__()
        
        # Set default Conv and BatchNorm if not provided
        if Conv is None:
            Conv = nn.Conv2d
        if BatchNorm is None:
            BatchNorm = nn.BatchNorm2d
        
        self.Conv = Conv
        self.BatchNorm = BatchNorm
        self.dropout_ratio = dropout_ratio
        self.in_channels_current = initial_channels
        self.initial_channels = initial_channels
        
        # First layer - 3x3 conv, no stride, no pooling
        if use_standard_first_layer:
            self.conv1 = nn.Conv2d(in_channels, initial_channels, kernel_size=3, 
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(initial_channels)
        else:
            self.conv1 = Conv(in_channels, initial_channels, kernel_size=3, 
                             stride=1, padding=1, bias=False)
            self.bn1 = BatchNorm(initial_channels)
        
        # Three stages with increasing channels
        self.layer1 = self._make_layer(initial_channels, num_blocks, stride=1)
        self.layer2 = self._make_layer(initial_channels * 2, num_blocks, stride=2)
        self.layer3 = self._make_layer(initial_channels * 4, num_blocks, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(initial_channels * 4, num_classes)
        
        # Store feature dimension for external access
        self.feature_dim = initial_channels * 4
        
        # Initialize weights
        self._initialize_weights()
    
    def get_conv_class(self):
        return self.Conv
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Create a ResNet stage consisting of multiple blocks
        
        Args:
            out_channels: Number of output channels for this stage
            num_blocks: Number of blocks in this stage
            stride: Stride for the first block
        """
        layers = []
        
        # First block (may have stride > 1 and/or channel change)
        layers.append(BasicBlockCIFAR(
            self.in_channels_current, out_channels, stride,
            self.Conv, self.BatchNorm, self.dropout_ratio
        ))
        self.in_channels_current = out_channels
        
        # Remaining blocks (stride=1, no channel change)
        for _ in range(1, num_blocks):
            layers.append(BasicBlockCIFAR(
                self.in_channels_current, out_channels, stride=1,
                Conv=self.Conv, BatchNorm=self.BatchNorm,
                dropout_ratio=self.dropout_ratio
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d,)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # First layer (no pooling for CIFAR)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Three stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_feature_extractor(self):
        """
        Returns a version of the model without the final classification layer
        Useful for transfer learning or feature extraction
        """
        return nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.layer1,
            self.layer2,
            self.layer3,
            self.avgpool,
            nn.Flatten(1)
        )


class MultiModalResNetCIFAR(nn.Module):
    """
    Multi-Modal and Multi-Location ResNet for CIFAR-style datasets
    
    This model creates separate ResNet backbones for each combination of modality and location,
    then fuses features using mean pooling.
    
    Args:
        model_name: ResNet variant ('resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110')
        modality_names: List of modality names (e.g., ['acoustic', 'seismic'])
        location_names: List of location names (e.g., ['loc1', 'loc2'])
        modality_in_channels: Dict mapping {location: {modality: in_channels}}
        num_classes: Number of output classes
        fc_dim: Dimension of the embedding layer before classification
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
        initial_channels: Number of channels after first conv (default: 16)
    """
    
    def __init__(self, model_name, modality_names, location_names, 
                 modality_in_channels, num_classes, fc_dim=256,
                 Conv=None, BatchNorm=None, dropout_ratio=0, 
                 use_standard_first_layer=True, initial_channels=16):
        super(MultiModalResNetCIFAR, self).__init__()
        
        self.modality_names = modality_names
        self.location_names = location_names
        self.multi_location_flag = len(location_names) > 1
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.Conv = Conv if Conv is not None else nn.Conv2d
        self.BatchNorm = BatchNorm if BatchNorm is not None else nn.BatchNorm2d
        
        # Determine number of blocks per stage
        model_configs = {
            'resnet20': 3,   # 6*3 + 2 = 20
            'resnet32': 5,   # 6*5 + 2 = 32
            'resnet44': 7,   # 6*7 + 2 = 44
            'resnet56': 9,   # 6*9 + 2 = 56
            'resnet110': 18, # 6*18 + 2 = 110
        }
        
        if model_name.lower() not in model_configs:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_configs.keys())}")
        
        num_blocks = model_configs[model_name.lower()]
        self.feature_dim = initial_channels * 4  # 64 for default initial_channels=16
        
        # Create ResNet backbone for each modality and location combination
        self.mod_loc_backbones = nn.ModuleDict()
        for loc in location_names:
            self.mod_loc_backbones[loc] = nn.ModuleDict()
            for mod in modality_names:
                # Create ResNet without final FC layer
                backbone = ResNetCIFAR(
                    num_blocks=num_blocks,
                    in_channels=modality_in_channels[loc][mod],
                    num_classes=1000,  # Dummy value, we'll remove the FC layer
                    Conv=Conv,
                    BatchNorm=BatchNorm,
                    dropout_ratio=dropout_ratio,
                    use_standard_first_layer=use_standard_first_layer,
                    initial_channels=initial_channels
                )
                # Remove the final FC layer - we'll add our own
                backbone.fc = nn.Identity()
                self.mod_loc_backbones[loc][mod] = backbone
        
        # Modality fusion for each location
        self.mod_fusion_layers = nn.ModuleDict()
        for loc in location_names:
            self.mod_fusion_layers[loc] = MeanFusionBlock()
        
        # Location fusion if multiple locations
        if self.multi_location_flag:
            self.loc_fusion_layer = MeanFusionBlock()
        
        # Final classification layers
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(self.feature_dim, fc_dim),
            nn.ReLU(),
        )
        
        self.class_layer = nn.Linear(fc_dim, num_classes)
    
    def get_conv_class(self):
        return self.Conv
    
    def forward(self, freq_x, return_embeddings=False):
        """
        Forward pass through multi-modal ResNet
        
        Args:
            freq_x: Dict of dicts mapping {location: {modality: tensor}}
                   Each tensor is of shape [batch, channels, height, width]
            return_embeddings: If True, return embeddings instead of logits
        
        Returns:
            logits: Classification logits of shape [batch, num_classes]
                   OR embeddings of shape [batch, fc_dim] if return_embeddings=True
        """
        loc_mod_features = {}
        
        # Extract features from each modality and location
        for loc in self.location_names:
            loc_mod_features[loc] = []
            for mod in self.modality_names:
                features = self.mod_loc_backbones[loc][mod](freq_x[loc][mod])
                loc_mod_features[loc].append(features)
            # Stack features: [batch, num_modalities, feature_dim]
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=1)
        
        # Fuse modalities for each location using mean fusion
        fused_loc_features = {}
        for loc in self.location_names:
            # Mean across modalities: [batch, num_modalities, feature_dim] -> [batch, feature_dim]
            fused_loc_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])
        
        # Fuse locations if multiple locations
        if not self.multi_location_flag:
            final_feature = fused_loc_features[self.location_names[0]]
        else:
            # Stack location features: [batch, num_locations, feature_dim]
            loc_features = torch.stack([fused_loc_features[loc] for loc in self.location_names], dim=1)
            # Mean across locations: [batch, num_locations, feature_dim] -> [batch, feature_dim]
            final_feature = self.loc_fusion_layer(loc_features)
        
        # Generate embeddings
        sample_features = self.sample_embd_layer(final_feature)
        
        if return_embeddings:
            return sample_features
        else:
            # Classification
            logits = self.class_layer(sample_features)
            return logits


# =============================================================================
# Factory Functions for Different ResNet CIFAR Variants
# =============================================================================

def resnet20(in_channels=3, num_classes=10, Conv=None, BatchNorm=None,
             dropout_ratio=0, use_standard_first_layer=True, initial_channels=16):
    """
    Constructs a ResNet-20 model for CIFAR-style datasets
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 10)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
        initial_channels: Number of channels after first conv (default: 16)
    
    Returns:
        ResNet-20 model
    """
    return ResNetCIFAR(num_blocks=3, in_channels=in_channels, num_classes=num_classes,
                       Conv=Conv, BatchNorm=BatchNorm, dropout_ratio=dropout_ratio,
                       use_standard_first_layer=use_standard_first_layer,
                       initial_channels=initial_channels)


def resnet32(in_channels=3, num_classes=10, Conv=None, BatchNorm=None,
             dropout_ratio=0, use_standard_first_layer=True, initial_channels=16):
    """
    Constructs a ResNet-32 model for CIFAR-style datasets
    """
    return ResNetCIFAR(num_blocks=5, in_channels=in_channels, num_classes=num_classes,
                       Conv=Conv, BatchNorm=BatchNorm, dropout_ratio=dropout_ratio,
                       use_standard_first_layer=use_standard_first_layer,
                       initial_channels=initial_channels)


def resnet44(in_channels=3, num_classes=10, Conv=None, BatchNorm=None,
             dropout_ratio=0, use_standard_first_layer=True, initial_channels=16):
    """
    Constructs a ResNet-44 model for CIFAR-style datasets
    """
    return ResNetCIFAR(num_blocks=7, in_channels=in_channels, num_classes=num_classes,
                       Conv=Conv, BatchNorm=BatchNorm, dropout_ratio=dropout_ratio,
                       use_standard_first_layer=use_standard_first_layer,
                       initial_channels=initial_channels)


def resnet56(in_channels=3, num_classes=10, Conv=None, BatchNorm=None,
             dropout_ratio=0, use_standard_first_layer=True, initial_channels=16):
    """
    Constructs a ResNet-56 model for CIFAR-style datasets
    """
    return ResNetCIFAR(num_blocks=9, in_channels=in_channels, num_classes=num_classes,
                       Conv=Conv, BatchNorm=BatchNorm, dropout_ratio=dropout_ratio,
                       use_standard_first_layer=use_standard_first_layer,
                       initial_channels=initial_channels)


def resnet110(in_channels=3, num_classes=10, Conv=None, BatchNorm=None,
              dropout_ratio=0, use_standard_first_layer=True, initial_channels=16):
    """
    Constructs a ResNet-110 model for CIFAR-style datasets
    """
    return ResNetCIFAR(num_blocks=18, in_channels=in_channels, num_classes=num_classes,
                       Conv=Conv, BatchNorm=BatchNorm, dropout_ratio=dropout_ratio,
                       use_standard_first_layer=use_standard_first_layer,
                       initial_channels=initial_channels)


def build_resnet_cifar(model_name, in_channels=3, num_classes=10, Conv=None, 
                       BatchNorm=None, dropout_ratio=0, use_standard_first_layer=True,
                       initial_channels=16):
    """
    Universal function to build any CIFAR-style ResNet variant with one function call
    
    Args:
        model_name: Name of the model ('resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110')
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 10)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
        initial_channels: Number of channels after first conv (default: 16)
    
    Returns:
        ResNet model of the specified variant
    
    Example:
        >>> from models.CustomConv import MyCustomConv
        >>> 
        >>> # Build ResNet-20 with custom layers
        >>> model = build_resnet_cifar('resnet20', 
        ...                            in_channels=3, 
        ...                            num_classes=10,
        ...                            Conv=MyCustomConv,
        ...                            dropout_ratio=0.1)
        >>>
        >>> # Build standard ResNet-56
        >>> model = build_resnet_cifar('resnet56', num_classes=100)
    """
    model_dict = {
        'resnet20': resnet20,
        'resnet32': resnet32,
        'resnet44': resnet44,
        'resnet56': resnet56,
        'resnet110': resnet110
    }
    
    if model_name.lower() not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_dict.keys())}")
    
    return model_dict[model_name.lower()](
        in_channels=in_channels,
        num_classes=num_classes,
        Conv=Conv,
        BatchNorm=BatchNorm,
        dropout_ratio=dropout_ratio,
        use_standard_first_layer=use_standard_first_layer,
        initial_channels=initial_channels
    )


def build_multimodal_resnet_cifar(model_name, modality_names, location_names,
                                  modality_in_channels, num_classes, fc_dim=256,
                                  Conv=None, BatchNorm=None, dropout_ratio=0,
                                  use_standard_first_layer=True, initial_channels=16):
    """
    Universal function to build multi-modal CIFAR-style ResNet with one function call
    
    Args:
        model_name: ResNet variant ('resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110')
        modality_names: List of modality names (e.g., ['acoustic', 'seismic'])
        location_names: List of location names (e.g., ['loc1', 'loc2'])
        modality_in_channels: Dict mapping {location: {modality: in_channels}}
        num_classes: Number of output classes
        fc_dim: Dimension of the embedding layer before classification (default: 256)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
        initial_channels: Number of channels after first conv (default: 16)
    
    Returns:
        MultiModalResNetCIFAR model
    
    Example:
        >>> modality_in_channels = {
        ...     'loc1': {'acoustic': 1, 'seismic': 1}
        ... }
        >>> model = build_multimodal_resnet_cifar(
        ...     model_name='resnet20',
        ...     modality_names=['acoustic', 'seismic'],
        ...     location_names=['loc1'],
        ...     modality_in_channels=modality_in_channels,
        ...     num_classes=10,
        ...     fc_dim=128
        ... )
    """
    return MultiModalResNetCIFAR(
        model_name=model_name,
        modality_names=modality_names,
        location_names=location_names,
        modality_in_channels=modality_in_channels,
        num_classes=num_classes,
        fc_dim=fc_dim,
        Conv=Conv,
        BatchNorm=BatchNorm,
        dropout_ratio=dropout_ratio,
        use_standard_first_layer=use_standard_first_layer,
        initial_channels=initial_channels
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Standard ResNet-20 for CIFAR-10
    print("=" * 60)
    print("Example 1: Standard ResNet-20 for CIFAR-10")
    print("=" * 60)
    model = build_resnet_cifar('resnet20', in_channels=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")
    
    # Example 2: ResNet-56 with custom layers (placeholder)
    print("\n" + "=" * 60)
    print("Example 2: ResNet-56 with wider channels")
    print("=" * 60)
    
    # Using wider initial channels (32 instead of 16)
    model = build_resnet_cifar('resnet56', in_channels=3, num_classes=100,
                               dropout_ratio=0.1, initial_channels=32)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")
    
    # Example 3: Multi-Modal ResNet-20
    print("\n" + "=" * 60)
    print("Example 3: Multi-Modal ResNet-20 (Single Location)")
    print("=" * 60)
    
    modality_in_channels = {
        'loc1': {'acoustic': 1, 'seismic': 1}
    }
    
    model = MultiModalResNetCIFAR(
        model_name='resnet20',
        modality_names=['acoustic', 'seismic'],
        location_names=['loc1'],
        modality_in_channels=modality_in_channels,
        num_classes=10,
        fc_dim=128,
        dropout_ratio=0.1
    )
    
    # Create dummy input (32x32 for CIFAR-style)
    batch_size = 4
    inputs = {
        'loc1': {
            'acoustic': torch.randn(batch_size, 1, 32, 32),
            'seismic': torch.randn(batch_size, 1, 32, 32)
        }
    }
    
    logits = model(inputs)
    embeddings = model(inputs, return_embeddings=True)
    
    print(f"Input modalities: {list(inputs['loc1'].keys())}")
    print(f"Input shapes: {[inputs['loc1'][m].shape for m in inputs['loc1']]}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")
    
    # Example 4: Compare all model sizes
    print("\n" + "=" * 60)
    print("Example 4: Parameter counts for all CIFAR ResNet variants")
    print("=" * 60)
    
    for model_name in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
        model = build_resnet_cifar(model_name, num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        print(f"{model_name}: {params / 1e6:.4f}M parameters")
    
    # Example 5: Test with HAR-style input dimensions
    print("\n" + "=" * 60)
    print("Example 5: ResNet-20 with HAR-style input (non-square)")
    print("=" * 60)
    
    model = build_resnet_cifar('resnet20', in_channels=9, num_classes=6)  # 9 sensor channels, 6 activities
    # Typical HAR input: 9 channels (3-axis accel, 3-axis gyro, 3-axis mag), window of samples
    x = torch.randn(4, 9, 128, 1)  # 128 time steps, 1 "width" (essentially 1D but using 2D conv)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")