"""
ResNet Implementation with Configurable Conv and BatchNorm Layers

This module provides a flexible ResNet implementation that allows custom
convolutional and batch normalization layers while using standard PyTorch
layers for the first and last layers.

Features:
- Supports ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- Custom Conv and BatchNorm layers for intermediate layers
- Multi-modal and multi-location setups with mean fusion
- Simple one-function-call API

Quick Usage:
    
    # Single-modal ResNet
    from src2.models.ResNet import build_resnet
    model = build_resnet('resnet18', in_channels=3, num_classes=10)
    
    # Multi-modal ResNet with custom layers
    from src2.models.ResNet import build_multimodal_resnet
    from models.QuantModules import QuanConv, CustomBatchNorm
    
    modality_in_channels = {
        'loc1': {'acoustic': 1, 'seismic': 1}
    }
    model = build_multimodal_resnet(
        model_name='resnet18',
        modality_names=['acoustic', 'seismic'],
        location_names=['loc1'],
        modality_in_channels=modality_in_channels,
        num_classes=10,
        Conv=QuanConv,
        BatchNorm=CustomBatchNorm
    )
    
    # Forward pass with multi-modal input
    inputs = {
        'loc1': {
            'acoustic': torch.randn(4, 1, 224, 224),
            'seismic': torch.randn(4, 1, 224, 224)
        }
    }
    logits = model(inputs)
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


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18 and ResNet-34
    
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
        super(BasicBlock, self).__init__()
        
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
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Conv(in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False),
                BatchNorm(self.expansion * out_channels)
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


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet-50, ResNet-101, ResNet-152
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (before expansion)
        stride: Stride for the second convolution
        Conv: Custom convolution class (must be child of nn.Module)
        BatchNorm: Custom batch normalization class (must be child of nn.Module)
        dropout_ratio: Dropout probability (default: 0)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, Conv=None, BatchNorm=None, dropout_ratio=0):
        super(Bottleneck, self).__init__()
        
        # Use provided Conv and BatchNorm classes
        if Conv is None:
            Conv = nn.Conv2d
        if BatchNorm is None:
            BatchNorm = nn.BatchNorm2d
        
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(out_channels)
        
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm(out_channels)
        
        self.conv3 = Conv(out_channels, out_channels * self.expansion,
                         kernel_size=1, bias=False)
        self.bn3 = BatchNorm(out_channels * self.expansion)
        
        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Conv(in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False),
                BatchNorm(self.expansion * out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        identity = self.shortcut(identity)
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet Architecture with Configurable Conv and BatchNorm layers
    
    Args:
        block: Block type (BasicBlock or Bottleneck)
        layers: List of integers specifying number of blocks in each layer
        in_channels: Number of input channels (default: 3 for RGB images)
        num_classes: Number of output classes (default: 1000)
        Conv: Custom convolution class for intermediate layers (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class for intermediate layers (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Whether to use standard nn.Conv2d for first layer (default: True)
    """
    
    def __init__(self, block, layers, in_channels=3, num_classes=1000, 
                 Conv=None, BatchNorm=None, dropout_ratio=0, use_standard_first_layer=True):
        super(ResNet, self).__init__()
        
        # Set default Conv and BatchNorm if not provided
        if Conv is None:
            Conv = nn.Conv2d
        if BatchNorm is None:
            BatchNorm = nn.BatchNorm2d
        
        self.Conv = Conv
        self.BatchNorm = BatchNorm
        self.block = block
        self.dropout_ratio = dropout_ratio
        self.in_channels_current = 64
        
        # First layer - always use standard PyTorch layers
        if use_standard_first_layer:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = Conv(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
            
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers - use custom Conv and BatchNorm
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers - always use standard PyTorch layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()


    def get_conv_class(self):
        return self.Conv
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Create a ResNet layer consisting of multiple blocks
        
        Args:
            block: Block type (BasicBlock or Bottleneck)
            out_channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for the first block
        """
        layers = []
        
        # First block (may have stride > 1 and/or channel change)
        layers.append(block(self.in_channels_current, out_channels, stride, 
                           self.Conv, self.BatchNorm, self.dropout_ratio))
        self.in_channels_current = out_channels * block.expansion
        
        # Remaining blocks (stride=1, no channel change)
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels_current, out_channels, stride=1,
                               Conv=self.Conv, BatchNorm=self.BatchNorm, 
                               dropout_ratio=self.dropout_ratio))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, self.Conv)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, self.BatchNorm)):
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
        # First layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
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
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            nn.Flatten(1)
        )


class MultiModalResNet(nn.Module):
    """
    Multi-Modal and Multi-Location ResNet with Mean Fusion
    
    This model creates separate ResNet backbones for each combination of modality and location,
    then fuses features using mean pooling.
    
    Args:
        model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        modality_names: List of modality names (e.g., ['acoustic', 'seismic'])
        location_names: List of location names (e.g., ['loc1', 'loc2'])
        modality_in_channels: Dict mapping {location: {modality: in_channels}}
        num_classes: Number of output classes
        fc_dim: Dimension of the embedding layer before classification
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
    
    Example:
        >>> # Single location, multiple modalities
        >>> modality_in_channels = {
        ...     'loc1': {'acoustic': 1, 'seismic': 1}
        ... }
        >>> model = MultiModalResNet(
        ...     model_name='resnet18',
        ...     modality_names=['acoustic', 'seismic'],
        ...     location_names=['loc1'],
        ...     modality_in_channels=modality_in_channels,
        ...     num_classes=10,
        ...     fc_dim=256
        ... )
        >>> 
        >>> # Input format: dict[location][modality] = tensor
        >>> inputs = {
        ...     'loc1': {
        ...         'acoustic': torch.randn(4, 1, 224, 224),
        ...         'seismic': torch.randn(4, 1, 224, 224)
        ...     }
        ... }
        >>> logits = model(inputs)
    """
    
    def __init__(self, model_name, modality_names, location_names, 
                 modality_in_channels, num_classes, fc_dim=512,
                 Conv=None, BatchNorm=None, dropout_ratio=0, 
                 use_standard_first_layer=True):
        super(MultiModalResNet, self).__init__()
        
        self.modality_names = modality_names
        self.location_names = location_names
        self.multi_location_flag = len(location_names) > 1
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.Conv = Conv if Conv is not None else nn.Conv2d
        self.BatchNorm = BatchNorm if BatchNorm is not None else nn.BatchNorm2d
        
        # Determine block type and feature dimension
        model_configs = {
            'resnet18': (BasicBlock, [2, 2, 2, 2]),
            'resnet34': (BasicBlock, [3, 4, 6, 3]),
            'resnet50': (Bottleneck, [3, 4, 6, 3]),
            'resnet101': (Bottleneck, [3, 4, 23, 3]),
            'resnet152': (Bottleneck, [3, 8, 36, 3])
        }
        
        if model_name.lower() not in model_configs:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_configs.keys())}")
        
        block_type, layers = model_configs[model_name.lower()]
        self.feature_dim = 512 * block_type.expansion
        
        # Create ResNet backbone for each modality and location combination
        self.mod_loc_backbones = nn.ModuleDict()
        for loc in location_names:
            self.mod_loc_backbones[loc] = nn.ModuleDict()
            for mod in modality_names:
                # Create ResNet without final FC layer
                backbone = ResNet(
                    block=block_type,
                    layers=layers,
                    in_channels=modality_in_channels[loc][mod],
                    num_classes=1000,  # Dummy value, we'll remove the FC layer
                    Conv=Conv,
                    BatchNorm=BatchNorm,
                    dropout_ratio=dropout_ratio,
                    use_standard_first_layer=use_standard_first_layer
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
# Factory Functions for Different ResNet Variants
# =============================================================================

def resnet18(in_channels=3, num_classes=1000, Conv=None, BatchNorm=None, 
             dropout_ratio=0, use_standard_first_layer=True):
    """
    Constructs a ResNet-18 model
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1000)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
    
    Returns:
        ResNet-18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, 
                  num_classes=num_classes, Conv=Conv, BatchNorm=BatchNorm,
                  dropout_ratio=dropout_ratio, use_standard_first_layer=use_standard_first_layer)


def resnet34(in_channels=3, num_classes=1000, Conv=None, BatchNorm=None,
             dropout_ratio=0, use_standard_first_layer=True):
    """
    Constructs a ResNet-34 model
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1000)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
    
    Returns:
        ResNet-34 model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels,
                  num_classes=num_classes, Conv=Conv, BatchNorm=BatchNorm,
                  dropout_ratio=dropout_ratio, use_standard_first_layer=use_standard_first_layer)


def resnet50(in_channels=3, num_classes=1000, Conv=None, BatchNorm=None,
             dropout_ratio=0, use_standard_first_layer=True):
    """
    Constructs a ResNet-50 model
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1000)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
    
    Returns:
        ResNet-50 model
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels,
                  num_classes=num_classes, Conv=Conv, BatchNorm=BatchNorm,
                  dropout_ratio=dropout_ratio, use_standard_first_layer=use_standard_first_layer)


def resnet101(in_channels=3, num_classes=1000, Conv=None, BatchNorm=None,
              dropout_ratio=0, use_standard_first_layer=True):
    """
    Constructs a ResNet-101 model
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1000)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
    
    Returns:
        ResNet-101 model
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels,
                  num_classes=num_classes, Conv=Conv, BatchNorm=BatchNorm,
                  dropout_ratio=dropout_ratio, use_standard_first_layer=use_standard_first_layer)


def resnet152(in_channels=3, num_classes=1000, Conv=None, BatchNorm=None,
              dropout_ratio=0, use_standard_first_layer=True):
    """
    Constructs a ResNet-152 model
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1000)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
    
    Returns:
        ResNet-152 model
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], in_channels=in_channels,
                  num_classes=num_classes, Conv=Conv, BatchNorm=BatchNorm,
                  dropout_ratio=dropout_ratio, use_standard_first_layer=use_standard_first_layer)


def build_resnet(model_name, in_channels=3, num_classes=1000, Conv=nn.Conv2d, 
                BatchNorm=nn.BatchNorm2d, dropout_ratio=0, use_standard_first_layer=True):
    """
    Universal function to build any ResNet variant with one function call
    
    Args:
        model_name: Name of the model ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1000)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
    
    Returns:
        ResNet model of the specified variant
    
    Example:
        >>> from models.CustomConv import MyCustomConv
        >>> from models.CustomBatchNorm import MyCustomBatchNorm
        >>> 
        >>> # Build ResNet-50 with custom layers
        >>> model = build_resnet('resnet50', 
        ...                      in_channels=3, 
        ...                      num_classes=10,
        ...                      Conv=MyCustomConv,
        ...                      BatchNorm=MyCustomBatchNorm,
        ...                      dropout_ratio=0.2)
        >>>
        >>> # Build standard ResNet-18
        >>> model = build_resnet('resnet18', num_classes=100)
    """
    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }
    
    if model_name.lower() not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_dict.keys())}")
    
    return model_dict[model_name.lower()](
        in_channels=in_channels,
        num_classes=num_classes,
        Conv=Conv,
        BatchNorm=BatchNorm,
        dropout_ratio=dropout_ratio,
        use_standard_first_layer=use_standard_first_layer
    )


def build_multimodal_resnet(model_name, modality_names, location_names,
                            modality_in_channels, num_classes, fc_dim=512,
                            Conv=None, BatchNorm=None, dropout_ratio=0,
                            use_standard_first_layer=True):
    """
    Universal function to build multi-modal ResNet with one function call
    
    Args:
        model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        modality_names: List of modality names (e.g., ['acoustic', 'seismic'])
        location_names: List of location names (e.g., ['loc1', 'loc2'])
        modality_in_channels: Dict mapping {location: {modality: in_channels}}
        num_classes: Number of output classes
        fc_dim: Dimension of the embedding layer before classification (default: 512)
        Conv: Custom convolution class (default: nn.Conv2d)
        BatchNorm: Custom batch normalization class (default: nn.BatchNorm2d)
        dropout_ratio: Dropout probability (default: 0)
        use_standard_first_layer: Use standard nn.Conv2d for first layer (default: True)
    
    Returns:
        MultiModalResNet model
    
    Example:
        >>> # Multi-modal, multi-location setup
        >>> modality_in_channels = {
        ...     'loc1': {'acoustic': 1, 'seismic': 1},
        ...     'loc2': {'acoustic': 1, 'seismic': 1}
        ... }
        >>> model = build_multimodal_resnet(
        ...     model_name='resnet18',
        ...     modality_names=['acoustic', 'seismic'],
        ...     location_names=['loc1', 'loc2'],
        ...     modality_in_channels=modality_in_channels,
        ...     num_classes=10,
        ...     fc_dim=256
        ... )
    """
    return MultiModalResNet(
        model_name=model_name,
        modality_names=modality_names,
        location_names=location_names,
        modality_in_channels=modality_in_channels,
        num_classes=num_classes,
        fc_dim=fc_dim,
        Conv=Conv,
        BatchNorm=BatchNorm,
        dropout_ratio=dropout_ratio,
        use_standard_first_layer=use_standard_first_layer
    )


# =============================================================================
# Configurable ResNet for Student Models (Distillation / Compression)
# =============================================================================

class EarlyExitBranch(nn.Module):
    """
    Early exit classifier branch: Global Average Pooling -> Linear.
    
    Lightweight classifier attached at intermediate network stages for
    early exit inference and distillation training.
    
    Args:
        in_channels: Number of feature map channels at the exit point
        num_classes: Number of output classes
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Feature map tensor [B, C, H, W]
        Returns:
            logits: [B, num_classes]
        """
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ConfigurableResNet(nn.Module):
    """
    Fully configurable ResNet backbone for distillation student models.
    
    Unlike standard ResNet which has fixed filter progression [64, 128, 256, 512]
    and fixed 7x7 stem, this variant allows arbitrary filter sizes, layer counts,
    and stem configuration. This enables creating very small models suitable for
    distillation targets (e.g., 700-800KB).
    
    Uses BasicBlock (expansion=1) for all stages.
    
    Args:
        layers: List of ints, number of BasicBlocks per stage. e.g., [1,1,1,1]
        filter_sizes: List of ints, output channels per stage. e.g., [16,32,48,96]
        in_channels: Number of input channels
        num_classes: Number of output classes
        fc_dim: Hidden dimension for final classifier embedding
        stem_channels: Channels after stem conv (default: filter_sizes[0])
        stem_kernel: Kernel size for stem conv (default: 3)
        stem_stride: Stride for stem conv (default: 1)
        use_maxpool: Whether to use maxpool after stem (default: False)
        dropout_ratio: Dropout probability for BasicBlocks
        early_exit_layers: List of stage indices where early exits are placed.
                          e.g., [1, 2] means exits after stage 1 and stage 2.
    
    Example:
        >>> model = ConfigurableResNet(
        ...     layers=[1, 1, 1, 1],
        ...     filter_sizes=[16, 32, 48, 96],
        ...     in_channels=6, num_classes=10, fc_dim=64,
        ...     early_exit_layers=[1, 2]
        ... )
        >>> x = torch.randn(4, 6, 128, 128)
        >>> out = model(x)
        >>> out['logits'].shape   # torch.Size([4, 10])
        >>> len(out['exits'])     # 2
    """
    
    def __init__(self, layers, filter_sizes, in_channels=3, num_classes=10,
                 fc_dim=256, stem_channels=None, stem_kernel=3, stem_stride=1,
                 use_maxpool=False, dropout_ratio=0, early_exit_layers=None):
        super().__init__()
        
        assert len(layers) == len(filter_sizes), \
            f"layers ({len(layers)}) and filter_sizes ({len(filter_sizes)}) must have same length"
        
        self.layers_config = layers
        self.filter_sizes = filter_sizes
        self.num_stages = len(layers)
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.fc_dim = fc_dim
        self.early_exit_layers = sorted(early_exit_layers or [])
        
        # Stem
        if stem_channels is None:
            stem_channels = filter_sizes[0]
        self.stem_channels = stem_channels
        
        stem_padding = stem_kernel // 2
        self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=stem_kernel,
                               stride=stem_stride, padding=stem_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()
        
        # Build stages (stride=1 for first stage, stride=2 for rest)
        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for i, (num_blocks, out_ch) in enumerate(zip(layers, filter_sizes)):
            stride = 1 if i == 0 else 2
            stage = self._make_stage(in_ch, out_ch, num_blocks, stride)
            self.stages.append(stage)
            in_ch = out_ch * BasicBlock.expansion
        
        # Feature dimension after last stage
        self.feature_dim = filter_sizes[-1] * BasicBlock.expansion
        
        # Final classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = nn.Sequential(
            nn.Linear(self.feature_dim, fc_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(fc_dim, num_classes)
        
        # Early exit branches
        self.exit_branches = nn.ModuleDict()
        for layer_idx in self.early_exit_layers:
            assert 0 <= layer_idx < self.num_stages, \
                f"early_exit layer {layer_idx} out of range [0, {self.num_stages})"
            exit_channels = filter_sizes[layer_idx] * BasicBlock.expansion
            self.exit_branches[str(layer_idx)] = EarlyExitBranch(exit_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """Create a stage of BasicBlocks."""
        blocks = []
        blocks.append(BasicBlock(in_channels, out_channels, stride=stride,
                                 dropout_ratio=self.dropout_ratio))
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels * BasicBlock.expansion, out_channels,
                                     stride=1, dropout_ratio=self.dropout_ratio))
        return nn.Sequential(*blocks)
    
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
        
        Useful for multi-modal fusion where intermediate features are needed.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            stage_features: List of tensors, one per stage.
                           stage_features[i] has shape [B, filter_sizes[i], H_i, W_i]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
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
        
        # Collect early exit outputs
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


class SingleModalResNet(nn.Module):
    """
    Single-modality ResNet with configurable architecture and early exits.
    
    Thin wrapper around ConfigurableResNet that handles the dict-format input
    used by the data pipeline. Extracts one modality from one location and
    passes it through the backbone.
    
    Input format: freq_x[location][modality] = tensor [B, C, H, W]
    
    Args:
        modality_name: Which modality to use (e.g., 'audio' or 'seismic')
        location_name: Which location to use (e.g., 'shake')
        in_channels: Number of input channels for the chosen modality
        num_classes: Number of output classes
        layers: List of BasicBlock counts per stage (e.g., [1,1,1,1])
        filter_sizes: List of channel counts per stage (e.g., [16,32,48,96])
        fc_dim: Embedding dimension before classifier
        early_exit_layers: List of stage indices for early exits (e.g., [1, 2])
        stem_channels: Channels after stem conv (default: filter_sizes[0])
        stem_kernel: Stem conv kernel size (default: 3)
        stem_stride: Stem conv stride (default: 1)
        use_maxpool: Whether to use maxpool after stem (default: False)
        dropout_ratio: Dropout probability
    
    Example:
        >>> model = SingleModalResNet(
        ...     modality_name='audio', location_name='shake',
        ...     in_channels=6, num_classes=10,
        ...     layers=[1,1,1,1], filter_sizes=[16,32,48,96],
        ...     fc_dim=64, early_exit_layers=[1, 2]
        ... )
        >>> inputs = {'shake': {'audio': torch.randn(4, 6, 128, 128)}}
        >>> out = model(inputs)
        >>> out['logits'].shape   # [4, 10]
        >>> len(out['exits'])     # 2
    """
    
    def __init__(self, modality_name, location_name, in_channels,
                 num_classes, layers, filter_sizes, fc_dim=256,
                 early_exit_layers=None, stem_channels=None, stem_kernel=3,
                 stem_stride=1, use_maxpool=False, dropout_ratio=0):
        super().__init__()
        
        self.modality_name = modality_name
        self.location_name = location_name
        
        self.backbone = ConfigurableResNet(
            layers=layers,
            filter_sizes=filter_sizes,
            in_channels=in_channels,
            num_classes=num_classes,
            fc_dim=fc_dim,
            stem_channels=stem_channels,
            stem_kernel=stem_kernel,
            stem_stride=stem_stride,
            use_maxpool=use_maxpool,
            dropout_ratio=dropout_ratio,
            early_exit_layers=early_exit_layers,
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
    # Example 1: Standard ResNet-18
    print("=" * 60)
    print("Example 1: Standard ResNet-18")
    print("=" * 60)
    model = build_resnet('resnet18', in_channels=3, num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Example 2: ResNet-50 with custom layers (placeholder)
    print("\n" + "=" * 60)
    print("Example 2: ResNet-50 (placeholder for custom layers)")
    print("=" * 60)
    
    # Define a simple custom Conv (just for demonstration)
    class CustomConv(nn.Conv2d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # print(f"Created CustomConv with in_channels={self.in_channels}, out_channels={self.out_channels}")
    
    model = build_resnet('resnet50', in_channels=3, num_classes=100, 
                        Conv=CustomConv, dropout_ratio=0.1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Example 3: Multi-Modal ResNet (Single Location, Multiple Modalities)
    print("\n" + "=" * 60)
    print("Example 3: Multi-Modal ResNet (Single Location)")
    print("=" * 60)
    
    modality_in_channels = {
        'loc1': {'acoustic': 1, 'seismic': 1, 'infrared': 3}
    }
    
    model = MultiModalResNet(
        model_name='resnet18',
        modality_names=['acoustic', 'seismic', 'infrared'],
        location_names=['loc1'],
        modality_in_channels=modality_in_channels,
        num_classes=10,
        fc_dim=256,
        dropout_ratio=0.1
    )
    
    # Create dummy input
    batch_size = 4
    inputs = {
        'loc1': {
            'acoustic': torch.randn(batch_size, 1, 224, 224),
            'seismic': torch.randn(batch_size, 1, 224, 224),
            'infrared': torch.randn(batch_size, 3, 224, 224)
        }
    }
    
    logits = model(inputs)
    embeddings = model(inputs, return_embeddings=True)
    
    print(f"Input modalities: {list(inputs['loc1'].keys())}")
    print(f"Input shapes: {[inputs['loc1'][m].shape for m in inputs['loc1']]}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Example 4: Multi-Modal, Multi-Location ResNet
    print("\n" + "=" * 60)
    print("Example 4: Multi-Modal, Multi-Location ResNet")
    print("=" * 60)
    
    modality_in_channels = {
        'loc1': {'acoustic': 1, 'seismic': 1},
        'loc2': {'acoustic': 1, 'seismic': 1},
    }
    
    model = MultiModalResNet(
        model_name='resnet34',
        modality_names=['acoustic', 'seismic'],
        location_names=['loc1', 'loc2'],
        modality_in_channels=modality_in_channels,
        num_classes=5,
        fc_dim=128
    )
    
    # Create dummy input
    batch_size = 2
    inputs = {
        'loc1': {
            'acoustic': torch.randn(batch_size, 1, 224, 224),
            'seismic': torch.randn(batch_size, 1, 224, 224)
        },
        'loc2': {
            'acoustic': torch.randn(batch_size, 1, 224, 224),
            'seismic': torch.randn(batch_size, 1, 224, 224)
        }
    }
    
    logits = model(inputs)
    
    print(f"Locations: {list(inputs.keys())}")
    print(f"Modalities per location: {list(inputs['loc1'].keys())}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Example 5: ConfigurableResNet (Student) with Early Exits
    print("\n" + "=" * 60)
    print("Example 5: ConfigurableResNet with Early Exits")
    print("=" * 60)
    
    model = ConfigurableResNet(
        layers=[1, 1, 1, 1],
        filter_sizes=[16, 32, 48, 96],
        in_channels=6,
        num_classes=10,
        fc_dim=64,
        stem_kernel=3,
        stem_stride=1,
        use_maxpool=False,
        dropout_ratio=0.1,
        early_exit_layers=[1, 2],
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
    
    # Example 6: SingleModalResNet - Audio with Early Exits
    print("\n" + "=" * 60)
    print("Example 6: SingleModalResNet - Audio (ACIDS)")
    print("=" * 60)
    
    model = SingleModalResNet(
        modality_name='audio',
        location_name='shake',
        in_channels=6,
        num_classes=10,
        layers=[1, 1, 1, 1],
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
    
    # Example 7: SingleModalResNet - Seismic with Early Exits
    print("\n" + "=" * 60)
    print("Example 7: SingleModalResNet - Seismic (ACIDS)")
    print("=" * 60)
    
    model = SingleModalResNet(
        modality_name='seismic',
        location_name='shake',
        in_channels=4,
        num_classes=10,
        layers=[1, 1, 1, 1],
        filter_sizes=[16, 32, 48, 96],
        fc_dim=64,
        early_exit_layers=[1, 2],
        dropout_ratio=0.1,
    )
    
    inputs = {'shake': {'seismic': torch.randn(4, 4, 128, 128)}}
    out = model(inputs)
    
    print(f"Modality: seismic")
    print(f"Final logits shape: {out['logits'].shape}")
    print(f"Early exits: {len(out['exits'])}")
    print(f"Model size: {model.get_model_size_bytes() / 1024:.1f} KB")


