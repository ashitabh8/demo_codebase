import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block WITHOUT shortcut projection (in_channels == out_channels, stride == 1)."""

    def __init__(self, channels: int, conv_class=None, **conv_kwargs):
        super().__init__()
        conv_class = conv_class or nn.Conv2d

        self.conv1 = conv_class(channels, channels, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()

        self.conv2 = conv_class(channels, channels, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu2(out)
        return out


class BasicBlockDown(nn.Module):
    """Basic residual block WITH shortcut projection (channel change and/or stride > 1)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, conv_class=None, **conv_kwargs):
        super().__init__()
        conv_class = conv_class or nn.Conv2d

        self.conv1 = conv_class(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, **conv_kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = conv_class(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        # Shortcut projection
        self.shortcut_conv = conv_class(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, **conv_kwargs)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.shortcut_conv(x)
        identity = self.shortcut_bn(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu2(out)
        return out


class SimpleResNet(nn.Module):
    """
    Simple ResNet for inference - no conditionals.

    Architecture is fixed to:
      - Stem: Conv2d(in_channels -> 16, k=5, s=2)
      - Stage 0: 2 x BasicBlock(16)
      - Stage 1: BasicBlockDown(16 -> 16, s=2) + BasicBlock(16)
      - Stage 2: BasicBlockDown(16 -> 32, s=2) + BasicBlock(32)
      - Stage 3: BasicBlockDown(32 -> 64, s=2) + BasicBlock(64)
      - Head: Global avg pool -> Linear(64 -> 64) -> ReLU -> Linear(64 -> num_classes)

    Expected input shape: (B, in_channels, H, W) e.g. (B, 6, 7, 256).
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 10):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Stage 0: 16 -> 16, stride=1, 2 blocks
        self.stage0_block0 = BasicBlock(16)
        self.stage0_block1 = BasicBlock(16)

        # Stage 1: 16 -> 16, stride=2 at first block, 2 blocks
        self.stage1_block0 = BasicBlockDown(16, 16, stride=2)
        self.stage1_block1 = BasicBlock(16)

        # Stage 2: 16 -> 32, stride=2 at first block, 2 blocks
        self.stage2_block0 = BasicBlockDown(16, 32, stride=2)
        self.stage2_block1 = BasicBlock(32)

        # Stage 3: 32 -> 64, stride=2 at first block, 2 blocks
        self.stage3_block0 = BasicBlockDown(32, 64, stride=2)
        self.stage3_block1 = BasicBlock(64)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 64)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stage 0
        x = self.stage0_block0(x)
        x = self.stage0_block1(x)

        # Stage 1
        x = self.stage1_block0(x)
        x = self.stage1_block1(x)

        # Stage 2
        x = self.stage2_block0(x)
        x = self.stage2_block1(x)

        # Stage 3
        x = self.stage3_block0(x)
        x = self.stage3_block1(x)

        # Head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        return x

class SingleModalSimpleResNet(nn.Module):
    def __init__(self, location_name: str, modality_name: str, backbone: nn.Module):
        super().__init__()
        self.location_name = location_name
        self.modality_name = modality_name
        self.backbone = backbone

    def forward(self, inputs):
        x = inputs[self.location_name][self.modality_name]
        return self.backbone(x)

class ResNetSimpleBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        layers,
        filter_sizes,
        stem_kernel: int,
        stem_stride: int,
        use_maxpool: bool,
        fc_dim: int,
        weight_only_bits: int = None,
    ):
        super().__init__()
        # assert len(layers) == 4 and len(filter_sizes) == 4

        if weight_only_bits is not None:
            from models.WeightOnlyQuant import WeightOnlyConv2d, WeightOnlyLinear
            conv_class = WeightOnlyConv2d
            conv_kwargs = {"nbit": weight_only_bits}
        else:
            conv_class = nn.Conv2d
            conv_kwargs = {}

        stem_out = filter_sizes[0]
        self.conv1 = conv_class(
            in_channels,
            stem_out,
            kernel_size=stem_kernel,
            stride=stem_stride,
            padding=stem_kernel // 2,
            bias=False,
            **conv_kwargs,
        )
        self.bn1 = nn.BatchNorm2d(stem_out)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if use_maxpool else None

        self.stages = nn.ModuleList()
        in_c = stem_out
        for stage_idx, (num_blocks, out_c) in enumerate(zip(layers, filter_sizes)):
            blocks = []
            if stage_idx == 0:
                for _ in range(num_blocks):
                    blocks.append(BasicBlock(out_c, conv_class=conv_class, **conv_kwargs))
            else:
                blocks.append(BasicBlockDown(in_c, out_c, stride=2, conv_class=conv_class, **conv_kwargs))
                for _ in range(num_blocks - 1):
                    blocks.append(BasicBlock(out_c, conv_class=conv_class, **conv_kwargs))
            self.stages.append(nn.Sequential(*blocks))
            in_c = out_c

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if weight_only_bits is not None:
            self.fc1 = WeightOnlyLinear(filter_sizes[-1], fc_dim, nbit=weight_only_bits)
            self.fc2 = WeightOnlyLinear(fc_dim, num_classes, nbit=weight_only_bits)
        else:
            self.fc1 = nn.Linear(filter_sizes[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, num_classes)
        self.fc1_relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        return x

def build_simple_resnet_from_config(config: dict, model_key: str):
    """
    Build a ResNetSimpleBackbone from an ACIDS-style config.

    If model config has weight_only_qat: true and weight_only_bits: 2 or 4,
    builds a weight-only QAT backbone (fake-quantized weights in training;
    compiler-friendly Conv2d/Linear subclasses so Tiny-NN-in-C compiles it).

    Returns:
        backbone: ResNetSimpleBackbone instance
        location_name: str
        modality_name: str
    """
    m_cfg = config["models"][model_key]
    task_cfg = config["vehicle_classification"]

    num_classes = task_cfg["num_classes"]
    location_name = config["location_names"][0]
    modality_name = m_cfg["active_modality"]
    in_channels = config["loc_mod_in_freq_channels"][location_name][modality_name]

    weight_only_bits = None
    if m_cfg.get("weight_only_qat", False):
        weight_only_bits = m_cfg.get("weight_only_bits", 4)
        if weight_only_bits not in (2, 4):
            weight_only_bits = 4

    backbone = ResNetSimpleBackbone(
        in_channels=in_channels,
        num_classes=num_classes,
        layers=m_cfg["layers"],
        filter_sizes=m_cfg["filter_sizes"],
        stem_kernel=m_cfg["stem_kernel"],
        stem_stride=m_cfg["stem_stride"],
        use_maxpool=m_cfg["use_maxpool"],
        fc_dim=m_cfg["fc_dim"],
        weight_only_bits=weight_only_bits,
    )
    return backbone, location_name, modality_name