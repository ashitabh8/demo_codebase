import torch
import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    """
    Ensure that all layers have a channel number divisible by `divisor`.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 Conv=None, BatchNorm=None):
        if Conv is None:
            Conv = nn.Conv2d
        if BatchNorm is None:
            BatchNorm = nn.BatchNorm2d
        padding = (kernel_size - 1) // 2
        super().__init__(
            Conv(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            BatchNorm(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """
    MobileNetV2 inverted residual block.
    """
    def __init__(self, inp, oup, stride, expand_ratio, Conv=None, BatchNorm=None):
        super().__init__()
        if Conv is None:
            Conv = nn.Conv2d
        if BatchNorm is None:
            BatchNorm = nn.BatchNorm2d

        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1,
                                     Conv=Conv, BatchNorm=BatchNorm))
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride,
                                 groups=hidden_dim, Conv=Conv, BatchNorm=BatchNorm))
        layers.append(Conv(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(BatchNorm(oup))

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 backbone with configurable Conv/BatchNorm for quantization support.
    """
    def __init__(self, in_channels=3, num_classes=1000, width_mult=1.0,
                 inverted_residual_setting=None, round_nearest=8,
                 Conv=None, BatchNorm=None, dropout_ratio=0.0,
                 use_standard_first_layer=True):
        super().__init__()
        if Conv is None:
            Conv = nn.Conv2d
        if BatchNorm is None:
            BatchNorm = nn.BatchNorm2d

        first_conv = nn.Conv2d if use_standard_first_layer else Conv
        first_bn = nn.BatchNorm2d if use_standard_first_layer else BatchNorm

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            # t, c, n, s
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(in_channels, input_channel, stride=2,
                               Conv=first_conv, BatchNorm=first_bn)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride,
                                     expand_ratio=t, Conv=Conv, BatchNorm=BatchNorm)
                )
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1,
                                   Conv=Conv, BatchNorm=BatchNorm))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = last_channel

        if dropout_ratio > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_ratio),
                nn.Linear(self.feature_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MeanFusionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)


class MultiModalMobileNetV2(nn.Module):
    """
    Multi-Modal and Multi-Location MobileNetV2 with Mean Fusion.
    """
    def __init__(self, modality_names, location_names, modality_in_channels,
                 num_classes, fc_dim=512, width_mult=1.0, Conv=None,
                 BatchNorm=None, dropout_ratio=0.0, use_standard_first_layer=True):
        super().__init__()
        self.modality_names = modality_names
        self.location_names = location_names
        self.multi_location_flag = len(location_names) > 1
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.Conv = Conv if Conv is not None else nn.Conv2d
        self.BatchNorm = BatchNorm if BatchNorm is not None else nn.BatchNorm2d

        self.mod_loc_backbones = nn.ModuleDict()
        for loc in location_names:
            self.mod_loc_backbones[loc] = nn.ModuleDict()
            for mod in modality_names:
                backbone = MobileNetV2(
                    in_channels=modality_in_channels[loc][mod],
                    num_classes=1000,  # Dummy; will replace classifier
                    width_mult=width_mult,
                    Conv=Conv,
                    BatchNorm=BatchNorm,
                    dropout_ratio=dropout_ratio,
                    use_standard_first_layer=use_standard_first_layer,
                )
                backbone.classifier = nn.Identity()
                self.mod_loc_backbones[loc][mod] = backbone

        # Feature dimension is shared across all backbones
        sample_backbone = next(iter(self.mod_loc_backbones[location_names[0]].values()))
        self.feature_dim = sample_backbone.feature_dim

        self.mod_fusion_layers = nn.ModuleDict()
        for loc in location_names:
            self.mod_fusion_layers[loc] = MeanFusionBlock()

        if self.multi_location_flag:
            self.loc_fusion_layer = MeanFusionBlock()

        self.sample_embd_layer = nn.Sequential(
            nn.Linear(self.feature_dim, fc_dim),
            nn.ReLU(),
        )
        self.class_layer = nn.Linear(fc_dim, num_classes)

    def get_conv_class(self):
        return self.Conv

    def forward(self, freq_x, return_embeddings=False):
        loc_mod_features = {}
        for loc in self.location_names:
            loc_mod_features[loc] = []
            for mod in self.modality_names:
                features = self.mod_loc_backbones[loc][mod](freq_x[loc][mod])
                loc_mod_features[loc].append(features)
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=1)

        fused_loc_features = {}
        for loc in self.location_names:
            fused_loc_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])

        if not self.multi_location_flag:
            final_feature = fused_loc_features[self.location_names[0]]
        else:
            loc_features = torch.stack(
                [fused_loc_features[loc] for loc in self.location_names], dim=1
            )
            final_feature = self.loc_fusion_layer(loc_features)

        sample_features = self.sample_embd_layer(final_feature)
        if return_embeddings:
            return sample_features
        logits = self.class_layer(sample_features)
        return logits


def build_mobilenet_v2(in_channels=3, num_classes=1000, width_mult=1.0,
                       Conv=None, BatchNorm=None, dropout_ratio=0.0,
                       use_standard_first_layer=True):
    return MobileNetV2(
        in_channels=in_channels,
        num_classes=num_classes,
        width_mult=width_mult,
        Conv=Conv,
        BatchNorm=BatchNorm,
        dropout_ratio=dropout_ratio,
        use_standard_first_layer=use_standard_first_layer,
    )


def build_multimodal_mobilenet_v2(modality_names, location_names, modality_in_channels,
                                 num_classes, fc_dim=512, width_mult=1.0, Conv=None,
                                 BatchNorm=None, dropout_ratio=0.0,
                                 use_standard_first_layer=True):
    return MultiModalMobileNetV2(
        modality_names=modality_names,
        location_names=location_names,
        modality_in_channels=modality_in_channels,
        num_classes=num_classes,
        fc_dim=fc_dim,
        width_mult=width_mult,
        Conv=Conv,
        BatchNorm=BatchNorm,
        dropout_ratio=dropout_ratio,
        use_standard_first_layer=use_standard_first_layer,
    )
