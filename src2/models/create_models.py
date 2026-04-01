import logging
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from models.ResNet import SingleModalResNet
from models.ConvOnlyModels import SingleModalConvOnly
from models.ResNetSimple import (
    ResNetSimpleBackbone,
    SingleModalSimpleResNet,
    build_simple_resnet_from_config,
)
from models.DeepSenseLatest import SingleModalDeepSense
from models.DeepSenseDepthwise import SingleModalDeepSenseDW
from models.DeepSenseDWSimple import (
    DeepSenseDWSimpleBackbone,
    SingleModalDeepSenseDWSimple,
)


def resolve_num_classes(config, model_cfg):
    """
    Classifier output width. SSL pretrain sets pretrain_mode True and does not
    define task_name; use a dummy width (the supervised head is unused in forward).
    Supervised runs set pretrain_mode False and require config['task_name'].
    """
    if model_cfg["pretrain_mode"]:
        return 1
    task_name = config["task_name"]
    return config[task_name]["num_classes"]


# =============================================================================
# Student Model Factory for Distillation
# =============================================================================


def create_single_modal_model(config, model_name):
    """
    Create a single-modality model from config.

    This factory creates unimodal models (ResNet or ConvOnly) that can serve as
    either teachers or students in distillation pipelines. The model accepts
    dict-format input but only processes one modality.

    Input format: freq_x[location][modality] = tensor [B, C, H, W]

    Returns a model whose forward() returns a dict:
        {'logits': [B, num_classes],
         'exits': [exit_0, exit_1, ...],
         'features': [B, fc_dim]}

    Config validation:
        - Uses strict dictionary access [] to raise KeyError for missing required keys
        - All keys MUST be explicitly present in the config

    Args:
        config: Full dataset configuration dictionary (e.g., loaded from ACIDS.yaml)
        model_config_key: Key in config["models"] for model settings
                         (e.g., 'teacher_audio_resnet18', 'student_audio_resnet')

    Returns:
        model: SingleModalResNet or SingleModalConvOnly instance

    Example:
        >>> import yaml
        >>> with open('src2/data/ACIDS.yaml') as f:
        ...     config = yaml.safe_load(f)
        >>> teacher = create_single_modal_model(config, 'teacher_audio_resnet18')
        >>> student = create_single_modal_model(config, 'student_audio_resnet')
        >>> inputs = {'shake': {'audio': torch.randn(4, 6, 128, 128)}}
        >>> teacher_out = teacher(inputs)
        >>> student_out = student(inputs)
    """
    # Model config is stored under `config["models"]` in our YAMLs.
    model_cfg = config["models"][model_name]
    model_type = model_cfg["model_type"]

    # DeepSense variants have their own factories — dispatch early
    if model_type == "deepsense":
        return create_deepsense(config, model_name)
    if model_type == "deepsense_dw":
        return create_deepsense_dw(config, model_name)
    if model_type == "deepsense_dw_simple":
        return create_deepsense_dw_simple(config, model_name)

    # Get the single active modality and location
    active_modality = model_cfg["active_modality"]
    location_names = config["location_names"]

    if len(location_names) != 1:
        raise ValueError(
            f"Single-modal models expect exactly one location, got {location_names}"
        )
    location_name = location_names[0]

    # Look up input channels for this modality
    all_channels = config["loc_mod_in_freq_channels"]
    in_channels = all_channels[location_name][active_modality]

    num_classes = resolve_num_classes(config, model_cfg)

    # Common parameters
    fc_dim = model_cfg["fc_dim"]
    dropout_ratio = model_cfg["dropout_ratio"]
    stem_kernel = model_cfg["stem_kernel"]
    stem_stride = model_cfg["stem_stride"]
    stem_channels = model_cfg["stem_channels"]

    logging.info(f"Creating single-modal model: {model_type} ({model_name})")
    logging.info(f"  Modality: {active_modality}")
    logging.info(f"  Location: {location_name}")
    logging.info(f"  Input channels: {in_channels}")
    logging.info(f"  Number of classes: {num_classes}")
    logging.info(f"  FC dim: {fc_dim}")

    if model_type in ("student_resnet", "resnet"):
        layers = model_cfg["layers"]
        filter_sizes = model_cfg["filter_sizes"]
        use_maxpool = model_cfg["use_maxpool"]

        logging.info(f"  ResNet layers: {layers}")
        logging.info(f"  Filter sizes: {filter_sizes}")

        print(f"model_cfg: {model_cfg}")  # DEBUG: print model_cfg
        # breakpoint()
        pretrain_mode = model_cfg["pretrain_mode"]
        proj_hidden_dim = model_cfg["proj_hidden_dim"]
        proj_out_dim = model_cfg["proj_out_dim"]

        model = SingleModalResNet(
            modality_name=active_modality,
            location_name=location_name,
            in_channels=in_channels,
            num_classes=num_classes,
            layers=layers,
            filter_sizes=filter_sizes,
            fc_dim=fc_dim,
            stem_channels=stem_channels,
            stem_kernel=stem_kernel,
            stem_stride=stem_stride,
            use_maxpool=use_maxpool,
            dropout_ratio=dropout_ratio,
            pretrain_mode=pretrain_mode,
            proj_hidden_dim=proj_hidden_dim,
            proj_out_dim=proj_out_dim,
        )

    elif model_type in ("student_convonly", "convonly"):
        num_blocks = model_cfg["num_blocks"]
        filter_sizes = model_cfg["filter_sizes"]
        kernel_sizes = model_cfg["kernel_sizes"]
        strides = model_cfg["strides"]

        logging.info(f"  ConvOnly blocks: {num_blocks}")
        logging.info(f"  Filter sizes: {filter_sizes}")

        model = SingleModalConvOnly(
            modality_name=active_modality,
            location_name=location_name,
            in_channels=in_channels,
            num_classes=num_classes,
            num_blocks=num_blocks,
            filter_sizes=filter_sizes,
            fc_dim=fc_dim,
            kernel_sizes=kernel_sizes,
            strides=strides,
            stem_channels=stem_channels,
            stem_kernel=stem_kernel,
            stem_stride=stem_stride,
            dropout_ratio=dropout_ratio,
        )

    # elif model_type == "resnet_simple":
    #     logging.info("  Using ResNetSimpleBackbone (no early exits)")
    #     backbone, location_name, active_modality = build_simple_resnet_from_config(config, model_config_key)
    #     model = SingleModalSimpleResNet(
    #         location_name=location_name,
    #         modality_name=active_modality,
    #         backbone=backbone,
    #     )

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: 'student_resnet', 'resnet', 'student_convonly', 'convonly', "
            f"'deepsense', 'deepsense_dw', 'deepsense_dw_simple'"
        )

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    model_size_kb = total_params * 4 / 1024

    logging.info(f"Model created successfully!")
    logging.info(
        f"  Total parameters: {total_params:,} ({total_params / 1e6:.4f}M)"
    )
    logging.info(f"  Trainable parameters: {trainable_params:,}")
    logging.info(
        f"  Estimated size (float32): {model_size_kb:.1f} KB ({model_size_kb / 1024:.2f} MB)"
    )

    return model


def create_deepsense(config, model_config_key):
    """
    Create a SingleModalDeepSense from YAML config.

    Required keys under config["models"][model_config_key]:
        model_type:       "deepsense"
        active_modality:  str
        channels:         list[int]           — output channels per conv layer
        kernel_sizes:     list[[int, int]]    — (freq_k, time_k) per layer
        strides:          list[[int, int]]    — (freq_s, time_s) per layer
                          len(channels) == len(kernel_sizes) == len(strides) required
        recurrent_dim:    int
        recurrent_layers: int
        fc_dim:           int
        dropout_ratio:    float

    in_channels looked up from:
        config["loc_mod_in_freq_channels"][location][modality]
    """
    model_cfg = config["models"][model_config_key]
    active_modality = model_cfg["active_modality"]

    location_names = config["location_names"]
    if len(location_names) != 1:
        raise ValueError(
            f"SingleModalDeepSense expects exactly one location, got {location_names}"
        )
    location_name = location_names[0]

    in_channels = config["loc_mod_in_freq_channels"][location_name][
        active_modality
    ]
    in_spectrum_len = config["loc_mod_spectrum_len"][location_name][
        active_modality
    ]
    num_classes = resolve_num_classes(config, model_cfg)

    channels = model_cfg["channels"]
    kernel_sizes = model_cfg["kernel_sizes"]
    strides = model_cfg["strides"]

    if not (len(channels) == len(kernel_sizes) == len(strides)):
        raise ValueError(
            f"[{model_config_key}] channels, kernel_sizes, and strides must have equal length; "
            f"got {len(channels)}, {len(kernel_sizes)}, {len(strides)}"
        )

    logging.info(f"Creating SingleModalDeepSense ({model_config_key})")
    logging.info(f"  modality={active_modality}, location={location_name}")
    logging.info(
        f"  in_channels={in_channels}, in_spectrum_len={in_spectrum_len}, conv_layers={len(channels)}"
    )
    logging.info(f"  channels={channels}")
    logging.info(f"  kernel_sizes={kernel_sizes}, strides={strides}")

    pretrain_mode = model_cfg["pretrain_mode"]
    proj_hidden_dim = model_cfg["proj_hidden_dim"]
    proj_out_dim = model_cfg["proj_out_dim"]

    model = SingleModalDeepSense(
        modality_name=active_modality,
        location_name=location_name,
        in_channels=in_channels,
        in_spectrum_len=in_spectrum_len,
        num_classes=num_classes,
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        recurrent_dim=model_cfg["recurrent_dim"],
        recurrent_layers=model_cfg["recurrent_layers"],
        fc_dim=model_cfg["fc_dim"],
        dropout_ratio=model_cfg["dropout_ratio"],
        pretrain_mode=pretrain_mode,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  Parameters: {total_params:,} ({total_params / 1e6:.4f}M)")
    return model


def create_deepsense_dw(config, model_config_key):
    """
    Create a SingleModalDeepSenseDW from YAML config.

    Required keys under config["models"][model_config_key]:
        model_type:          "deepsense_dw"
        active_modality:     str
        channels_freq:       list[int]           — output channels per freq conv layer
        kernel_sizes_freq:   list[[int, int]]    — (time_k, freq_k) per layer
        strides_freq:        list[[int, int]]    — (time_s, freq_s) per layer
                             len(channels_freq) == len(kernel_sizes_freq) == len(strides_freq)
        temporal_channels:   int                 — channel dim for temporal DW conv stack
        num_temporal_layers: int
        temporal_kernel:     int
        fc_dim:              int
        dropout_ratio:       float

    in_channels and in_spectrum_len looked up from:
        config["loc_mod_in_freq_channels"][location][modality]
        config["loc_mod_spectrum_len"][location][modality]
    """
    model_cfg = config["models"][model_config_key]
    active_modality = model_cfg["active_modality"]

    location_names = config["location_names"]
    if len(location_names) != 1:
        raise ValueError(
            f"SingleModalDeepSenseDW expects exactly one location, got {location_names}"
        )
    location_name = location_names[0]

    # Allow per-model overrides for in_channels / in_spectrum_len so that a
    # mel-spectrogram model (different shape than FFT) can reuse active_modality
    # "audio" without modifying the dataset-level config dicts.
    if "in_channels" in model_cfg:
        in_channels = model_cfg["in_channels"]
    else:
        in_channels = config["loc_mod_in_freq_channels"][location_name][
            active_modality
        ]
    if "in_spectrum_len" in model_cfg:
        in_spectrum_len = model_cfg["in_spectrum_len"]
    else:
        in_spectrum_len = config["loc_mod_spectrum_len"][location_name][
            active_modality
        ]
    num_classes = resolve_num_classes(config, model_cfg)

    channels_freq = model_cfg["channels_freq"]
    kernel_sizes_freq = model_cfg["kernel_sizes_freq"]
    strides_freq = model_cfg["strides_freq"]

    if not (len(channels_freq) == len(kernel_sizes_freq) == len(strides_freq)):
        raise ValueError(
            f"[{model_config_key}] channels_freq, kernel_sizes_freq, strides_freq must have "
            f"equal length; got {len(channels_freq)}, {len(kernel_sizes_freq)}, {len(strides_freq)}"
        )

    temporal_channels = model_cfg["temporal_channels"]
    num_temporal_layers = model_cfg["num_temporal_layers"]
    temporal_kernel = model_cfg["temporal_kernel"]

    logging.info(f"Creating SingleModalDeepSenseDW ({model_config_key})")
    logging.info(f"  modality={active_modality}, location={location_name}")
    logging.info(
        f"  in_channels={in_channels}, in_spectrum_len={in_spectrum_len}, "
        f"freq_layers={len(channels_freq)}"
    )
    logging.info(f"  channels_freq={channels_freq}")
    logging.info(
        f"  kernel_sizes_freq={kernel_sizes_freq}, strides_freq={strides_freq}"
    )
    logging.info(
        f"  temporal_channels={temporal_channels}, "
        f"num_temporal_layers={num_temporal_layers}, temporal_kernel={temporal_kernel}"
    )

    pretrain_mode = model_cfg["pretrain_mode"]
    proj_hidden_dim = model_cfg["proj_hidden_dim"]
    proj_out_dim = model_cfg["proj_out_dim"]

    use_bigru = False
    if "use_bigru" in model_cfg:
        use_bigru = model_cfg["use_bigru"]
    recurrent_dim = 256
    if "recurrent_dim" in model_cfg:
        recurrent_dim = model_cfg["recurrent_dim"]
    recurrent_layers = 2
    if "recurrent_layers" in model_cfg:
        recurrent_layers = model_cfg["recurrent_layers"]
    output_dims = None
    if "output_dims" in model_cfg:
        output_dims = model_cfg["output_dims"]
    w8a8 = False
    if "w8a8" in model_cfg:
        w8a8 = bool(model_cfg["w8a8"])
    w8a16 = False
    if "w8a16" in model_cfg:
        w8a16 = bool(model_cfg["w8a16"])

    if use_bigru:
        logging.info(
            f"  use_bigru=True recurrent_dim={recurrent_dim} "
            f"recurrent_layers={recurrent_layers}"
        )
    if output_dims is not None:
        logging.info(f"  output_dims={output_dims}")
    if w8a8:
        logging.info("  w8a8=True  (QuantDWConv2d + QuantDWConv1d layers, act_bits=8 — calibrate before QAT)")
    if w8a16:
        logging.info("  w8a16=True (QuantDWConv2d + QuantDWConv1d layers, act_bits=16 — calibrate before QAT)")

    model = SingleModalDeepSenseDW(
        modality_name=active_modality,
        location_name=location_name,
        in_channels=in_channels,
        in_spectrum_len=in_spectrum_len,
        num_classes=num_classes,
        channels_freq=channels_freq,
        kernel_sizes_freq=kernel_sizes_freq,
        strides_freq=strides_freq,
        temporal_channels=temporal_channels,
        num_temporal_layers=num_temporal_layers,
        temporal_kernel=temporal_kernel,
        fc_dim=model_cfg["fc_dim"],
        dropout_ratio=model_cfg["dropout_ratio"],
        pretrain_mode=pretrain_mode,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
        use_bigru=use_bigru,
        recurrent_dim=recurrent_dim,
        recurrent_layers=recurrent_layers,
        output_dims=output_dims,
        w8a8=w8a8,
        w8a16=w8a16,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  Parameters: {total_params:,} ({total_params / 1e6:.4f}M)")
    return model


def create_deepsense_dw_simple(config, model_config_key):
    """
    Create a SingleModalDeepSenseDWSimple from YAML config.

    Compiler-friendly variant: ReLU only, no Dropout, Conv2d-based temporal
    layers (no Conv1d / BN1d), pure Tensor I/O, single execution path.
    Supports the same per-model in_channels / in_spectrum_len overrides as
    create_deepsense_dw.

    Required keys under config["models"][model_config_key]:
        model_type:          "deepsense_dw_simple"
        active_modality:     str
        channels_freq:       list[int]
        kernel_sizes_freq:   list[[int, int]]
        strides_freq:        list[[int, int]]
        temporal_channels:   int
        num_temporal_layers: int
        temporal_kernel:     int
        fc_dim:              int
    """
    model_cfg = config["models"][model_config_key]
    active_modality = model_cfg["active_modality"]

    location_names = config["location_names"]
    if len(location_names) != 1:
        raise ValueError(
            f"SingleModalDeepSenseDWSimple expects exactly one location, got {location_names}"
        )
    location_name = location_names[0]

    if "in_channels" in model_cfg:
        in_channels = model_cfg["in_channels"]
    else:
        in_channels = config["loc_mod_in_freq_channels"][location_name][active_modality]

    if "in_spectrum_len" in model_cfg:
        in_spectrum_len = model_cfg["in_spectrum_len"]
    else:
        in_spectrum_len = config["loc_mod_spectrum_len"][location_name][active_modality]

    num_classes = resolve_num_classes(config, model_cfg)

    channels_freq = model_cfg["channels_freq"]
    kernel_sizes_freq = model_cfg["kernel_sizes_freq"]
    strides_freq = model_cfg["strides_freq"]

    if not (len(channels_freq) == len(kernel_sizes_freq) == len(strides_freq)):
        raise ValueError(
            f"[{model_config_key}] channels_freq, kernel_sizes_freq, strides_freq must have "
            f"equal length; got {len(channels_freq)}, {len(kernel_sizes_freq)}, {len(strides_freq)}"
        )

    logging.info(f"Creating SingleModalDeepSenseDWSimple ({model_config_key})")
    logging.info(f"  modality={active_modality}, location={location_name}")
    logging.info(
        f"  in_channels={in_channels}, in_spectrum_len={in_spectrum_len}, "
        f"freq_layers={len(channels_freq)}"
    )
    logging.info(f"  channels_freq={channels_freq}")
    logging.info(f"  kernel_sizes_freq={kernel_sizes_freq}, strides_freq={strides_freq}")

    backbone = DeepSenseDWSimpleBackbone(
        in_channels=in_channels,
        in_spectrum_len=in_spectrum_len,
        num_classes=num_classes,
        channels_freq=channels_freq,
        kernel_sizes_freq=kernel_sizes_freq,
        strides_freq=strides_freq,
        temporal_channels=model_cfg["temporal_channels"],
        num_temporal_layers=model_cfg["num_temporal_layers"],
        temporal_kernel=model_cfg["temporal_kernel"],
        fc_dim=model_cfg["fc_dim"],
    )

    model = SingleModalDeepSenseDWSimple(
        location_name=location_name,
        modality_name=active_modality,
        backbone=backbone,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  Parameters: {total_params:,} ({total_params / 1e6:.4f}M)")
    return model


def get_parameter_memory(model, unit="MB"):
    """
    Calculate memory used by model parameters (weights and biases).

    Args:
        model: PyTorch model
        unit: 'B', 'KB', or 'MB'

    Returns:
        float: Parameter memory in specified unit
    """
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params * 4  # float32

    if unit == "KB":
        return size_bytes / 1024
    elif unit == "MB":
        return size_bytes / (1024 * 1024)
    return size_bytes


def get_activation_memory(model, input_dict, unit="MB"):
    """
    Estimate peak activation/feature map memory during forward pass (normalized to batch size 1).

    Returns the size of the LARGEST single activation tensor (the peak),
    not the sum of all activations, since PyTorch frees tensors as they
    go out of scope. Memory is normalized to batch size 1.

    Args:
        model: PyTorch model
        input_dict: Input dict format: {'location': {'modality': tensor}}
                   e.g., {'shake': {'audio': torch.randn(B, C, H, W)}}
        unit: 'B', 'KB', or 'MB'

    Returns:
        float: Peak activation memory per sample in specified unit
    """
    import torch

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Get batch size from input
    batch_size = None
    for loc in input_dict:
        for mod in input_dict[loc]:
            batch_size = input_dict[loc][mod].shape[0]
            break
        if batch_size is not None:
            break

    # Track the largest activation tensor
    peak_activation_bytes = 0

    def hook_fn(module, input, output):
        nonlocal peak_activation_bytes

        # Handle different output types
        tensors_to_check = []

        if isinstance(output, torch.Tensor):
            tensors_to_check.append(output)
        elif isinstance(output, dict):
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    tensors_to_check.append(v)
                elif isinstance(v, list):
                    tensors_to_check.extend(
                        [t for t in v if isinstance(t, torch.Tensor)]
                    )
        elif isinstance(output, (list, tuple)):
            tensors_to_check.extend(
                [t for t in output if isinstance(t, torch.Tensor)]
            )

        # Find peak among current tensors
        for tensor in tensors_to_check:
            tensor_bytes = tensor.numel() * tensor.element_size()
            peak_activation_bytes = max(peak_activation_bytes, tensor_bytes)

    # Register hooks on all modules
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    # Run forward pass
    with torch.no_grad():
        _ = model(input_dict)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Normalize to batch size 1
    if batch_size and batch_size > 0:
        peak_activation_bytes = peak_activation_bytes / batch_size

    if unit == "KB":
        return peak_activation_bytes / 1024
    elif unit == "MB":
        return peak_activation_bytes / (1024 * 1024)
    return peak_activation_bytes


def get_total_memory(model, input_dict, unit="MB"):
    """
    Get total peak memory usage: parameters + peak activations.

    Args:
        model: PyTorch model
        input_dict: Input dict format: {'location': {'modality': tensor}}
        unit: 'B', 'KB', or 'MB'

    Returns:
        dict: {
            'parameter_memory': float,
            'activation_memory': float,  # Peak activation, not sum
            'total_memory': float,
            'unit': str
        }
    """
    param_mem = get_parameter_memory(model, unit=unit)
    act_mem = get_activation_memory(model, input_dict, unit=unit)
    total_mem = param_mem + act_mem

    return {
        "parameter_memory": param_mem,
        "activation_memory": act_mem,
        "total_memory": total_mem,
        "unit": unit,
    }


def get_input_memory(input_dict, unit="KB"):
    """
    Calculate memory requirements for input data (normalized to batch size 1).

    Args:
        input_dict: Input dict format: {'location': {'modality': tensor}}
                   e.g., {'shake': {'audio': torch.randn(B, C, H, W)}}
        unit: 'B', 'KB', or 'MB' (default: 'KB')

    Returns:
        dict: {
            'shape_info': list of dicts with location, modality, shape info,
            'total_memory': float (total input memory per sample),
            'unit': str,
            'batch_size': int
        }
    """
    import torch

    def convert_bytes(size_bytes):
        if unit == "KB":
            return size_bytes / 1024
        elif unit == "MB":
            return size_bytes / (1024 * 1024)
        return size_bytes

    shape_info = []
    total_bytes = 0
    batch_size = None

    for loc in input_dict:
        for mod in input_dict[loc]:
            tensor = input_dict[loc][mod]

            # Get batch size from first dimension
            if batch_size is None:
                batch_size = tensor.shape[0]

            # Calculate total bytes and normalize to batch size 1
            tensor_bytes = tensor.numel() * tensor.element_size()
            tensor_bytes_per_sample = tensor_bytes / batch_size
            total_bytes += tensor_bytes_per_sample

            shape_info.append(
                {
                    "location": loc,
                    "modality": mod,
                    "shape": list(tensor.shape),
                    "memory": convert_bytes(tensor_bytes_per_sample),
                }
            )

    return {
        "shape_info": shape_info,
        "total_memory": convert_bytes(total_bytes),
        "unit": unit,
        "batch_size": batch_size,
    }


def log_memory_info(memory_info, input_memory_info=None, logger=None):
    """
    Log model memory information in a readable format.
    All memory values are per-sample (batch size 1).

    Args:
        memory_info: Dict from get_total_memory()
        input_memory_info: Dict from get_input_memory() (optional)
        logger: Logger instance (if None, uses logging.info)
    """
    if logger is None:
        log_fn = logging.info
    else:
        log_fn = logger.info

    unit = memory_info["unit"]

    # Log input information first if provided
    if input_memory_info is not None:
        log_fn("=" * 80)
        log_fn("INPUT DATA INFORMATION (Per Sample, Batch Size 1)")
        log_fn("=" * 80)
        if (
            "batch_size" in input_memory_info
            and input_memory_info["batch_size"]
        ):
            log_fn(f"  Original batch size: {input_memory_info['batch_size']}")
        for info in input_memory_info["shape_info"]:
            # Show original shape with batch dimension
            shape_str = "x".join(map(str, info["shape"]))
            # Calculate per-sample shape (remove batch dimension for display)
            per_sample_shape = (
                info["shape"][1:] if len(info["shape"]) > 1 else info["shape"]
            )
            per_sample_shape_str = "x".join(map(str, per_sample_shape))
            log_fn(
                f"  {info['location']}/{info['modality']}: "
                f"batch_shape={shape_str}, per_sample_shape={per_sample_shape_str}, "
                f"memory={info['memory']:.2f} {input_memory_info['unit']}"
            )
        log_fn(
            f"  Total Input Memory (per sample): {input_memory_info['total_memory']:.2f} {input_memory_info['unit']}"
        )
        log_fn("=" * 80)
        log_fn("")

    log_fn("=" * 80)
    log_fn("MODEL MEMORY REQUIREMENTS (Per Sample, Batch Size 1)")
    log_fn("=" * 80)
    log_fn(f"  Parameters: {memory_info['parameter_memory']:.2f} {unit}")
    log_fn(
        f"  Activations (per sample): {memory_info['activation_memory']:.2f} {unit}"
    )
    log_fn(f"  Total (per sample): {memory_info['total_memory']:.2f} {unit}")
    log_fn("=" * 80)


def get_model_config(config, model_config_key):
    """
    Extract and validate model configuration.

    Useful for inspecting what model would be created before actually creating it.

    Args:
        config: Full dataset config
        model_config_key: Key for model config section

    Returns:
        dict: Validated model configuration with all defaults filled in
    """
    model_cfg = config[model_config_key]
    model_type = model_cfg["model_type"]
    active_modality = model_cfg["active_modality"]
    location_name = config["location_names"][0]

    result = {
        "model_type": model_type,
        "active_modality": active_modality,
        "location_name": location_name,
        "fc_dim": model_cfg["fc_dim"],
        "dropout_ratio": model_cfg["dropout_ratio"],
        "stem_kernel": model_cfg["stem_kernel"],
        "stem_stride": model_cfg["stem_stride"],
    }

    if model_type in ("student_resnet", "resnet"):
        result.update(
            {
                "layers": model_cfg["layers"],
                "filter_sizes": model_cfg["filter_sizes"],
                "use_maxpool": model_cfg["use_maxpool"],
            }
        )
    elif model_type in ("student_convonly", "convonly"):
        result.update(
            {
                "num_blocks": model_cfg["num_blocks"],
                "filter_sizes": model_cfg["filter_sizes"],
                "kernel_sizes": model_cfg["kernel_sizes"],
                "strides": model_cfg.get("strides", None),
            }
        )

    return result
