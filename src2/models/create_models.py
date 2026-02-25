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





# =============================================================================
# Student Model Factory for Distillation
# =============================================================================

def create_single_modal_model(config, model_config_key):
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
        - Only 3 keys allow .get() with defaults:
          1. early_exits: defaults to [] (no early exits is valid)
          2. stem_channels: defaults to None (model will infer from filter_sizes)
          3. kernel_sizes/strides (ConvOnly): defaults to None (model has internal defaults)
        - All other keys MUST be explicitly present in the config
    
    Args:
        config: Full dataset configuration dictionary (e.g., loaded from ACIDS.yaml)
        model_config_key: Key in config for model settings
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
    model_cfg = config[model_config_key]
    model_type = model_cfg["model_type"]
    
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
    
    # Classification parameters
    task_cfg = config["vehicle_classification"]
    num_classes = task_cfg["num_classes"]
    
    # Common parameters
    fc_dim = model_cfg["fc_dim"]
    dropout_ratio = model_cfg["dropout_ratio"]
    early_exit_layers = model_cfg.get("early_exits", [])  # OK to default to empty list
    stem_kernel = model_cfg["stem_kernel"]
    stem_stride = model_cfg["stem_stride"]
    stem_channels = model_cfg.get("stem_channels", None)  # OK to be None
    
    logging.info(f"Creating single-modal model: {model_type} ({model_config_key})")
    logging.info(f"  Modality: {active_modality}")
    logging.info(f"  Location: {location_name}")
    logging.info(f"  Input channels: {in_channels}")
    logging.info(f"  Number of classes: {num_classes}")
    logging.info(f"  FC dim: {fc_dim}")
    logging.info(f"  Early exits at layers: {early_exit_layers}")
    
    if model_type in ("student_resnet", "resnet"):
        layers = model_cfg["layers"]
        filter_sizes = model_cfg["filter_sizes"]
        use_maxpool = model_cfg["use_maxpool"]
        
        logging.info(f"  ResNet layers: {layers}")
        logging.info(f"  Filter sizes: {filter_sizes}")
        
        model = SingleModalResNet(
            modality_name=active_modality,
            location_name=location_name,
            in_channels=in_channels,
            num_classes=num_classes,
            layers=layers,
            filter_sizes=filter_sizes,
            fc_dim=fc_dim,
            early_exit_layers=early_exit_layers,
            stem_channels=stem_channels,
            stem_kernel=stem_kernel,
            stem_stride=stem_stride,
            use_maxpool=use_maxpool,
            dropout_ratio=dropout_ratio,
        )
    
    elif model_type in ("student_convonly", "convonly"):
        num_blocks = model_cfg["num_blocks"]
        filter_sizes = model_cfg["filter_sizes"]
        kernel_sizes = model_cfg.get("kernel_sizes", None)  # OK to be None (has internal default)
        strides = model_cfg.get("strides", None)  # OK to be None (has internal default)
        early_exit_type = model_cfg["early_exit_type"]
        
        logging.info(f"  ConvOnly blocks: {num_blocks}")
        logging.info(f"  Filter sizes: {filter_sizes}")
        logging.info(f"  Early exit type: {early_exit_type}")
        
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
            early_exit_layers=early_exit_layers,
            early_exit_type=early_exit_type,
        )
    
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: 'student_resnet', 'resnet', 'student_convonly', 'convonly'"
        )
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_kb = total_params * 4 / 1024
    
    logging.info(f"Model created successfully!")
    logging.info(f"  Total parameters: {total_params:,} ({total_params / 1e6:.4f}M)")
    logging.info(f"  Trainable parameters: {trainable_params:,}")
    logging.info(f"  Estimated size (float32): {model_size_kb:.1f} KB ({model_size_kb / 1024:.2f} MB)")
    
    return model


def get_parameter_memory(model, unit='MB'):
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
    
    if unit == 'KB':
        return size_bytes / 1024
    elif unit == 'MB':
        return size_bytes / (1024 * 1024)
    return size_bytes


def get_activation_memory(model, input_dict, unit='MB'):
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
    
    device = torch.device('cpu')
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
                    tensors_to_check.extend([t for t in v if isinstance(t, torch.Tensor)])
        elif isinstance(output, (list, tuple)):
            tensors_to_check.extend([t for t in output if isinstance(t, torch.Tensor)])
        
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
    
    if unit == 'KB':
        return peak_activation_bytes / 1024
    elif unit == 'MB':
        return peak_activation_bytes / (1024 * 1024)
    return peak_activation_bytes


def get_total_memory(model, input_dict, unit='MB'):
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
        'parameter_memory': param_mem,
        'activation_memory': act_mem,
        'total_memory': total_mem,
        'unit': unit
    }


def get_early_exit_memory(model, input_dict, unit='KB'):
    """
    Calculate per-exit memory requirements for early exit models.
    
    For models with early exits, computes parameter and activation memory
    for each exit point (if I exit at exit 1, how much memory is needed?).
    
    Args:
        model: PyTorch model (SingleModalResNet or SingleModalConvOnly)
        input_dict: Sample input dict format: {'location': {'modality': tensor}}
                   e.g., {'shake': {'audio': torch.randn(B, C, H, W)}}
                   MUST be actual augmented data (after apply_augmentation)
        unit: 'B', 'KB', or 'MB' (default: 'KB')
    
    Returns:
        dict: {
            'has_early_exits': bool,
            'exits': [
                {'exit_index': int, 'parameter_memory': float, 
                 'activation_memory': float, 'total_memory': float},
                ...
            ],
            'final': {'parameter_memory': float, 'activation_memory': float, 
                     'total_memory': float},
            'unit': str
        }
    """
    import torch
    original_device = next(model.parameters()).device
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # Move input to CPU
    input_cpu = {}
    for loc in input_dict:
        input_cpu[loc] = {}
        for mod in input_dict[loc]:
            input_cpu[loc][mod] = input_dict[loc][mod].to(device)
    
    # Access backbone (works for both SingleModalResNet and SingleModalConvOnly)
    backbone = model.backbone
    early_exit_layers = backbone.early_exit_layers
    
    has_early_exits = len(early_exit_layers) > 0
    
    # Helper to count parameters in a list of modules
    def count_params(modules):
        total = 0
        for module in modules:
            total += sum(p.numel() for p in module.parameters())
        return total * 4  # float32 bytes
    
    # Helper to convert bytes to target unit
    def convert_bytes(size_bytes):
        if unit == 'KB':
            return size_bytes / 1024
        elif unit == 'MB':
            return size_bytes / (1024 * 1024)
        return size_bytes
    
    result = {
        'has_early_exits': has_early_exits,
        'exits': [],
        'final': {},
        'unit': unit
    }
    
    if not has_early_exits:
        # No early exits - just compute total memory
        param_mem = get_parameter_memory(model, unit=unit)
        act_mem = get_activation_memory(model, input_cpu, unit=unit)
        result['final'] = {
            'parameter_memory': param_mem,
            'activation_memory': act_mem,
            'total_memory': param_mem + act_mem
        }
        return result
    
    # === Calculate parameter memory per exit ===
    # Stem parameters (shared by all exits)
    # Handle both ConvOnly (has .stem) and ResNet (has .conv1, .bn1, .relu, .maxpool)
    if hasattr(backbone, 'stem'):
        # ConvOnly model
        stem_params = count_params([backbone.stem])
    else:
        # ResNet model
        stem_modules = [backbone.conv1, backbone.bn1, backbone.relu]
        if hasattr(backbone, 'maxpool'):
            stem_modules.append(backbone.maxpool)
        stem_params = count_params(stem_modules)
    
    # For each early exit
    for exit_idx, layer_idx in enumerate(early_exit_layers):
        # Parameters: stem + stages[0..layer_idx] + exit_branch
        stage_params = count_params(backbone.stages[:layer_idx + 1])
        exit_branch_params = count_params([backbone.exit_branches[str(layer_idx)]])
        total_params = stem_params + stage_params + exit_branch_params
        
        result['exits'].append({
            'exit_index': exit_idx,
            'parameter_memory': convert_bytes(total_params),
            'activation_memory': 0.0,  # Will be filled below
            'total_memory': 0.0  # Will be filled below
        })
    
    # Final exit parameters: all stages + final head
    final_stage_params = count_params(backbone.stages)
    final_head_params = count_params([backbone.avgpool, backbone.embed, backbone.classifier])
    final_total_params = stem_params + final_stage_params + final_head_params
    
    result['final'] = {
        'parameter_memory': convert_bytes(final_total_params),
        'activation_memory': 0.0,  # Will be filled below
        'total_memory': 0.0  # Will be filled below
    }
    
    # === Calculate activation memory per exit ===
    # We need to track peak activation for each exit path separately
    # Activations will be normalized to batch size 1
    
    # Get batch size from input
    batch_size = None
    for loc in input_cpu:
        for mod in input_cpu[loc]:
            batch_size = input_cpu[loc][mod].shape[0]
            break
        if batch_size is not None:
            break
    
    # Track activations per module
    module_activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            # Handle different output types
            tensors_to_check = []
            
            if isinstance(output, torch.Tensor):
                tensors_to_check.append(output)
            elif isinstance(output, dict):
                for v in output.values():
                    if isinstance(v, torch.Tensor):
                        tensors_to_check.append(v)
                    elif isinstance(v, list):
                        tensors_to_check.extend([t for t in v if isinstance(t, torch.Tensor)])
            elif isinstance(output, (list, tuple)):
                tensors_to_check.extend([t for t in output if isinstance(t, torch.Tensor)])
            
            # Find largest tensor for this module
            max_bytes = 0
            for tensor in tensors_to_check:
                tensor_bytes = tensor.numel() * tensor.element_size()
                max_bytes = max(max_bytes, tensor_bytes)
            
            module_activations[name] = max_bytes
        return hook
    
    # Register hooks on all relevant modules
    hooks = []
    
    # Register hook on stem (either as single module or individual modules)
    if hasattr(backbone, 'stem'):
        # ConvOnly model
        hooks.append(backbone.stem.register_forward_hook(hook_fn('stem')))
    else:
        # ResNet model - register on individual stem modules
        hooks.append(backbone.conv1.register_forward_hook(hook_fn('stem_conv1')))
        hooks.append(backbone.bn1.register_forward_hook(hook_fn('stem_bn1')))
        hooks.append(backbone.relu.register_forward_hook(hook_fn('stem_relu')))
        if hasattr(backbone, 'maxpool'):
            hooks.append(backbone.maxpool.register_forward_hook(hook_fn('stem_maxpool')))
    
    for i, stage in enumerate(backbone.stages):
        hooks.append(stage.register_forward_hook(hook_fn(f'stage_{i}')))
    
    for layer_idx in early_exit_layers:
        exit_branch = backbone.exit_branches[str(layer_idx)]
        hooks.append(exit_branch.register_forward_hook(hook_fn(f'exit_branch_{layer_idx}')))
    
    hooks.append(backbone.avgpool.register_forward_hook(hook_fn('avgpool')))
    hooks.append(backbone.embed.register_forward_hook(hook_fn('embed')))
    hooks.append(backbone.classifier.register_forward_hook(hook_fn('classifier')))
    
    # Run forward pass
    with torch.no_grad():
        _ = model(input_cpu)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate peak activation for each exit (normalized to batch size 1)
    for exit_idx, layer_idx in enumerate(early_exit_layers):
        # Peak for this exit: max among stem modules, stages[0..layer_idx], exit_branch
        # Handle both stem types
        if hasattr(backbone, 'stem'):
            stem_module_names = ['stem']
        else:
            stem_module_names = ['stem_conv1', 'stem_bn1', 'stem_relu']
            if hasattr(backbone, 'maxpool'):
                stem_module_names.append('stem_maxpool')
        
        relevant_modules = stem_module_names + [f'stage_{i}' for i in range(layer_idx + 1)] + [f'exit_branch_{layer_idx}']
        peak_bytes = max(module_activations.get(name, 0) for name in relevant_modules)
        
        # Normalize to batch size 1
        if batch_size and batch_size > 0:
            peak_bytes = peak_bytes / batch_size
        
        result['exits'][exit_idx]['activation_memory'] = convert_bytes(peak_bytes)
        result['exits'][exit_idx]['total_memory'] = (
            result['exits'][exit_idx]['parameter_memory'] + 
            result['exits'][exit_idx]['activation_memory']
        )
    
    # Calculate peak activation for final exit (normalized to batch size 1)
    if hasattr(backbone, 'stem'):
        stem_module_names = ['stem']
    else:
        stem_module_names = ['stem_conv1', 'stem_bn1', 'stem_relu']
        if hasattr(backbone, 'maxpool'):
            stem_module_names.append('stem_maxpool')
    
    final_modules = stem_module_names + [f'stage_{i}' for i in range(len(backbone.stages))] + ['avgpool', 'embed', 'classifier']
    final_peak_bytes = max(module_activations.get(name, 0) for name in final_modules)
    
    # Normalize to batch size 1
    if batch_size and batch_size > 0:
        final_peak_bytes = final_peak_bytes / batch_size
    
    result['final']['activation_memory'] = convert_bytes(final_peak_bytes)
    result['final']['total_memory'] = (
        result['final']['parameter_memory'] + 
        result['final']['activation_memory']
    )
    
    model = model.to(original_device)
    return result


def get_input_memory(input_dict, unit='KB'):
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
        if unit == 'KB':
            return size_bytes / 1024
        elif unit == 'MB':
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
            
            shape_info.append({
                'location': loc,
                'modality': mod,
                'shape': list(tensor.shape),
                'memory': convert_bytes(tensor_bytes_per_sample)
            })
    
    return {
        'shape_info': shape_info,
        'total_memory': convert_bytes(total_bytes),
        'unit': unit,
        'batch_size': batch_size
    }


def log_memory_info(memory_info, input_memory_info=None, logger=None):
    """
    Log per-exit memory information in a readable table format.
    All memory values are per-sample (batch size 1).
    
    Args:
        memory_info: Dict from get_early_exit_memory()
        input_memory_info: Dict from get_input_memory() (optional)
        logger: Logger instance (if None, uses logging.info)
    """
    if logger is None:
        log_fn = logging.info
    else:
        log_fn = logger.info
    
    unit = memory_info['unit']
    
    # Log input information first if provided
    if input_memory_info is not None:
        log_fn("=" * 80)
        log_fn("INPUT DATA INFORMATION (Per Sample, Batch Size 1)")
        log_fn("=" * 80)
        if 'batch_size' in input_memory_info and input_memory_info['batch_size']:
            log_fn(f"  Original batch size: {input_memory_info['batch_size']}")
        for info in input_memory_info['shape_info']:
            # Show original shape with batch dimension
            shape_str = "x".join(map(str, info['shape']))
            # Calculate per-sample shape (remove batch dimension for display)
            per_sample_shape = info['shape'][1:] if len(info['shape']) > 1 else info['shape']
            per_sample_shape_str = "x".join(map(str, per_sample_shape))
            log_fn(f"  {info['location']}/{info['modality']}: "
                   f"batch_shape={shape_str}, per_sample_shape={per_sample_shape_str}, "
                   f"memory={info['memory']:.2f} {input_memory_info['unit']}")
        log_fn(f"  Total Input Memory (per sample): {input_memory_info['total_memory']:.2f} {input_memory_info['unit']}")
        log_fn("=" * 80)
        log_fn("")
    
    if not memory_info['has_early_exits']:
        log_fn("=" * 80)
        log_fn("MODEL MEMORY REQUIREMENTS (Per Sample, No Early Exits)")
        log_fn("=" * 80)
        final = memory_info['final']
        log_fn(f"  Parameters: {final['parameter_memory']:.2f} {unit}")
        log_fn(f"  Activations (per sample): {final['activation_memory']:.2f} {unit}")
        log_fn(f"  Total (per sample): {final['total_memory']:.2f} {unit}")
        log_fn("=" * 80)
        return
    
    log_fn("=" * 80)
    log_fn("MODEL MEMORY REQUIREMENTS (Per Sample, Batch Size 1)")
    log_fn("=" * 80)
    log_fn(f"{'Exit':<10} {'Parameters':<15} {'Activations':<15} {'Total':<15}")
    log_fn(f"{'':10} {'':15} {'(per sample)':<15} {'(per sample)':<15}")
    log_fn("-" * 80)
    
    for exit_data in memory_info['exits']:
        exit_num = exit_data['exit_index'] + 1
        log_fn(f"Exit {exit_num:<5} "
               f"{exit_data['parameter_memory']:>10.2f} {unit:<4} "
               f"{exit_data['activation_memory']:>10.2f} {unit:<4} "
               f"{exit_data['total_memory']:>10.2f} {unit:<4}")
    
    final = memory_info['final']
    log_fn(f"Final{' ' * 5} "
           f"{final['parameter_memory']:>10.2f} {unit:<4} "
           f"{final['activation_memory']:>10.2f} {unit:<4} "
           f"{final['total_memory']:>10.2f} {unit:<4}")
    
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
        "early_exits": model_cfg.get("early_exits", []),
        "stem_kernel": model_cfg["stem_kernel"],
        "stem_stride": model_cfg["stem_stride"],
    }
    
    if model_type in ("student_resnet", "resnet"):
        result.update({
            "layers": model_cfg["layers"],
            "filter_sizes": model_cfg["filter_sizes"],
            "use_maxpool": model_cfg["use_maxpool"],
        })
    elif model_type in ("student_convonly", "convonly"):
        result.update({
            "num_blocks": model_cfg["num_blocks"],
            "filter_sizes": model_cfg["filter_sizes"],
            "kernel_sizes": model_cfg.get("kernel_sizes", None),
            "strides": model_cfg.get("strides", None),
            "early_exit_type": model_cfg["early_exit_type"],
        })
    
    return result

