import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class LayerCost:
    """Cost metrics for a single layer."""
    name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    weight_shape: Optional[Tuple[int, ...]]
    stride: int
    lower_bitwidth: int
    upper_bitwidth: int
    bitops_gb: float
    bitops_baseline_gb: float
    # speedup: float
    # weight_memory_mb: float
    # weight_memory_baseline_mb: float
    # compression: float

@dataclass
class ModalityCost:
    """Cost metrics for a single modality."""
    modality_name: str
    layer_range: Tuple[int, int]  # (start, end) e.g., (1, 19)
    num_layers: int
    total_bitops_gb: float
    total_bitops_baseline_gb: float

@dataclass
class NetworkCost:
    """Cost metrics for entire network."""
    layer_costs: Dict[str, LayerCost]
    total_bitops_gb: float
    total_bitops_baseline_gb: float
    # overall_speedup: float
    # total_weight_memory_mb: float
    # total_weight_memory_baseline_mb: float
    # overall_compression: float
    # avg_bitwidth_weights: float
    # avg_bitwidth_activations: float
    # requantization_overhead_gb: float
    # total_with_overhead_gb: float
    # speedup_with_overhead: float
    modality_costs: Optional[Dict[str, 'ModalityCost']] = None
    layers_per_modality: int = 0
    


def extract_modality_from_name(layer_name: str) -> Optional[str]:
    """
    Extract modality from layer name.
    'mod_loc_backbones.loc1.acoustic.layer1.0.conv1' -> 'acoustic'
    """
    parts = layer_name.split('.')
    if len(parts) >= 3 and parts[0] == 'mod_loc_backbones':
        return parts[2]  # modality is the 3rd part
    return None


def compute_conv_bitops_uniform(
    layer: nn.Module,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    bitwidth: int,
    baseline_bitwidth: int = 8
) -> Dict[str, float]:
    """
    Compute BitOPs for a single conv layer with uniform quantization (no frequency split).
    
    Uses standard BitOPs metric: BitOPs = b_w × b_a × |f| × h × w / s²
    
    Args:
        layer: Conv2d layer
        input_shape: (B, C_in, H_in, W_in)
        output_shape: (B, C_out, H_out, W_out)
        bitwidth: Uniform bitwidth for the entire layer
        baseline_bitwidth: Baseline bitwidth for comparison (8 or 16)
    
    Returns:
        Dictionary with BitOPs metrics
    """
    # Extract layer properties
    C_out = layer.out_channels
    C_in = layer.in_channels
    K_h, K_w = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
    stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
    stride = stride[0]  # Assume square stride
    
    # Input spatial dimensions
    _, _, H_in, W_in = input_shape
    
    # Filter cardinality
    filter_size = C_out * C_in * K_h * K_w
    
    # Uniform BitOPs computation (no split)
    bitops_total = (bitwidth ** 2) * filter_size * H_in * W_in / (stride ** 2)
    
    # Baseline computation
    bitops_baseline = (baseline_bitwidth ** 2) * filter_size * H_in * W_in / (stride ** 2)
    
    return {
        'bitops': bitops_total,
        'bitops_baseline': bitops_baseline,
        'avg_bitwidth': bitwidth
    }

# def compute_fc_bitops(
#     layer: nn.Module,
#     input_shape: Tuple[int, ...],
#     output_shape: Tuple[int, ...],
#     bitwidth: int = 4,  # Typically use single precision for FC
#     baseline_bitwidth: int = 8
# ) -> Dict[str, float]:
#     """
#     Compute BitOPs for a fully connected layer.
    
#     Args:
#         layer: Linear layer
#         input_shape: (B, features_in) or (B, C, H, W)
#         output_shape: (B, features_out)
#         bitwidth: Bitwidth for quantization
#         baseline_bitwidth: Baseline bitwidth
    
#     Returns:
#         Dictionary with BitOPs metrics
#     """
#     in_features = layer.in_features
#     out_features = layer.out_features
#     batch_size = output_shape[0]
    
#     # BitOPs for FC layer
#     ops = in_features * out_features * batch_size
#     bitops_total = (bitwidth ** 2) * ops
#     bitops_baseline = (baseline_bitwidth ** 2) * ops
    
#     # Weight memory
#     # weight_mem = in_features * out_features * bitwidth / 8
#     # weight_mem_baseline = in_features * out_features * baseline_bitwidth / 8
    
#     return {
#         'bitops': bitops_total,
#         'bitops_baseline': bitops_baseline,
#         # 'speedup': bitops_baseline / bitops_total if bitops_total > 0 else 0,
#         # 'weight_memory': weight_mem,
#         # 'weight_memory_baseline': weight_mem_baseline,
#         # 'compression': weight_mem_baseline / weight_mem if weight_mem > 0 else 0,
#         'avg_bitwidth': bitwidth
#     }

def get_output_shape(layer: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute output shape for a layer given input shape.
    
    Args:
        layer: PyTorch layer
        input_shape: Input tensor shape
    
    Returns:
        Output tensor shape
    """
    with torch.no_grad():
        dummy_input = torch.zeros(input_shape)
        try:
            output = layer(dummy_input)
            return tuple(output.shape)
        except:

            raise ValueError(f"Failed to get output shape for layer: {layer.__class__.__name__}")
        

def analyze_resnet_cost(
    model: nn.Module,
    sample_input: torch.Tensor,
    baseline_bitwidth: int = 8,
    verbose: bool = True
) -> NetworkCost:
    """
    Analyze the inference cost of a ResNet model with frequency-split quantization.
    
    This function extracts bitwidth information from conv layers (assumes each Conv2d 
    has .upper_bitwidth and .lower_bitwidth attributes) and computes the total 
    inference cost relative to a baseline precision.
    
    Args:
        model: PyTorch ResNet model with conv layers having .upper_bitwidth and 
               .lower_bitwidth attributes
        sample_input: Sample input tensor (e.g., [batch_size, channels, height, width])
        baseline_bitwidth: Baseline bitwidth for comparison (default: 8-bit, can use 16)
        verbose: If True, print detailed layer-by-layer breakdown
    
    Returns:
        NetworkCost object containing all cost metrics
    
    Example:
        >>> model = resnet18()
        >>> # Assuming each conv layer has .upper_bitwidth and .lower_bitwidth set
        >>> sample_input = torch.randn(128, 2, 10, 800)
        >>> cost = analyze_resnet_cost(model, sample_input, baseline_bitwidth=8)
        >>> print(f"Overall speedup: {cost.overall_speedup:.2f}x")
        >>> print(f"Model compression: {cost.overall_compression:.2f}x")
    """
    
    model.eval()
    layer_costs = {}
    
    # Track activations through the network
    activation_shapes = {}
    hooks = []
    
    def get_activation_hook(name):
        def hook(module, input, output):
            activation_shapes[name] = {
                'input': tuple(input[0].shape) if isinstance(input, tuple) else tuple(input.shape),
                'output': tuple(output.shape)
            }
        return hook
    
    # Register hooks for all Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(get_activation_hook(name)))
    
    # Forward pass to get shapes
    with torch.no_grad():
        _ = model(sample_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute costs for each layer
    total_bitops = 0
    total_bitops_baseline = 0
    # total_weight_mem = 0
    # total_weight_mem_baseline = 0
    # total_elements = 0  # For average bitwidth calculation
    # total_bitwidth_weighted = 0

    # Track by modality
    modality_layer_costs = {}  # {modality_name: [LayerCost, ...]}
    modality_order = []        # Track order modalities are seen
    
    for name, module in model.named_modules():
        if name not in activation_shapes:
            continue
        
        shapes = activation_shapes[name]
        input_shape = shapes['input']
        output_shape = shapes['output']
        
        if isinstance(module, nn.Conv2d):
            # Get bitwidth from module attributes (no split, use upper only)
            if hasattr(module, 'upper_bitwidth'):
                # lower_bits = module.lower_bitwidth
                upper_bits = module.upper_bitwidth
            else:
                # Default to 4-bit if not specified
                # lower_bits = 2
                upper_bits = 4
                if verbose:
                    print(f"Warning: {name} missing bitwidth attributes, using default 4-bit")
            
            # Compute cost (no split, pass upper_bits for both)
            cost_dict = compute_conv_bitops(
                module, input_shape, output_shape,
                upper_bits, upper_bits, baseline_bitwidth
            )
            
            # Create LayerCost object (no split, same bitwidth for both)
            layer_cost = LayerCost(
                name=name,
                layer_type='Conv2d',
                input_shape=input_shape,
                output_shape=output_shape,
                weight_shape=tuple(module.weight.shape),
                stride=module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                lower_bitwidth=upper_bits,  # No split, same as upper
                upper_bitwidth=upper_bits,
                bitops_gb=cost_dict['bitops'] / 1e9,
                bitops_baseline_gb=cost_dict['bitops_baseline'] / 1e9,
                # speedup=cost_dict['speedup'],
                # weight_memory_mb=cost_dict['weight_memory'] / 1e6,
                # weight_memory_baseline_mb=cost_dict['weight_memory_baseline'] / 1e6,
                # compression=cost_dict['compression']
            )
            
            layer_costs[name] = layer_cost
                        # Track by modality
            modality = extract_modality_from_name(name)
            if modality:
                if modality not in modality_layer_costs:
                    modality_layer_costs[modality] = []
                    modality_order.append(modality)
                modality_layer_costs[modality].append(layer_cost)

            total_bitops += cost_dict['bitops']
            total_bitops_baseline += cost_dict['bitops_baseline']
            # total_weight_mem += cost_dict['weight_memory']
            # total_weight_mem_baseline += cost_dict['weight_memory_baseline']
            
            # For average bitwidth calculation
            # num_elements = np.prod(module.weight.shape)
            # total_elements += num_elements
            # total_bitwidth_weighted += num_elements * cost_dict['avg_bitwidth']
            
        # elif isinstance(module, nn.Linear):
        #     # FC layers typically use single precision
        #     cost_dict = compute_fc_bitops(
        #         module, input_shape, output_shape,
        #         fc_bitwidth, baseline_bitwidth
        #     )
            
        #     layer_cost = LayerCost(
        #         name=name,
        #         layer_type='Linear',
        #         input_shape=input_shape,
        #         output_shape=output_shape,
        #         weight_shape=tuple(module.weight.shape),
        #         stride=1,
        #         lower_bitwidth=fc_bitwidth,
        #         upper_bitwidth=fc_bitwidth,
        #         bitops_gb=cost_dict['bitops'] / 1e9,
        #         bitops_baseline_gb=cost_dict['bitops_baseline'] / 1e9,
        #         # speedup=cost_dict['speedup'],
        #         # weight_memory_mb=cost_dict['weight_memory'] / 1e6,
        #         # weight_memory_baseline_mb=cost_dict['weight_memory_baseline'] / 1e6,
        #         # compression=cost_dict['compression']
        #     )
            
        #     layer_costs[name] = layer_cost

        #                 # Track by modality
        #     modality = extract_modality_from_name(name)
        #     if modality:
        #         if modality not in modality_layer_costs:
        #             modality_layer_costs[modality] = []
        #             modality_order.append(modality)
        #         modality_layer_costs[modality].append(layer_cost)
            
        #     total_bitops += cost_dict['bitops']
        #     total_bitops_baseline += cost_dict['bitops_baseline']
        #     # total_weight_mem += cost_dict['weight_memory']
        #     # total_weight_mem_baseline += cost_dict['weight_memory_baseline']
            
        #     # num_elements = np.prod(module.weight.shape)
        #     # total_elements += num_elements
        #     # total_bitwidth_weighted += num_elements * fc_bitwidth
    

            # Compute per-modality costs
    modality_costs = {}
    layer_counter = 0
    
    for modality in modality_order:
        mod_layers = modality_layer_costs.get(modality, [])
        if not mod_layers:
            continue
        
        mod_bitops = sum(lc.bitops_gb for lc in mod_layers)
        mod_bitops_baseline = sum(lc.bitops_baseline_gb for lc in mod_layers)
        
        start_idx = layer_counter + 1
        end_idx = layer_counter + len(mod_layers)
        
        modality_costs[modality] = ModalityCost(
            modality_name=modality,
            layer_range=(start_idx, end_idx),
            num_layers=len(mod_layers),
            total_bitops_gb=mod_bitops,
            total_bitops_baseline_gb=mod_bitops_baseline
        )
        layer_counter += len(mod_layers)
    
    layers_per_modality = layer_counter // len(modality_order) if modality_order else 0
    # Compute average bitwidth
    # avg_bitwidth = total_bitwidth_weighted / total_elements if total_elements > 0 else 0
    
    # Compute requantization overhead (~1% typically)
    # num_conv_layers = sum(1 for lc in layer_costs.values() if lc.layer_type == 'Conv2d')
    # avg_output_elements = sample_input.shape[0] * 64 * 5 * 400  # Rough estimate
    # requant_overhead = num_conv_layers * avg_output_elements * 3 * 32 / 1e9  # 3 ops per element, 32-bit
    
    # total_with_overhead = total_bitops + requant_overhead
    
    # Create summary
    network_cost = NetworkCost(
        layer_costs=layer_costs,
        total_bitops_gb=total_bitops / 1e9,
        total_bitops_baseline_gb=total_bitops_baseline / 1e9,
        # overall_speedup=total_bitops_baseline / total_bitops if total_bitops > 0 else 0,
        # total_weight_memory_mb=total_weight_mem / 1e6,
        # total_weight_memory_baseline_mb=total_weight_mem_baseline / 1e6,
        # overall_compression=total_weight_mem_baseline / total_weight_mem if total_weight_mem > 0 else 0,
        # avg_bitwidth_weights=avg_bitwidth,
        # avg_bitwidth_activations=avg_bitwidth,  # Same for your approach
        # requantization_overhead_gb=requant_overhead / 1e9,
        # total_with_overhead_gb=total_with_overhead / 1e9,
        # speedup_with_overhead=total_bitops_baseline / total_with_overhead if total_with_overhead > 0 else 0
        modality_costs=modality_costs,
        layers_per_modality=layers_per_modality
    )
    
        # Print summary if verbose
    if verbose:
        print("\n" + "="*80)
        print(f"{'OVERALL SUMMARY':^80}")
        print("-"*80)
        print(f"Total BitOPs (Split):        {network_cost.total_bitops_gb:>10.2f} Gb-ops")
        print(f"Total BitOPs ({baseline_bitwidth}-bit):       {network_cost.total_bitops_baseline_gb:>10.2f} Gb-ops")
        print("="*80 + "\n")
    
    return network_cost


def print_cost_comparison(costs: Dict[str, NetworkCost], baseline_name: str = "8-bit"):
    """
    Print a comparison table of multiple cost analyses.
    
    Args:
        costs: Dictionary mapping config name to NetworkCost
        baseline_name: Name of the baseline configuration
    """
    print("\n" + "="*100)
    print(f"{'CONFIGURATION COMPARISON':^100}")
    print("="*100)
    print(f"{'Config':<20} {'BitOPs (Gb)':<15} {'BitOPs Baseline (Gb)':<20}")
    # print(f"{'Config':<20} {'Avg Bits':<12} {'BitOPs (Gb)':<15} {'Speedup':<12} {'Memory (MB)':<15} {'Compression':<12}")
    print("-"*100)
    
    for name, cost in costs.items():
        print(f"{name:<20} {cost.total_bitops_gb:<15.2f} {cost.total_bitops_baseline_gb:<20.2f}")
        # print(f"{name:<20} {cost.avg_bitwidth_weights:<12.2f} {cost.total_with_overhead_gb:<15.2f} "
        #       f"{cost.speedup_with_overhead:<12.2f}× {cost.total_weight_memory_mb:<15.2f} "
        #       f"{cost.overall_compression:<12.2f}×")
    
    print("="*100 + "\n")


