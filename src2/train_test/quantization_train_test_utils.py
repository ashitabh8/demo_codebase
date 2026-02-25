"""
Quantization-Aware Training Utilities

This module provides training utilities specifically for quantization-aware training,
including:
- Joint Quantization training with random bitwidth sampling
- Random bitwidths validation with statistical reporting
- Layer-wise quantization setup and bitwidth management
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.Temperature_Scheduler import build_temp_scheduler
# Import validation function from train_test_utils
from train_test.train_test_utils import (
    validate, calculate_confusion_matrix, plot_confusion_matrix, calculate_macro_f1_from_confusion_matrix,
    setup_optimizer, setup_scheduler
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from models.QuantModules import BitMixerController


def get_conv_class_from_model(model):
    """
    Extract the Conv class being used in the model by inspecting its modules.
    
    Args:
        model: PyTorch model
    
    Returns:
        Conv class type, or None if no quantized Conv layers found
    """
    return model.get_conv_class()


def get_average_bitwidth(model):
    """
    Calculate the average bitwidth across all quantized Conv layers in the model.
    
    Args:
        model: PyTorch model containing quantized Conv layers
    
    Returns:
        float: Average bitwidth across all layers, or None if no quantized layers found
    """
    conv_class = get_conv_class_from_model(model)
    if conv_class is None:
        return None
    
    bitwidths = []
    for module in model.modules():
        if isinstance(module, conv_class):
            if hasattr(module, 'curr_bitwidth') and module.curr_bitwidth is not None:
                bitwidths.append(module.curr_bitwidth)
    
    if len(bitwidths) == 0:
        return None
    
    return np.mean(bitwidths)


def set_random_bitwidth_all_layers(model, bitwidth_options):
    """
    Set random bitwidths for all quantized Conv layers in the model.
    Each layer independently samples from the provided bitwidth options.
    
    Args:
        model: PyTorch model containing quantized Conv layers
        bitwidth_options: List of bitwidth options to sample from (e.g., [4, 6, 8])
    """
    conv_class = get_conv_class_from_model(model)
    if conv_class is None:
        return  # No quantized Conv layers found
    
    for module in model.modules():
        if isinstance(module, conv_class):
            # Each layer samples independently
            bw = bitwidth_options[torch.randint(0, len(bitwidth_options), (1,)).item()]
            module.set_bitwidth(bw)


def get_relative_memory_consumption(model, max_bitwidth=8):
    """
    Calculate the relative BitOps consumption compared to full max_bitwidth precision.
    
    BitOps (Bit Operations) = MACs × weight_bitwidth × activation_bitwidth
    
    For relative calculation (input shapes are the same across comparisons):
        current_bitops = Σ (Nw_i × bw_w_i × bw_a_i) for each layer i
        max_bitops = Σ (Nw_i × max_bitwidth²) for each layer i
    
    Since weight and activation use the same bitwidth in these quantization methods:
        - For QuanConv (single bitwidth): bitops contribution = Nw × bw²
        - For QuanConvSplit (split bitwidths for upper/lower frequency bands):
          bitops contribution = 0.5 × Nw × (bw_upper² + bw_lower²)
          (half the input is processed at each bitwidth)

    Args:
        model: PyTorch model containing quantized Conv layers
        max_bitwidth: Reference maximum bitwidth (default: 8)

    Returns:
        float: Relative BitOps consumption (0 to 1), or None if not applicable.
    """
    conv_class = get_conv_class_from_model(model)
    if conv_class is None:
        return None

    total_max_bitops = 0.0
    current_bitops = 0.0
    
    for module in model.modules():
        if isinstance(module, conv_class):
            Nw = module.weight.numel()
            
            # Max bitops for this layer (assuming max_bitwidth for both weight and activation)
            max_bitops_layer = Nw * float(max_bitwidth) ** 2
            total_max_bitops += max_bitops_layer

            bw = module.get_bitwidth()
            if bw is None:
                continue
            
            # Check if this is a split conv (returns tuple) or regular conv (returns int)
            if isinstance(bw, tuple):
                # QuanConvSplit: (bw_upper, bw_lower)
                bw_upper, bw_lower = bw
                if bw_upper is None or bw_lower is None:
                    continue
                # Half the input is processed at each bitwidth
                # BitOps = 0.5 × Nw × bw_upper² + 0.5 × Nw × bw_lower²
                current_bitops += 0.5 * Nw * (float(bw_upper) ** 2 + float(bw_lower) ** 2)
            else:
                # QuanConv: single bitwidth for both weight and activation
                # BitOps = Nw × bw²
                current_bitops += Nw * float(bw) ** 2

    if total_max_bitops == 0.0:
        return None

    return current_bitops / total_max_bitops


def set_all_bitwidths_given_list(model, bitwidth_list):
    """
    Set bitwidths for all quantized Conv layers in the model from a list.
    
    Args:
        model: PyTorch model containing quantized Conv layers
        bitwidth_list: List of bitwidths, one per quantized Conv layer (in module order)
    
    Raises:
        ValueError: If no quantized Conv layers found
        AssertionError: If bitwidth_list length doesn't match number of layers
    """
    conv_class = get_conv_class_from_model(model)
    if conv_class is None:
        raise ValueError("No quantized Conv layers found in model - some flags or args are wrong")
    
    # Count conv layers and set bitwidths
    conv_layer_idx = 0
    for module in model.modules():
        if isinstance(module, conv_class):
            assert conv_layer_idx < len(bitwidth_list), \
                f"Bitwidth list (len={len(bitwidth_list)}) is not long enough for layer {conv_layer_idx}"
            module.set_bitwidth(bitwidth_list[conv_layer_idx])
            conv_layer_idx += 1
    
    # Verify we used all bitwidths
    assert conv_layer_idx == len(bitwidth_list), \
        f"Bitwidth list length ({len(bitwidth_list)}) doesn't match number of conv layers ({conv_layer_idx})"


def setup_quantization_layers(model, quant_config):
    """
    Setup quantization functions for all quantized Conv layers in the model.
    
    Args:
        model: PyTorch model containing quantized Conv layers
        quant_config: Quantization configuration dictionary
    
    Returns:
        model: Model with quantization layers configured
    """
    conv_class = get_conv_class_from_model(model)
    if conv_class is None:
        logging.warning("No quantized Conv layers found in model")
        return model
    
    # Setup quantization for all quantized Conv layers
    for module in model.modules():
        if isinstance(module, conv_class):
            module.setup_quantize_funcs(quant_config)
    
    logging.info("Quantization layers configured successfully")
    logging.info(f"  Weight quantization: {quant_config.get('weight_quantization', 'N/A')}")
    logging.info(f"  Activation quantization: {quant_config.get('activation_quantization', 'N/A')}")
    logging.info(f"  Bitwidth options: {quant_config.get('bitwidth_options', 'N/A')}")
    
    return model


def setup_quantization_for_testing(model, config, test_function, device):
    """
    Setup quantization layers for testing based on the test function.
    
    This function:
    1. Checks if quantization is needed based on test_function
    2. Extracts quantization configuration from config if needed
    3. Sets up quantization layers (creates importance_vector parameters)
    4. Moves model to the appropriate device
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        test_function: Test function name ('float', 'single_precision_quantized', 
                      'random_bitwidth', 'single_precision_and_random_bitwidth')
        device: torch.device to move model to
    
    Returns:
        model: Model with quantization layers configured (if needed) and moved to device
    """
    # Check if quantization setup is needed based on test function

    valid_test_functions = ['float', 'single_precision_quantized', 'random_bitwidth', 
                           'single_precision_and_random_bitwidth']
    valid_quantized_tessting_functions = ['single_precision_quantized', 'random_bitwidth', 
                                          'single_precision_and_random_bitwidth']
    if test_function not in valid_test_functions:
        raise ValueError(f"Invalid test function: {test_function}, "
                        f"must be one of: {valid_test_functions}")

    quantization_enabled = config.get('quantization', {}).get('enable', False)
    if test_function in valid_quantized_tessting_functions and not quantization_enabled:
        raise ValueError("Quantization is not enabled in config. "
                        "Cannot use quantized test functions without quantization enabled.")
    

    # Float test - no quantization layers needed
    if test_function == 'float':
        logging.info("Float test selected - no quantization layers needed")
        model = model.to(device)
        return model
    
    # For quantized tests, setup quantization layers
    logging.info(f"Setting up quantization layers for {test_function} testing...")
    
    # Extract quantization configuration
    quantization_method = config.get('quantization_method', None)
    assert quantization_method is not None, "Quantization method is not provided in the config"
    logging.info(f"Using quantization method: {quantization_method}")
    
    # Get nested quantization config (same as in training)
    quant_config = config['quantization'][quantization_method]
    
    # Setup quantization layers FIRST
    # This creates importance_vector parameters for QuanConvImportance layers
    # or alpha parameters for PACT
    # or bitwidth parameters for LSQ
    model = setup_quantization_layers(model, quant_config)
    model = model.to(device)
    
    logging.info("Quantization layers setup successfully")
    
    return model


def validate_simple(model, val_loader, loss_fn, device, augmenter, apply_augmentation_fn, bitwidth):
    """
    Simple validation with a single fixed bitwidth.
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        bitwidth: Fixed bitwidth to use for validation
    
    Returns:
        dict: Validation metrics (accuracy, loss)
    """
    model.eval()
    
    # Set fixed bitwidth for all layers
    conv_class = get_conv_class_from_model(model)
    if conv_class is not None:
        for module in model.modules():
            if isinstance(module, conv_class):
                module.set_bitwidth(bitwidth)
    
    logging.info(f"Validating with fixed bitwidth: {bitwidth}")
    
    # Run standard validation
    val_result = validate(
        model, val_loader, loss_fn, device,
        augmenter, apply_augmentation_fn
    )
    
    f1_str = ""
    if val_result.get("f1_macro") is not None:
        f1_str = f", Macro-F1: {val_result['f1_macro']:.4f}"
    logging.info(f"  Validation - Acc: {val_result['accuracy']:.4f}, Loss: {val_result['loss']:.4f}{f1_str}")
    
    return val_result


def validate_random_bitwidths(model, val_loader, loss_fn, device, 
                               augmenter, apply_augmentation_fn, num_configs, 
                               bitwidth_options, bin_tolerance=0.5):
    """
    Validate model with multiple random bitwidth configurations and report statistics.
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        num_configs: Number of random configurations to test
        bitwidth_options: List of bitwidth options
        bin_tolerance: Tolerance for grouping relative memory into bins (default: 0.5)
    
    Returns:
        dict: Statistics including mean, min, max, std for accuracy, loss, and relative memory
    """
    model.eval()
    results = []
    
    logging.info(f"Validating with {num_configs} random bitwidth configurations...")
    logging.info(f" validate_random_bitwidths: Bitwidth options: {bitwidth_options}")
    logging.info(f" validate_random_bitwidths: Bin tolerance: {bin_tolerance}")
    
    for i in range(num_configs):
        # Set random bitwidths for all layers
        set_random_bitwidth_all_layers(model, bitwidth_options)
        
        # Get relative memory consumption for this configuration
        relative_memory_consumption = get_relative_memory_consumption(model, max_bitwidth=max(bitwidth_options))
        
        # Run validation
        val_result = validate(
            model, val_loader, loss_fn, device,
            augmenter, apply_augmentation_fn
        )
        
        results.append({
            'accuracy': val_result['accuracy'],
            'loss': val_result['loss'],
            'f1_macro': val_result.get('f1_macro', None),
            'relative_memory_consumption': relative_memory_consumption if relative_memory_consumption is not None else 0.0
        })
        
        f1_str = f", Macro-F1={val_result['f1_macro']:.4f}" if val_result.get("f1_macro") is not None else ""
        memory_str = f", Relative Memory={relative_memory_consumption:.4f}" if relative_memory_consumption is not None else ""
        logging.info(f"  Config {i+1}/{num_configs}: Acc={val_result['accuracy']:.4f}, "
                    f"Loss={val_result['loss']:.4f}{f1_str}{memory_str}")
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in results]
    losses = [r['loss'] for r in results]
    f1_macros = [r['f1_macro'] for r in results if r.get('f1_macro') is not None]
    relative_memory_consumptions = [r['relative_memory_consumption'] for r in results]
    
    # Calculate bin-level statistics
    # Create bins based on unique relative memory values (rounded to 2 decimals)
    unique_rel_mems = np.unique(np.round(relative_memory_consumptions, 2))
    
    bin_stats = []
    # Assign each data point to the closest bin center (no overlapping bins)
    rel_mem_array = np.array(relative_memory_consumptions)
    accuracies_array = np.array(accuracies)
    
    for rm in unique_rel_mems:
        # Find data points closest to this bin center
        distances = np.abs(rel_mem_array - rm)
        
        # Assign each point to its closest bin
        # For each data point, check if this bin is the closest
        mask = np.array([np.abs(rel_m - rm) == np.min([np.abs(rel_m - urm) for urm in unique_rel_mems]) 
                         for rel_m in rel_mem_array])
        
        if np.sum(mask) > 0:
            bin_accs = accuracies_array[mask]
            bin_stats.append({
                'relative_memory': float(rm),
                'mean_acc': float(np.mean(bin_accs)),
                'std_acc': float(np.std(bin_accs)),
                'min_acc': float(np.min(bin_accs)),
                'max_acc': float(np.max(bin_accs)),
                'count': int(np.sum(mask))
            })
    
    # Calculate average validation std across all bins
    avg_val_std = np.mean([b['std_acc'] for b in bin_stats]) if bin_stats else 0.0
    
    stats = {
        'mean_acc': np.mean(accuracies),
        'min_acc': np.min(accuracies),
        'max_acc': np.max(accuracies),
        'std_acc': np.std(accuracies),
        'mean_f1_macro': float(np.mean(f1_macros)) if len(f1_macros) > 0 else None,
        'min_f1_macro': float(np.min(f1_macros)) if len(f1_macros) > 0 else None,
        'max_f1_macro': float(np.max(f1_macros)) if len(f1_macros) > 0 else None,
        'std_f1_macro': float(np.std(f1_macros)) if len(f1_macros) > 0 else None,
        'mean_loss': np.mean(losses),
        'min_loss': np.min(losses),
        'max_loss': np.max(losses),
        'std_loss': np.std(losses),
        'mean_relative_memory_consumption': np.mean(relative_memory_consumptions),
        'min_relative_memory_consumption': np.min(relative_memory_consumptions),
        'max_relative_memory_consumption': np.max(relative_memory_consumptions),
        'std_relative_memory_consumption': np.std(relative_memory_consumptions),
        'accuracy': np.mean(accuracies),  # For compatibility with standard validation
        'f1_macro': float(np.mean(f1_macros)) if len(f1_macros) > 0 else None,
        'loss': np.mean(losses),
        # Store raw data for plotting
        'all_accuracies': accuracies,
        'all_f1_macros': f1_macros,
        'all_relative_memory_consumptions': relative_memory_consumptions,
        # Store bin-level statistics
        'bin_stats': bin_stats,
        'avg_val_std': avg_val_std
    }
    
    # Log summary statistics
    logging.info("Random Bitwidths Validation Statistics:")
    logging.info(f"  Accuracy - Mean: {stats['mean_acc']:.4f}, Min: {stats['min_acc']:.4f}, "
                f"Max: {stats['max_acc']:.4f}, Std: {stats['std_acc']:.4f}")
    if stats.get("mean_f1_macro") is not None:
        logging.info(f"  Macro-F1 - Mean: {stats['mean_f1_macro']:.4f}, Min: {stats['min_f1_macro']:.4f}, "
                    f"Max: {stats['max_f1_macro']:.4f}, Std: {stats['std_f1_macro']:.4f}")
    logging.info(f"  Loss - Mean: {stats['mean_loss']:.4f}, Min: {stats['min_loss']:.4f}, "
                f"Max: {stats['max_loss']:.4f}, Std: {stats['std_loss']:.4f}")
    logging.info(f"  Relative Memory - Mean: {stats['mean_relative_memory_consumption']:.4f}, Min: {stats['min_relative_memory_consumption']:.4f}, "
                f"Max: {stats['max_relative_memory_consumption']:.4f}, Std: {stats['std_relative_memory_consumption']:.4f}")
    
    # Log bin-level statistics
    if bin_stats:
        logging.info(f"  Bin Statistics ({len(bin_stats)} bins, tolerance=±{bin_tolerance}):")
        for bin_stat in bin_stats:
            logging.info(f"    Relative Memory {bin_stat['relative_memory']:.4f}: "
                        f"Acc={bin_stat['mean_acc']:.4f}±{bin_stat['std_acc']:.4f} "
                        f"(n={bin_stat['count']})")
        logging.info(f"  Average Validation Std across bins: {avg_val_std:.4f}")
    
    return stats


def validate_importance_vector_comprehensive(model, val_loader, loss_fn, device, 
                                             augmenter, apply_augmentation_fn, 
                                             num_configs, bitwidth_options=None):
    """
    Comprehensive validation for importance vector training with three modes:
    1. Random bitwidths (baseline comparison - uniform sampling)
    2. Importance-based sampling (stochastic sampling from learned distributions)
    3. Learned best bitwidths (argmax of importance per layer - deterministic)
    
    Args:
        model: PyTorch model with QuanConvImportance layers
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        num_configs: Number of configurations to test
        bitwidth_options: List of bitwidth options (for determining max_bitwidth)
    
    Returns:
        dict: Comprehensive statistics for all three validation modes
    """
    conv_class = model.get_conv_class()
    
    # Determine max_bitwidth for relative memory calculation
    if bitwidth_options is None:
        bitwidth_options = [8]  # Default to 8-bit
    max_bw = max(bitwidth_options)
    
    # Helper function to get relative memory consumption for QuanConvImportance layers
    # Note: For stochastic modes, this reflects bitwidths from the last forward pass
    def get_rel_memory():
        return get_relative_memory_consumption(model, max_bitwidth=max_bw)
    
    model.eval()
    
    logging.info("=" * 80)
    logging.info("Comprehensive Importance Vector Validation")
    logging.info("=" * 80)
    
    # ========================================================================
    # Mode 1: Random Bitwidths (Baseline - Uniform Sampling)
    # ========================================================================
    logging.info("\n[Mode 1] Random Bitwidth Validation (Uniform Baseline)")
    logging.info("-" * 80)
    
    # Set all layers to hard_random mode (uniform sampling)
    conv_class.set_all_layers_mode(model, 'uniform_sampling')
    
    random_results = []
    for i in range(num_configs):
        # Run validation (bitwidths are sampled per forward pass in hard_random mode)
        val_result = validate(
            model, val_loader, loss_fn, device,
            augmenter, apply_augmentation_fn
        )
        
        # Get relative memory consumption (from last batch)
        rel_memory = get_rel_memory()
        
        random_results.append({
            'accuracy': val_result['accuracy'],
            'loss': val_result['loss'],
            'f1_macro': val_result.get('f1_macro', None),
            'relative_memory_consumption': rel_memory if rel_memory is not None else 0.0
        })
        
        memory_str = f", Relative Memory={rel_memory:.4f}" if rel_memory is not None else ""
        logging.info(f"  Config {i+1}/{num_configs}: Acc={val_result['accuracy']:.4f}{memory_str}")
    
    # Calculate random bitwidth statistics
    random_accs = [r['accuracy'] for r in random_results]
    random_losses = [r['loss'] for r in random_results]
    random_f1s = [r['f1_macro'] for r in random_results if r.get('f1_macro') is not None]
    random_rel_mems = [r['relative_memory_consumption'] for r in random_results]
    
    random_stats = {
        'mean_acc': np.mean(random_accs),
        'std_acc': np.std(random_accs),
        'mean_loss': np.mean(random_losses),
        'mean_relative_memory_consumption': np.mean(random_rel_mems),
        'mean_f1_macro': float(np.mean(random_f1s)) if len(random_f1s) > 0 else None,
        'std_f1_macro': float(np.std(random_f1s)) if len(random_f1s) > 0 else None,
    }
    
    f1_str = ""
    if random_stats.get("mean_f1_macro") is not None:
        f1_str = f", Macro-F1={random_stats['mean_f1_macro']:.4f}±{random_stats['std_f1_macro']:.4f}"
    logging.info(f"Random BW Stats: Acc={random_stats['mean_acc']:.4f}±{random_stats['std_acc']:.4f}, "
                f"Loss={random_stats['mean_loss']:.4f}{f1_str}, Rel Mem={random_stats['mean_relative_memory_consumption']:.4f}")
    
    # ========================================================================
    # Mode 2: Importance-Based Sampling (Stochastic from learned distribution)
    # ========================================================================
    # logging.info("\n[Mode 2] Importance-Based Sampling Validation (Stochastic)")
    # logging.info("-" * 80)
    
    # # Set all layers to hard_sampled_from_iv mode (sample from importance)
    # conv_class.set_all_layers_mode(model, 'hard_gumbel_softmax')
    
    # importance_results = []
    # for i in range(num_configs):
    #     # Run validation (bitwidths are sampled from importance per forward pass)
    #     val_result = validate(
    #         model, val_loader, loss_fn, device,
    #         augmenter, apply_augmentation_fn
    #     )
        
    #     # Get average bitwidth (from last batch)
    #     avg_bitwidth = get_avg_bitwidth()
        
    #     importance_results.append({
    #         'accuracy': val_result['accuracy'],
    #         'loss': val_result['loss'],
    #         'avg_bitwidth': avg_bitwidth
    #     })
        
    #     bitwidth_str = f", Avg BW={avg_bitwidth:.2f}"
    #     logging.info(f"  Config {i+1}/{num_configs}: Acc={val_result['accuracy']:.4f}{bitwidth_str}")
    
    # # Calculate importance sampling statistics
    # importance_accs = [r['accuracy'] for r in importance_results]
    # importance_losses = [r['loss'] for r in importance_results]
    # importance_bws = [r['avg_bitwidth'] for r in importance_results]
    
    # importance_stats = {
    #     'mean_acc': np.mean(importance_accs),
    #     'std_acc': np.std(importance_accs),
    #     'mean_loss': np.mean(importance_losses),
    #     'mean_bitwidth': np.mean(importance_bws),
    # }
    
    # logging.info(f"Importance Sampling Stats: Acc={importance_stats['mean_acc']:.4f}±{importance_stats['std_acc']:.4f}, "
    #             f"Loss={importance_stats['mean_loss']:.4f}, BW={importance_stats['mean_bitwidth']:.2f}")
    
    # ========================================================================
    # Mode 3: Learned Best Bitwidths (Argmax per layer - Deterministic)
    # ========================================================================
    logging.info("\n[Mode 3] Learned Best Bitwidth Validation (Argmax/Deterministic)")
    logging.info("-" * 80)
    
    # Set all layers to hard_best mode (argmax of importance)
    conv_class.set_all_layers_mode(model, 'best_bitwidth')
    
    # Get importance distributions and log choices
    importance_dists = conv_class.get_all_importance_distributions(model)
    
    for module in model.modules():
        if isinstance(module, conv_class):
            best_bw = module.get_best_bitwidth()
            if module.layer_name in importance_dists:
                dist = importance_dists[module.layer_name]['distribution']
                confidence = dist.max().item()
                logging.info(f"  {module.layer_name}: Best BW={best_bw} (confidence={confidence:.4f})")
    
    
    # Run full validation with learned bitwidths
    val_result = validate(
        model, val_loader, loss_fn, device,
        augmenter, apply_augmentation_fn
    )
    rel_memory = get_rel_memory()
    
    highest_conf_stats = {
        'accuracy': val_result['accuracy'],
        'loss': val_result['loss'],
        'f1_macro': val_result.get('f1_macro', None),
        'relative_memory_consumption': rel_memory if rel_memory is not None else 0.0
    }
    
    logging.info(f"Highest Confidence Stats: Acc={highest_conf_stats['accuracy']:.4f}, "
                f"Loss={highest_conf_stats['loss']:.4f}, "
                f"Macro-F1={highest_conf_stats['f1_macro']:.4f}, Rel Mem={highest_conf_stats['relative_memory_consumption']:.4f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("Validation Summary")
    logging.info("=" * 80)
    if random_stats.get("mean_f1_macro") is not None:
        logging.info(f"Random BW (uniform):     Acc={random_stats['mean_acc']:.4f}±{random_stats['std_acc']:.4f}, "
                    f"Macro-F1={random_stats['mean_f1_macro']:.4f}±{random_stats['std_f1_macro']:.4f}, "
                    f"Rel Mem={random_stats['mean_relative_memory_consumption']:.4f}")
    else:
        logging.info(f"Random BW (uniform):     Acc={random_stats['mean_acc']:.4f}±{random_stats['std_acc']:.4f}, Rel Mem={random_stats['mean_relative_memory_consumption']:.4f}")
    # logging.info(f"Importance Sampling:     Acc={importance_stats['mean_acc']:.4f}±{importance_stats['std_acc']:.4f}, Rel Mem={importance_stats['mean_relative_memory_consumption']:.4f}")
    logging.info(f"Highest Confidence:      Acc={highest_conf_stats['accuracy']:.4f}, "
                f"Macro-F1={highest_conf_stats['f1_macro']:.4f}, Rel Mem={highest_conf_stats['relative_memory_consumption']:.4f}")
    logging.info("=" * 80)
    
    # Return comprehensive statistics
    return {
        # Mode 1: Random bitwidths (baseline)
        'random': {
            'mean_acc': random_stats['mean_acc'],
            'std_acc': random_stats['std_acc'],
            'mean_loss': random_stats['mean_loss'],
            'mean_relative_memory_consumption': random_stats['mean_relative_memory_consumption'],
            'mean_f1_macro': random_stats.get('mean_f1_macro', None),
            'std_f1_macro': random_stats.get('std_f1_macro', None),
            'all_accuracies': random_accs,
            'all_f1_macros': random_f1s,
            'all_relative_memory_consumptions': random_rel_mems,
        },
        # Mode 2: Importance sampling (stochastic)
        # 'importance': {
        #     'mean_acc': importance_stats['mean_acc'],
        #     'std_acc': importance_stats['std_acc'],
        #     'mean_loss': importance_stats['mean_loss'],
        #     'mean_relative_memory_consumption': importance_stats['mean_relative_memory_consumption'],
        #     'all_accuracies': importance_accs,
        #     'all_relative_memory_consumptions': importance_rel_mems,
        # },
        # Mode 3: Highest confidence (deterministic argmax)
        'highest_conf': {
            'accuracy': highest_conf_stats['accuracy'],
            'loss': highest_conf_stats['loss'],
            'f1_macro': highest_conf_stats.get('f1_macro', None),
            'relative_memory_consumption': highest_conf_stats['relative_memory_consumption'],
        },
        # Overall metrics (use highest_conf as main metric for model selection)
        'mean_acc': highest_conf_stats['accuracy'],
        'accuracy': highest_conf_stats['accuracy'],
        'f1_macro': highest_conf_stats.get('f1_macro', None),
        'loss': highest_conf_stats['loss'],
    }


def plot_relative_memory_vs_accuracy(relative_memories, accuracies, epoch, title="Relative Memory vs Accuracy", bin_tolerance=0.5):
    """
    Create a scatter plot or candlestick-style plot of relative memory vs accuracy.
    
    Groups results by relative memory bins and shows mean and std deviation.
    
    Args:
        relative_memories: List of relative memory consumptions
        accuracies: List of corresponding accuracies
        epoch: Current epoch number (for title)
        title: Plot title
        bin_tolerance: Tolerance for grouping relative memories into bins (default: 0.5)
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy arrays
    relative_memories = np.array(relative_memories)
    accuracies = np.array(accuracies)
    
    # Create bins based on unique relative memories (rounded to 2 decimals)
    unique_rel_mems = np.unique(np.round(relative_memories, 2))
    
    if len(unique_rel_mems) > 1:
        # Create bins - assign each point to its closest bin center
        bin_means = []
        bin_stds = []
        bin_mins = []
        bin_maxs = []
        bin_centers = []
        
        for rm in unique_rel_mems:
            # Assign each point to its closest bin (no overlapping bins)
            mask = np.array([np.abs(rm_val - rm) == np.min([np.abs(rm_val - urm) for urm in unique_rel_mems]) 
                             for rm_val in relative_memories])
            
            if np.sum(mask) > 0:
                bin_accs = accuracies[mask]
                bin_centers.append(rm)
                bin_means.append(np.mean(bin_accs))
                bin_stds.append(np.std(bin_accs))
                bin_mins.append(np.min(bin_accs))
                bin_maxs.append(np.max(bin_accs))
        
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        bin_mins = np.array(bin_mins)
        bin_maxs = np.array(bin_maxs)
        
        # Plot individual points with transparency
        ax.scatter(relative_memories, accuracies, alpha=0.3, s=50, c='lightblue', 
                  edgecolors='blue', linewidth=0.5, label='Individual configs')
        
        # Plot mean line
        ax.plot(bin_centers, bin_means, 'ro-', linewidth=2, markersize=8, 
               label='Mean accuracy', zorder=5)
        
        # Plot error bars (std deviation)
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='none', 
                   ecolor='red', capsize=5, capthick=2, alpha=0.7, zorder=4)
        
        # Plot min-max range as vertical lines
        for i in range(len(bin_centers)):
            ax.plot([bin_centers[i], bin_centers[i]], [bin_mins[i], bin_maxs[i]], 
                   'k-', alpha=0.3, linewidth=1, zorder=3)
    else:
        # If only one relative memory value, just show scatter
        ax.scatter(relative_memories, accuracies, alpha=0.6, s=100, c='blue', 
                  edgecolors='darkblue', linewidth=1)
        ax.axhline(y=np.mean(accuracies), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(accuracies):.4f}')
    
    ax.set_xlabel('Relative Memory Consumption', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} (Epoch {epoch + 1})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    # Add statistics text box
    stats_text = f'Mean Acc: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n'
    stats_text += f'Mean Rel Mem: {np.mean(relative_memories):.4f} ± {np.std(relative_memories):.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def train_epoch_vanilla_single_precision(model, train_loader, optimizer, loss_fn, 
                                         quant_config, augmenter, apply_augmentation_fn, 
                                         device, writer, epoch, clip_grad=None):
    """
    Train one epoch using vanilla single precision training with fixed bitwidth.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        quant_config: Quantization configuration dictionary
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        device: Device to run training on
        writer: TensorBoard writer
        epoch: Current epoch number
        clip_grad: Gradient clipping value (optional)
    
    Returns:
        dict: Training metrics (loss, accuracy)
    """
    model.train()
    
    # Extract bitwidth (should be a single value)
    bitwidth_options = quant_config.get('bitwidth_options', [8])
    assert len(bitwidth_options) == 1, "For training_method: vanilla_single_precision_training, bitwidth_options should contain only one value"
    bitwidth = bitwidth_options[0] if isinstance(bitwidth_options, list) else bitwidth_options
    
    # Set fixed bitwidth for all layers at the start
    conv_class = get_conv_class_from_model(model)
    if conv_class is not None:
        for module in model.modules():
            if isinstance(module, conv_class):
                module.set_bitwidth(bitwidth)
    
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    logging.info(f"Training with fixed bitwidth: {bitwidth}")
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack batch
        if len(batch_data) == 3:
            data, labels, idx = batch_data
        else:
            data, labels = batch_data[0], batch_data[1]
        
        # Apply augmentation if provided
        if augmenter is not None and apply_augmentation_fn is not None:
            data, labels = apply_augmentation_fn(augmenter, data, labels)
        
        # Move to device
        labels = labels.to(device)
        if isinstance(data, dict):
            # Multi-modal data
            for loc in data:
                for mod in data[loc]:
                    data[loc][mod] = data[loc][mod].to(device)
        else:
            data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        
        # Handle one-hot labels if needed
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            loss_labels = torch.argmax(labels, dim=1)
        else:
            loss_labels = labels
        
        loss = loss_fn(outputs, loss_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Calculate metrics
        train_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(outputs, dim=1)
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            labels_idx = torch.argmax(labels, dim=1)
        else:
            labels_idx = labels
        
        train_correct += (predictions == labels_idx).sum().item()
        train_total += labels.size(0)
        
        # Log to TensorBoard every 50 batches
        if batch_idx % 50 == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
    
    # Calculate epoch metrics
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    
    return {
        'loss': epoch_train_loss,
        'accuracy': epoch_train_acc
    }


def train_epoch_joint_quantization(model, train_loader, optimizer, loss_fn, 
                                   quant_config, augmenter, apply_augmentation_fn, 
                                   device, writer, epoch, clip_grad=None):
    """
    Train one epoch using joint quantization strategy.
    
    For each batch, multiple forward passes are performed with different random
    bitwidth configurations, and the losses are averaged.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        quant_config: Quantization configuration dictionary
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        device: Device to run training on
        writer: TensorBoard writer
        epoch: Current epoch number
        clip_grad: Gradient clipping value (optional)
    
    Returns:
        dict: Training metrics (loss, accuracy)
    """
    model.train()
    
    # Extract joint quantization parameters
    joint_quant_config = quant_config.get('joint_quantization', {})
    joint_quantization_batch_size = joint_quant_config.get('joint_quantization_batch_size', 2)
    bitwidth_options = quant_config.get('bitwidth_options', [4, 6, 8])
    
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    logging.info(f"Training with joint quantization (batch_size={joint_quantization_batch_size})")
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack batch
        if len(batch_data) == 3:
            data, labels, idx = batch_data
        else:
            data, labels = batch_data[0], batch_data[1]
        
        # Apply augmentation if provided
        if augmenter is not None and apply_augmentation_fn is not None:
            data, labels = apply_augmentation_fn(augmenter, data, labels)
        
        # Move to device
        labels = labels.to(device)
        if isinstance(data, dict):
            # Multi-modal data
            for loc in data:
                for mod in data[loc]:
                    data[loc][mod] = data[loc][mod].to(device)
        else:
            data = data.to(device)
        
        # Joint quantization: accumulate loss over multiple forward passes
        accumulated_loss = 0.0
        
        for _ in range(joint_quantization_batch_size):
            # Set random bitwidths for all layers
            set_random_bitwidth_all_layers(model, bitwidth_options)
            
            # Forward pass
            outputs = model(data)
            
            # Handle one-hot labels if needed
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            loss = loss_fn(outputs, loss_labels)
            if not torch.isfinite(loss):
                logging.error(
                    f"Non-finite loss detected (loss={loss}) at epoch={epoch}, batch_idx={batch_idx}. "
                    f"Aborting training epoch early to avoid NaN propagation."
                )
                return {"loss": float("nan"), "accuracy": 0.0, "status": "non_finite_loss"}
            accumulated_loss += loss
        
        # Average the loss
        avg_loss = accumulated_loss / joint_quantization_batch_size
        
        # Backward pass
        optimizer.zero_grad()
        avg_loss.backward()
        
        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Calculate metrics (using last outputs from loop)
        train_loss += avg_loss.item() * labels.size(0)
        predictions = torch.argmax(outputs, dim=1)
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            labels_idx = torch.argmax(labels, dim=1)
        else:
            labels_idx = labels
        
        train_correct += (predictions == labels_idx).sum().item()
        train_total += labels.size(0)
        
        # Log to TensorBoard every 50 batches
        if batch_idx % 50 == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', avg_loss.item(), global_step)
    
    # Calculate epoch metrics
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    
    return {
        'loss': epoch_train_loss,
        'accuracy': epoch_train_acc
    }


def train_epoch_bitmixer(
    model,
    train_loader,
    optimizer,
    loss_fn,
    quant_config,
    augmenter,
    apply_augmentation_fn,
    device,
    writer,
    epoch,
    bitmixer_controller: BitMixerController,
    bitmixer_epoch_for_sigma: int,
    clip_grad=None,
):
    """
    Train one epoch using BitMixer.

    Behavior:
    - Each batch: call BitMixerController.step(...) BEFORE forward() to set bitwidths
      and enable/disable weight quantization per BitMixer stage.
    - Single forward pass per batch (unlike joint_quantization's K passes).
    """
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack batch
        if len(batch_data) == 3:
            data, labels, idx = batch_data
        else:
            data, labels = batch_data[0], batch_data[1]

        # Apply augmentation if provided
        if augmenter is not None and apply_augmentation_fn is not None:
            data, labels = apply_augmentation_fn(augmenter, data, labels)

        # Move to device
        labels = labels.to(device)
        if isinstance(data, dict):
            for loc in data:
                for mod in data[loc]:
                    data[loc][mod] = data[loc][mod].to(device)
        else:
            data = data.to(device)

        # BitMixer bitwidth assignment (per batch)
        bitmixer_controller.step(bitmixer_epoch_for_sigma)

        # Forward/backward
        optimizer.zero_grad()
        outputs = model(data)

        if len(labels.shape) == 2 and labels.shape[1] > 1:
            loss_labels = torch.argmax(labels, dim=1)
        else:
            loss_labels = labels

        loss = loss_fn(outputs, loss_labels)
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Metrics
        train_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(outputs, dim=1)
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            labels_idx = torch.argmax(labels, dim=1)
        else:
            labels_idx = labels

        train_correct += (predictions == labels_idx).sum().item()
        train_total += labels.size(0)

        if batch_idx % 50 == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total

    return {'loss': epoch_train_loss, 'accuracy': epoch_train_acc}


def train_epoch_importance_vector(model, train_loader, optimizer, temp_scheduler, loss_fn, 
                                   quant_config, augmenter, apply_augmentation_fn, 
                                   device, writer, epoch, num_epochs, clip_grad=None):
    """
    Train one epoch using importance vector-based adaptive quantization.
    
    For each batch, performs K forward passes with different bitwidth configurations
    sampled from learned importance distributions. Uses multi-component loss with
    variance regularization to encourage robustness across bitwidth choices.
    
    Args:
        model: PyTorch model to train (with QuanConvImportance layers)
        train_loader: Training data loader
        optimizer: Optimizer (optimizes both network weights and importance vectors)
        temp_scheduler: Temperature scheduler (updates temperature in all layers)
        loss_fn: Loss function (should be ImportanceVectorLoss)
        quant_config: Quantization configuration dictionary
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        device: Device to run training on
        writer: TensorBoard writer
        epoch: Current epoch number
        num_epochs: Total number of epochs (for temperature annealing)
        clip_grad: Gradient clipping value (optional)
    
    Returns:
        dict: Training metrics (loss, accuracy, task_loss, variance_loss, budget_loss, temperature)
    """
    
    model.train()

    # Running totals for metrics
    train_loss = 0.0
    train_task_loss = 0.0
    train_variance_loss = 0.0
    train_budget_loss = 0.0
    train_raw_variance = 0.0
    train_correct = 0
    train_total = 0
    num_batches = 0

    conv_class = model.get_conv_class()
    mode = quant_config.get('importance_vector', {}).get('sampling_strategy', 'hard_gumbel_softmax')
    conv_class.set_all_layers_mode(model, mode)
    
    # Create tqdm progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", 
                leave=True, dynamic_ncols=True)
    
    num_configs = quant_config.get('number_of_configs', 8)
    assert num_configs > 1, "Number of configurations must be greater than 1"
    
    logging.info(f"Training with importance vector (num_configs={num_configs}, mode={mode})")
    
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch
        if len(batch_data) == 3:
            data, labels, idx = batch_data
        else:
            data, labels = batch_data[0], batch_data[1]
        
        # Apply augmentation if provided
        if augmenter is not None and apply_augmentation_fn is not None:
            data, labels = apply_augmentation_fn(augmenter, data, labels)
        
        # Move to device
        labels = labels.to(device)
        if isinstance(data, dict):
            # Multi-modal data
            for loc in data:
                for mod in data[loc]:
                    data[loc][mod] = data[loc][mod].to(device)
        else:
            data = data.to(device)
        
        # Prepare labels for loss computation
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            loss_labels = torch.argmax(labels, dim=1)
        else:
            loss_labels = labels
        
        # ================================================================
        # Multi-config forward passes (like joint_quantization)
        # Each forward pass samples different bitwidths from importance
        # ================================================================
        outputs_list = []
        labels_list = []
        
        for _ in range(num_configs):
            # Forward pass - bitwidths are sampled based on mode
            # (each forward will sample different bitwidths from importance distribution)
            outputs = model(data)
            outputs_list.append(outputs)
            labels_list.append(loss_labels)
        
        # ================================================================
        # Compute multi-component loss using ImportanceVectorLoss
        # Returns (total_loss, loss_components) when given lists
        # ================================================================
        loss, loss_components = loss_fn(outputs_list, labels_list, model=model)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # ================================================================
        # Accumulate metrics from loss components
        # ================================================================
        batch_size = labels.size(0)
        train_loss += loss_components['total_loss'] * batch_size
        train_task_loss += loss_components['task_loss'] * batch_size
        train_variance_loss += loss_components['variance_loss'] * batch_size
        train_budget_loss += loss_components['budget_loss'] * batch_size
        train_raw_variance += loss_components['raw_variance'] * batch_size
        
        # Use mean accuracy from loss_components (averaged across K configs)
        train_correct += loss_components['mean_accuracy'] * batch_size
        train_total += batch_size
        num_batches += 1
        
        # Update progress bar with current metrics
        current_acc = train_correct / train_total if train_total > 0 else 0
        current_loss = train_loss / train_total if train_total > 0 else 0
        current_task_loss = train_task_loss / train_total if train_total > 0 else 0
        current_var_loss = train_variance_loss / train_total if train_total > 0 else 0
        
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'task': f'{current_task_loss:.4f}',
            'var': f'{current_var_loss:.4f}',
            'acc': f'{current_acc:.4f}',
            'temp': f'{temp_scheduler.get_last_temp():.3f}'
        })
        
        # Log to TensorBoard every 50 batches
        if batch_idx % 50 == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Total_Loss', loss_components['total_loss'], global_step)
            writer.add_scalar('Train/Batch_Task_Loss', loss_components['task_loss'], global_step)
            writer.add_scalar('Train/Batch_Variance_Loss', loss_components['variance_loss'], global_step)
            writer.add_scalar('Train/Batch_Budget_Loss', loss_components['budget_loss'], global_step)
            writer.add_scalar('Train/Batch_Raw_Variance', loss_components['raw_variance'], global_step)
            writer.add_scalar('Train/Batch_Mean_Accuracy', loss_components['mean_accuracy'], global_step)
            writer.add_scalar('Train/Batch_Accuracy_Std', loss_components['accuracy_std'], global_step)
            writer.add_scalar('Train/Temperature', temp_scheduler.get_last_temp(), global_step)
    
    # Calculate epoch metrics
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    epoch_task_loss = train_task_loss / train_total
    epoch_variance_loss = train_variance_loss / train_total
    epoch_budget_loss = train_budget_loss / train_total
    epoch_raw_variance = train_raw_variance / train_total
    epoch_temperature = temp_scheduler.get_last_temp()

    logging.info(f"Epoch Temperature: {epoch_temperature:.3f}")
    # Temperature scheduler step
    temp_scheduler.step()
    
    # Log epoch-level metrics
    logging.info(f"  Epoch Loss Components - Total: {epoch_train_loss:.4f}, "
                f"Task: {epoch_task_loss:.4f}, Variance: {epoch_variance_loss:.4f}, "
                f"Budget: {epoch_budget_loss:.4f}")
    logging.info(f"  Raw Variance: {epoch_raw_variance:.6f}, Mean Accuracy: {epoch_train_acc:.4f}")
    
    return {
        'loss': epoch_train_loss,
        'accuracy': epoch_train_acc,
        'task_loss': epoch_task_loss,
        'variance_loss': epoch_variance_loss,
        'budget_loss': epoch_budget_loss,
        'raw_variance': epoch_raw_variance,
        'temperature': epoch_temperature
    }


def train_with_quantization(model, train_loader, val_loader, config, experiment_dir,
                            loss_fn, augmenter, apply_augmentation_fn):
    """
    Main training function for quantization-aware training.
    
    This function handles:
    - Setup of quantization layers (BEFORE optimizer creation to include all params)
    - Optimizer and scheduler creation (AFTER quantization setup)
    - Training loop with appropriate training method (joint quantization, etc.)
    - Validation with appropriate validation function (random bitwidths, etc.)
    - Checkpointing and best model tracking
    
    Note: Optimizer and scheduler are created internally AFTER setup_quantization_layers()
    to ensure all parameters (including importance vectors) are included.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Full configuration dictionary
        experiment_dir: Path to experiment directory
        loss_fn: Loss function
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
    
    Returns:
        model: Trained model
        train_history: Dictionary with training history
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    
    # Extract quantization configuration
    quantization_enabled = config.get('quantization', {}).get('enable', False)
    if not quantization_enabled:
        raise ValueError("Quantization is not enabled in config. Use standard train() function.")
    
    quantization_method = config.get('quantization_method', 'dorefa')
    logging.info(f"Using quantization method: {quantization_method}")
    
    # Get nested quantization config
    if quantization_method not in config['quantization']:
        raise ValueError(f"Quantization method '{quantization_method}' not found in config. "
                        f"Available methods: {list(config['quantization'].keys())}")
    
    quant_config = config['quantization'][quantization_method]
    
    # Setup quantization layers FIRST
    # This creates importance_vector parameters for QuanConvImportance layers
    model = setup_quantization_layers(model, quant_config)
    model = model.to(device)
    
    # Create optimizer AFTER quantization setup so all parameters are included
    # (importance_vector parameters are created in setup_quantization_layers)
    optimizer = setup_optimizer(model, config)
    
    # Create scheduler
    scheduler = setup_scheduler(optimizer, config)

    
    # Extract training and validation methods
    training_method = quant_config.get('training_method', 'joint_quantization')
    validation_function = quant_config.get('validation_function', 'random_bitwidths')
    
    logging.info(f"Training method: {training_method}")
    logging.info(f"Validation function: {validation_function}")
    
    # Setup directories
    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    models_dir = experiment_path / "models"
    tensorboard_dir = experiment_path / "tensorboard"
    
    # Note: File logging is already set up in train.py, so we don't duplicate it here
    
    # Setup TensorBoard
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Training parameters
    model_name = config.get('model', 'ResNet')
    num_epochs = config.get(model_name, {}).get('lr_scheduler', {}).get('train_epochs', 50)
    num_classes = config.get('vehicle_classification', {}).get('num_classes', 7)
    class_names = config.get('vehicle_classification', {}).get('class_names', None)
    clip_grad = config.get(model_name, {}).get('optimizer', {}).get('clip_grad', None)

    if quant_config.get(quantization_method, {}).get('temp_scheduler', {}):
        temp_config = quant_config.get(quantization_method, {}).get('temp_scheduler', {})
        temperature_scheduler = build_temp_scheduler(model, temp_config, num_epochs)
    else:
        temperature_scheduler = None

    temperature_scheduler = None

    # Only importance-vector training needs temperature scheduling
    if training_method == "joint_quantization_with_importance_vector":
        temperature_scheduler = build_temp_scheduler(model, quant_config, num_epochs)

    # Training history
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_mean_acc': [],
        'val_std_acc': [],
        'learning_rates': [],
        'bitwidth_bin_stats': [],  # Store bin stats for each epoch
        'avg_val_std_history': [],  # Store avg val std for each epoch
        # Loss components for importance vector training
        'train_task_loss': [],
        'train_variance_loss': [],
        'train_budget_loss': [],
        'train_raw_variance': [],
        'train_temperature': []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0
    
    logging.info("=" * 80)
    logging.info("Starting Quantization-Aware Training")
    logging.info(f"Device: {device}")
    logging.info(f"Number of epochs: {num_epochs}")
    logging.info(f"Quantization method: {quantization_method}")
    logging.info(f"Training method: {training_method}")
    logging.info(f"Validation function: {validation_function}")
    logging.info("=" * 80)

    # --------------------------------------------------------------------
    # BitMixer controller setup (if needed)
    # --------------------------------------------------------------------
    bitmixer_controller = None
    bitmixer_cfg = quant_config.get("bitmixer", {})
    if training_method == "bitmixer":
        bitwidth_options = quant_config.get("bitwidth_options", [2, 3, 4, 8])
        stage1_epochs = int(bitmixer_cfg.get("stage1_epochs", 0))
        stage2_epochs = int(bitmixer_cfg.get("stage2_epochs", 0))
        sigma_start = float(bitmixer_cfg.get("sigma_start", 1.0))
        sigma_end = float(bitmixer_cfg.get("sigma_end", 0.25))
        stage3_epochs = int(bitmixer_cfg.get("stage3_epochs", max(num_epochs - stage1_epochs - stage2_epochs, 1)))

        # Initialize in Stage I/II/III depending on epoch; will update each epoch
        bitmixer_controller = BitMixerController(
            model=model,
            bitwidth_options=bitwidth_options,
            stage=1,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            total_epochs=stage3_epochs,
        )
        logging.info(
            f"BitMixer enabled: stage1_epochs={stage1_epochs}, stage2_epochs={stage2_epochs}, "
            f"stage3_epochs={stage3_epochs}, sigma={sigma_start}->{sigma_end}"
        )
    
    for epoch in range(num_epochs):
        # ====================================================================
        # Training Phase
        # ====================================================================
        logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
        
        if training_method == "joint_quantization":
            train_metrics = train_epoch_joint_quantization(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                quant_config=quant_config,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                device=device,
                writer=writer,
                epoch=epoch,
                clip_grad=clip_grad
            )
        elif training_method == "bitmixer":
            assert bitmixer_controller is not None, "BitMixerController was not initialized"

            bitwidth_options = quant_config.get("bitwidth_options", [2, 3, 4, 8])
            stage1_epochs = int(bitmixer_cfg.get("stage1_epochs", 0))
            stage2_epochs = int(bitmixer_cfg.get("stage2_epochs", 0))
            stage3_start = stage1_epochs + stage2_epochs

            # Determine current stage by epoch
            if epoch < stage1_epochs:
                stage = 1
                stage3_epoch = 0
            elif epoch < stage3_start:
                stage = 2
                stage3_epoch = 0
            else:
                stage = 3
                stage3_epoch = epoch - stage3_start

            bitmixer_controller.stage = stage

            logging.info(f"BitMixer stage for epoch {epoch+1}: stage={stage} (stage3_epoch={stage3_epoch})")

            train_metrics = train_epoch_bitmixer(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                quant_config=quant_config,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                device=device,
                writer=writer,
                epoch=epoch,
                bitmixer_controller=bitmixer_controller,
                bitmixer_epoch_for_sigma=stage3_epoch,
                clip_grad=clip_grad,
            )
        elif training_method == "vanilla_single_precision_training":
            train_metrics = train_epoch_vanilla_single_precision(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                quant_config=quant_config,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                device=device,
                writer=writer,
                epoch=epoch,
                clip_grad=clip_grad
            )
        elif training_method == "joint_quantization_with_importance_vector":
            # Setup Temperature Scheduler
            assert temperature_scheduler is not None, "Temperature scheduler is not set"
            train_metrics = train_epoch_importance_vector(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                temp_scheduler=temperature_scheduler,
                loss_fn=loss_fn,
                quant_config=quant_config,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                device=device,
                writer=writer,
                epoch=epoch,
                num_epochs=num_epochs,
                clip_grad=clip_grad
            )
        else:
            raise ValueError(f"Unknown training method: {training_method}")
        
        epoch_train_loss = train_metrics['loss']
        epoch_train_acc = train_metrics['accuracy']
        
        train_history['train_loss'].append(epoch_train_loss)
        train_history['train_acc'].append(epoch_train_acc)
        
        # Append loss components for importance vector training
        if training_method == "joint_quantization_with_importance_vector":
            train_history['train_task_loss'].append(train_metrics.get('task_loss', 0.0))
            train_history['train_variance_loss'].append(train_metrics.get('variance_loss', 0.0))
            train_history['train_budget_loss'].append(train_metrics.get('budget_loss', 0.0))
            train_history['train_raw_variance'].append(train_metrics.get('raw_variance', 0.0))
            train_history['train_temperature'].append(train_metrics.get('temperature', 0.0))
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        if validation_function == "random_bitwidths":
            # Get validation config from quantization.random_bitwidths
            val_config = config.get('quantization', {}).get('random_bitwidths', {})
            num_configs = val_config.get('number_of_configs', 4)
            bitwidth_options = val_config.get('bitwidth_options', [4, 6, 8])
            bin_tolerance = val_config.get('bin_tolerance', 0.5)  # Configurable tolerance
            
            val_stats = validate_random_bitwidths(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                num_configs=num_configs,
                bitwidth_options=bitwidth_options,
                bin_tolerance=bin_tolerance
            )
            
            epoch_val_loss = val_stats['mean_loss']
            epoch_val_acc = val_stats['mean_acc']
            
            train_history['val_mean_acc'].append(val_stats['mean_acc'])
            train_history['val_std_acc'].append(val_stats['std_acc'])
            train_history['bitwidth_bin_stats'].append(val_stats.get('bin_stats', []))
            train_history['avg_val_std_history'].append(val_stats.get('avg_val_std', 0.0))
        elif validation_function == "simple_validation":
            # Simple validation with fixed bitwidth
            bitwidth_options = quant_config.get('bitwidth_options', [8])
            bitwidth = bitwidth_options[0] if isinstance(bitwidth_options, list) else bitwidth_options
            
            val_stats = validate_simple(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                bitwidth=bitwidth
            )
            
            epoch_val_loss = val_stats['loss']
            epoch_val_acc = val_stats['accuracy']
        elif validation_function == "importance_vector_comprehensive_validation":
            # Comprehensive validation for importance vector training
            # Get validation config
            val_config = config.get('quantization', {}).get('importance_vector_comprehensive_validation', {})
            num_configs = val_config.get('number_of_configs', 2)
            bitwidth_options = quant_config.get('bitwidth_options', [8])
            
            val_stats = validate_importance_vector_comprehensive(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                num_configs=num_configs,
                bitwidth_options=bitwidth_options
            )
            
            # Use random bitwidth mean as the main metric (for model selection)
            epoch_val_loss = val_stats['loss']
            epoch_val_acc = val_stats['accuracy']
            
            # Store all three validation modes in history
            if 'val_random_acc' not in train_history:
                train_history['val_random_acc'] = []
                train_history['val_importance_acc'] = []
                train_history['val_highest_conf_acc'] = []
            
            train_history['val_random_acc'].append(val_stats['random']['mean_acc'])
            # train_history['val_importance_acc'].append(val_stats['importance']['mean_acc'])
            train_history['val_highest_conf_acc'].append(val_stats['highest_conf']['accuracy'])
        else:
            raise ValueError(f"Unknown validation function: {validation_function}")
        
        train_history['val_loss'].append(epoch_val_loss)
        train_history['val_acc'].append(epoch_val_acc)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        train_history['learning_rates'].append(current_lr)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # ====================================================================
        # Logging
        # ====================================================================
        logging.info(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        logging.info(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        logging.info(f"  Learning Rate: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Additional validation metrics for random_bitwidths
        if validation_function == "random_bitwidths":
            writer.add_scalar('Validation/mean_acc', val_stats['mean_acc'], epoch)
            writer.add_scalar('Validation/std_acc', val_stats['std_acc'], epoch)
            writer.add_scalar('Validation/min_acc', val_stats['min_acc'], epoch)
            writer.add_scalar('Validation/max_acc', val_stats['max_acc'], epoch)

            if val_stats.get("mean_f1_macro") is not None:
                writer.add_scalar('F1/val_macro', val_stats['mean_f1_macro'], epoch)
                writer.add_scalar('Validation/mean_f1_macro', val_stats['mean_f1_macro'], epoch)
                writer.add_scalar('Validation/std_f1_macro', val_stats['std_f1_macro'], epoch)
                if val_stats.get("min_f1_macro") is not None:
                    writer.add_scalar('Validation/min_f1_macro', val_stats['min_f1_macro'], epoch)
                if val_stats.get("max_f1_macro") is not None:
                    writer.add_scalar('Validation/max_f1_macro', val_stats['max_f1_macro'], epoch)
            
            # Log relative memory consumption statistics
            writer.add_scalar('Relative_Memory_Consumption/mean_relative_memory_consumption', val_stats['mean_relative_memory_consumption'], epoch)
            writer.add_scalar('Relative_Memory_Consumption/std_relative_memory_consumption', val_stats['std_relative_memory_consumption'], epoch)
            writer.add_scalar('Relative_Memory_Consumption/min_relative_memory_consumption', val_stats['min_relative_memory_consumption'], epoch)
            writer.add_scalar('Relative_Memory_Consumption/max_relative_memory_consumption', val_stats['max_relative_memory_consumption'], epoch)
        
        # Additional validation metrics for importance_vector_comprehensive
        if validation_function == "importance_vector_comprehensive_validation":
            # Log all three validation modes
            writer.add_scalar('Validation/random_mean_acc', val_stats['random']['mean_acc'], epoch)
            writer.add_scalar('Validation/random_std_acc', val_stats['random']['std_acc'], epoch)
            # writer.add_scalar('Validation/importance_mean_acc', val_stats['importance']['mean_acc'], epoch)
            # writer.add_scalar('Validation/importance_std_acc', val_stats['importance']['std_acc'], epoch)
            writer.add_scalar('Validation/highest_conf_acc', val_stats['highest_conf']['accuracy'], epoch)
            
            # Log relative memory consumption statistics for all modes
            writer.add_scalar('Relative_Memory_Consumption/random_mean_rel_mem', val_stats['random']['mean_relative_memory_consumption'], epoch)
            # writer.add_scalar('Relative_Memory_Consumption/importance_mean_rel_mem', val_stats['importance']['mean_relative_memory_consumption'], epoch)
            writer.add_scalar('Relative_Memory_Consumption/highest_conf_rel_mem', val_stats['highest_conf']['relative_memory_consumption'], epoch)
            
            # Create comparison plot
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                modes = ['Random\nBitwidths', 'Highest\nConfidence']
                accs = [
                    val_stats['random']['mean_acc'],
                    val_stats['highest_conf']['accuracy']
                ]
                stds = [
                    val_stats['random']['std_acc'],
                    0.0  # Single measurement for highest confidence
                ]
                
                bars = ax.bar(modes, accs, yerr=stds, capsize=5, alpha=0.7, 
                             color=['steelblue', 'orange'])
                ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
                ax.set_title(f'Validation Accuracy Comparison (Epoch {epoch + 1})', 
                           fontsize=14, fontweight='bold')
                ax.set_ylim([min(accs) - 0.05, max(accs) + 0.05])
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, acc, std in zip(bars, accs, stds):
                    height = bar.get_height()
                    label = f'{acc:.4f}'
                    if std > 0:
                        label += f'±{std:.4f}'
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           label, ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                writer.add_figure('Validation/mode_comparison', fig, epoch)
                plt.close(fig)
            except Exception as e:
                logging.warning(f"Could not create validation comparison plot: {e}")
            
            # Create and log relative memory vs accuracy plot (every 5 epochs or last epoch)
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                rel_mem_acc_fig = plot_relative_memory_vs_accuracy(
                    relative_memories=val_stats['all_relative_memory_consumptions'],
                    accuracies=val_stats['all_accuracies'],
                    epoch=epoch,
                    title="Relative Memory vs Validation Accuracy",
                    bin_tolerance=bin_tolerance
                )
                writer.add_figure('Relative_Memory_Analysis/relative_memory_vs_accuracy', rel_mem_acc_fig, epoch)
                plt.close(rel_mem_acc_fig)
                logging.info(f"  Relative Memory vs Accuracy plot logged to TensorBoard")
        
        # Additional logging for importance vector training
        if training_method == "joint_quantization_with_importance_vector":
            from models.QuantModules import QuanConvImportance
            
            # Log temperature
            if 'temperature' in train_metrics:
                writer.add_scalar('Train/Epoch_Temperature', train_metrics['temperature'], epoch)
            
            # Log all loss components (epoch-level)
            if 'task_loss' in train_metrics:
                writer.add_scalar('Loss/Epoch_Task_Loss', train_metrics['task_loss'], epoch)
            
            if 'variance_loss' in train_metrics:
                writer.add_scalar('Loss/Epoch_Variance_Loss', train_metrics['variance_loss'], epoch)
            
            if 'budget_loss' in train_metrics:
                writer.add_scalar('Loss/Epoch_Budget_Loss', train_metrics['budget_loss'], epoch)
            
            if 'raw_variance' in train_metrics:
                writer.add_scalar('Loss/Epoch_Raw_Variance', train_metrics['raw_variance'], epoch)
            
            # Create loss components breakdown plot
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot 1: Loss components bar chart for this epoch
                loss_names = ['Total', 'Task', 'Variance', 'Budget']
                loss_values = [
                    train_metrics.get('loss', 0),
                    train_metrics.get('task_loss', 0),
                    train_metrics.get('variance_loss', 0),
                    train_metrics.get('budget_loss', 0)
                ]
                colors = ['steelblue', 'forestgreen', 'orange', 'crimson']
                
                bars = axes[0].bar(loss_names, loss_values, color=colors, alpha=0.8, edgecolor='black')
                axes[0].set_ylabel('Loss Value', fontsize=12, fontweight='bold')
                axes[0].set_title(f'Loss Components (Epoch {epoch + 1})', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, val in zip(bars, loss_values):
                    height = bar.get_height()
                    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                # Plot 2: Loss components as percentage of total
                total_loss = train_metrics.get('loss', 1)
                if total_loss > 0:
                    percentages = [v / total_loss * 100 for v in loss_values[1:]]  # Skip total
                    wedges, texts, autotexts = axes[1].pie(
                        percentages, labels=['Task', 'Variance', 'Budget'],
                        colors=colors[1:], autopct='%1.1f%%',
                        startangle=90, explode=[0.02, 0.02, 0.02]
                    )
                    axes[1].set_title(f'Loss Component Distribution (Epoch {epoch + 1})', 
                                     fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                writer.add_figure('Loss/Components_Breakdown', fig, epoch)
                plt.close(fig)
            except Exception as e:
                logging.warning(f"Could not create loss components plot: {e}")
            
            # Visualize importance distributions
            iv_config = quant_config.get('importance_vector', {})
            visualize_freq = quant_config.get('visualize_importance_every', 5)
            
            if (epoch + 1) % visualize_freq == 0 or epoch == num_epochs - 1:
                importance_dists = QuanConvImportance.get_all_importance_distributions(model)
                if len(importance_dists) > 0:
                    bitwidth_options = quant_config.get('bitwidth_options', [4, 8])
                    fig = QuanConvImportance.plot_importance_distributions(importance_dists, bitwidth_options, epoch)
                    writer.add_figure('Importance/distributions', fig, epoch)
                    plt.close(fig)
                    logging.info(f"  Importance distributions logged to TensorBoard")
        
        # ====================================================================
        # Save Checkpoints
        # ====================================================================
        # Save best model (based on mean accuracy)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_model_path = models_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
                'config': config,
                'quantization_method': quantization_method
            }, best_model_path)
            logging.info(f"  Best model saved! (Val Acc: {best_val_acc:.4f})")

        # Save checkpoint every 10 epochs for later testing
        if (epoch + 1) % 10 == 0:
            checkpoint_path = models_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
                'config': config,
                'quantization_method': quantization_method
            }, checkpoint_path)
            logging.info(f"  Checkpoint saved at epoch {epoch + 1}")
        
        # Save last epoch
        last_model_path = models_dir / "last_epoch.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': epoch_val_acc,
            'val_loss': epoch_val_loss,
            'config': config,
            'quantization_method': quantization_method
        }, last_model_path)
    
    # Final summary
    logging.info("=" * 80)
    logging.info("Training Complete!")
    logging.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    logging.info(f"Models saved to: {models_dir}")
    logging.info(f"TensorBoard logs: {tensorboard_dir}")
    logging.info("=" * 80)
    
    writer.close()
    
    return model, train_history

