"""
Quantization Testing Functions

This module provides testing utilities specifically for quantization-aware models,
including:
- Float precision testing (baseline)
- Single precision quantized testing (fixed bitwidth)
- Random bitwidths testing with statistical reporting
- Load and test functions that combine checkpoint loading with testing
"""

import logging
import torch
import numpy as np
from pathlib import Path
from train_test.train_test_utils import validate
from train_test.quantization_train_test_utils import (
    get_conv_class_from_model,
    get_average_bitwidth,
    set_random_bitwidth_all_layers,
    set_all_bitwidths_given_list,
    get_relative_memory_consumption
)
from train_test.quantization_test_utils import (
    generate_schemes_in_relative_memory_bin,
    get_num_quantized_layers
)


def test_float(model, test_loader, loss_fn, device, augmenter, apply_augmentation_fn):
    """
    Test model with float precision (no quantization).
    
    Args:
        model: PyTorch model to test
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to run testing on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
    
    Returns:
        dict: Test metrics (accuracy, loss)
    """
    model.eval()
    
    logging.info("Testing with float precision (no quantization)")
    
    # Run standard validation/test
    test_result = validate(
        model, test_loader, loss_fn, device,
        augmenter, apply_augmentation_fn
    )
    
    logging.info(f"  Test - Acc: {test_result['accuracy']:.4f}, Loss: {test_result['loss']:.4f}")
    
    return test_result


def test_simple(model, test_loader, loss_fn, device, augmenter, apply_augmentation_fn, bitwidth):
    """
    Simple testing with a single fixed bitwidth.
    
    Args:
        model: PyTorch model to test
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to run testing on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        bitwidth: Fixed bitwidth to use for testing
    
    Returns:
        dict: Test metrics (accuracy, loss)
    """
    model.eval()
    
    # Set fixed bitwidth for all layers
    conv_class = get_conv_class_from_model(model)
    if conv_class is not None:
        for module in model.modules():
            if isinstance(module, conv_class):
                module.set_bitwidth(bitwidth)
    
    logging.info(f"Testing with fixed bitwidth: {bitwidth}")
    
    # Run standard validation/test
    test_result = validate(
        model, test_loader, loss_fn, device,
        augmenter, apply_augmentation_fn
    )
    
    logging.info(f"  Test - Acc: {test_result['accuracy']:.4f}, Loss: {test_result['loss']:.4f}")
    
    return test_result


def test_random_bitwidths(model, test_loader, loss_fn, device, 
                          augmenter, apply_augmentation_fn, num_quantization_schemes, 
                          bitwidth_options, bitwidth_bin_size=None):
    """
    Test model with multiple random bitwidth quantization schemes and report statistics.
    
    When bitwidth_bin_size is provided, generates schemes with target relative memory
    consumption within the specified range using smart sampling (greedy algorithm).
    Relative memory is computed as current_config_memory / max_8bit_memory.
    Otherwise, uses uniform random sampling from bitwidth_options.
    
    Args:
        model: PyTorch model to test
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to run testing on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        num_quantization_schemes: Number of different quantization schemes to test
        bitwidth_options: List of bitwidth options (e.g., [2, 4, 8])
        bitwidth_bin_size: Tuple of (min, max) for target relative memory range (0 to 1).
                          If None, uses uniform random sampling from bitwidth_options.
    
    Returns:
        dict: Statistics including mean, min, max, std for accuracy, loss, and relative memory
    """
    model.eval()
    results = []
    
    logging.info(f"Testing with {num_quantization_schemes} different quantization schemes...")
    logging.info(f"  Bitwidth options: {bitwidth_options}")
    logging.info(f"  Relative memory target range: {bitwidth_bin_size}")
    
    # Get number of quantized layers in model
    conv_class = get_conv_class_from_model(model)
    num_layers = get_num_quantized_layers(model, conv_class)
    logging.info(f"  Number of quantized layers: {num_layers}")

    # === NEW: Extract layers, input_shapes, output_shapes via forward hooks ===
    layers = []
    input_shapes = []
    output_shapes = []
    hooks = []
    
    def make_hook(module):
        def hook(mod, inp, out):
            layers.append(module)
            input_shapes.append(tuple(inp[0].shape) if isinstance(inp, tuple) else tuple(inp.shape))
            output_shapes.append(tuple(out.shape))
        return hook
    
    # Register hooks for all conv layers matching conv_class
    for module in model.modules():
        if isinstance(module, conv_class):
            hooks.append(module.register_forward_hook(make_hook(module)))
    
    # Run one forward pass to capture shapes
    sample_batch = next(iter(test_loader))
    
    # Unpack batch (same pattern as training code)
    if len(sample_batch) == 3:
        data, labels, idx = sample_batch
    else:
        data, labels = sample_batch[0], sample_batch[1]
    
    # Apply augmentation if provided
    if augmenter is not None and apply_augmentation_fn is not None:
        data, labels = apply_augmentation_fn(augmenter, data, labels)
    
    # Move data to device (handle dict for multi-modal)
    if isinstance(data, dict):
        for loc in data:
            for mod in data[loc]:
                data[loc][mod] = data[loc][mod].to(device)
        sample_input = data
    else:
        sample_input = data.to(device)

    default_bitwidth = bitwidth_options[0]
    for module in model.modules():
        if isinstance(module, conv_class):
            if hasattr(module, 'set_bitwidth'):
                module.set_bitwidth(default_bitwidth)
            elif hasattr(module, 'curr_bitwidth'):
                module.curr_bitwidth = default_bitwidth
    
    with torch.no_grad():
        _ = model(sample_input)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    logging.info(f"  Captured {len(layers)} layer shapes")
    
    # Generate quantization schemes based on whether bin_size is provided
    if bitwidth_bin_size is not None:
        # Smart sampling: generate schemes with relative memory in the target range
        bin_min, bin_max = bitwidth_bin_size
        logging.info(f"  Using targeted sampling with relative memory in [{bin_min}, {bin_max}]")
        quantization_schemes = generate_schemes_in_relative_memory_bin(
            model=model,
            conv_class=conv_class,
            bitwidth_options=bitwidth_options,
            bin_min=bin_min,
            bin_max=bin_max,
            layers=layers,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            num_schemes=num_quantization_schemes
        )
        if quantization_schemes is None:
            raise ValueError("Failed to generate valid schemes for bin [{bin_min}, {bin_max}]")
        use_pregenerated_schemes = True
        logging.info(f"  Generated {len(quantization_schemes)} quantization schemes")
    else:
        # Uniform random sampling is not supported - must provide bitwidth_bin_size
        raise ValueError("Uniform random sampling is not supported. Please provide --relative_memory_consumption_bin_size with a memory range (e.g., --relative_memory_consumption_bin_size 0.1 0.9)")
    
    for i in range(len(quantization_schemes)):
        # Use pre-generated quantization scheme with target average
        bitwidth_list = quantization_schemes[i][1]
        relative_memory_consumption = quantization_schemes[i][0]
        set_all_bitwidths_given_list(model, bitwidth_list)
        
        # Run test
        test_result = validate(
            model, test_loader, loss_fn, device,
            augmenter, apply_augmentation_fn
        )
        
        results.append({
            'accuracy': test_result['accuracy'],
            'loss': test_result['loss'],
            'f1_macro': test_result.get('f1_macro', None),
            'relative_memory_consumption': relative_memory_consumption
        })
        
        # Only log first 10 schemes to avoid cluttering logs
        if i < 10 or i == len(quantization_schemes) - 1:
            bitwidth_str = f", Avg Bitwidth={relative_memory_consumption:.2f}" if relative_memory_consumption is not None else ""
            f1_str = f", Macro-F1={test_result['f1_macro']:.4f}" if test_result.get("f1_macro") is not None else ""
            logging.info(f"  Scheme {i+1}/{len(quantization_schemes)}: Acc={test_result['accuracy']:.4f}, "
                        f"Loss={test_result['loss']:.4f}{f1_str}{bitwidth_str}")

    # Compute statistics
    import numpy as np
    accuracies = [r['accuracy'] for r in results]
    losses = [r['loss'] for r in results]
    rel_memories = [r['relative_memory_consumption'] for r in results if r['relative_memory_consumption'] is not None]
    
    statistics = {
        'accuracy': {
            'mean': np.mean(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'std': np.std(accuracies)
        },
        'loss': {
            'mean': np.mean(losses),
            'min': np.min(losses),
            'max': np.max(losses),
            'std': np.std(losses)
        },
        'num_schemes_tested': len(results),
        'raw_results': results  # Keep raw results for detailed analysis
    }
    
    if rel_memories:
        statistics['relative_memory_consumption'] = {
            'mean': np.mean(rel_memories),
            'min': np.min(rel_memories),
            'max': np.max(rel_memories),
            'std': np.std(rel_memories)
        }
    
    # Log statistics
    logging.info("\nRandom Bitwidths Test Statistics:")
    logging.info(f"  Accuracy - Mean: {statistics['accuracy']['mean']:.4f}, "
                 f"Min: {statistics['accuracy']['min']:.4f}, "
                 f"Max: {statistics['accuracy']['max']:.4f}, "
                 f"Std: {statistics['accuracy']['std']:.4f}")
    logging.info(f"  Loss - Mean: {statistics['loss']['mean']:.4f}, "
                 f"Min: {statistics['loss']['min']:.4f}, "
                 f"Max: {statistics['loss']['max']:.4f}, "
                 f"Std: {statistics['loss']['std']:.4f}")
    if rel_memories:
        logging.info(f"  Relative Memory - Mean: {statistics['relative_memory_consumption']['mean']:.4f}, "
                     f"Min: {statistics['relative_memory_consumption']['min']:.4f}, "
                     f"Max: {statistics['relative_memory_consumption']['max']:.4f}, "
                     f"Std: {statistics['relative_memory_consumption']['std']:.4f}")

    return statistics
    


# =============================================================================
# Load and Test Functions (Checkpoint Loading + Testing)
# =============================================================================

def load_and_test_float(model, checkpoint_path, test_loader, loss_fn, device, 
                        augmenter, apply_augmentation_fn):
    """
    Load checkpoint and run float precision test.
    
    Args:
        model: PyTorch model to test
        checkpoint_path: Path to checkpoint file
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to run testing on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
    
    Returns:
        dict: Test results with 'float' key
    """
    # Load checkpoint
    logging.info("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        logging.info(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Run float test
    logging.info("Running float precision test...")
    test_results = {
        'float': test_float(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation_fn
        )
    }
    
    return test_results


def load_and_test_single_precision(model, checkpoint_path, test_loader, loss_fn, device,
                                   augmenter, apply_augmentation_fn, config, test_args):
    """
    Load checkpoint and run single precision quantized tests for multiple bitwidths.
    
    Args:
        model: PyTorch model to test (quantization already set up)
        checkpoint_path: Path to checkpoint file
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to run testing on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        config: Configuration dictionary
        test_args: Test arguments with override options
    
    Returns:
        dict: Test results with bitwidth as keys
    """
    # Load checkpoint
    logging.info("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        logging.info(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Get bitwidth options from command line or config
    if test_args.override_single_bitwidth is not None:
        bitwidth_options = test_args.override_single_bitwidth
        logging.info(f"Using bitwidth options from command line: {bitwidth_options}")
    else:
        # Fallback to config
        quantization_method = config.get('quantization_method', 'dorefa')
        quant_config = config['quantization'][quantization_method]
        bitwidth_options = quant_config.get('bitwidth_options', None)
        if bitwidth_options is None:
            raise ValueError("Bitwidth options are not set in the config.\n"
                           "Something is wrong with the config file - diagnose the issue.\n"
                           "The config.yaml used for training should have had the bitwidth options set")
        logging.info(f"Using bitwidth options from config: {bitwidth_options}")
    
    # Run tests for each bitwidth
    test_results = {}
    for bitwidth in bitwidth_options:
        logging.info(f"Running single precision quantized test with bitwidth: {bitwidth}...")
        test_results[bitwidth] = test_simple(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation_fn,
            bitwidth=bitwidth
        )
    
    return test_results


def load_and_test_random_bitwidths(model, checkpoint_path, test_loader, loss_fn, device,
                                   augmenter, apply_augmentation_fn, config, test_args):
    """
    Load checkpoint and run random bitwidth tests.
    
    Args:
        model: PyTorch model to test (quantization already set up)
        checkpoint_path: Path to checkpoint file
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to run testing on
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        config: Configuration dictionary
        test_args: Test arguments with override options
    
    Returns:
        dict: Test statistics from random bitwidth testing
    """
    # Load checkpoint
    logging.info("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Get bitwidth options from command line or config
    if test_args.override_bitwidth_options is not None:
        # Here you can add a different bitwidth options than the ones used
        # during training Useful for zero-shot testing - test on different
        # bitwidth options than the ones used during training
        bitwidth_options = test_args.override_bitwidth_options
        logging.info(f"Using bitwidth options from command line: {bitwidth_options}")
    else:
        # Here bitwidth options are the same as they were used during training
        quantization_method = config.get('quantization_method',None)
        assert quantization_method is not None, "Quantization method is not set in the config \n"
        bitwidth_options_from_config = config['quantization'][quantization_method]['bitwidth_options']
        assert bitwidth_options_from_config is not None, "Bitwidth options are not set in the config \n"
        bitwidth_options = bitwidth_options_from_config
        logging.info(f"Using bitwidth options from config: {bitwidth_options}")
    
    # Get test parameters
    num_quantization_schemes = test_args.num_test_configs
    
    # Get bitwidth_bin_size from test_args (can be None for uniform random sampling)
    bitwidth_bin_size = test_args.bitwidth_bin_size
    
    # If bitwidth_bin_size is None, run single test with uniform random sampling
    if bitwidth_bin_size is None:
        logging.info(f"Running random bitwidth test with {num_quantization_schemes} schemes (uniform random sampling)...")
        logging.info(f"  Bitwidth options: {bitwidth_options}")
        logging.info(f"  Number of quantization schemes: {num_quantization_schemes}")
        logging.info(f"  Bitwidth bin size: None (uniform random sampling)")
        
        test_results = test_random_bitwidths(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation_fn,
            num_quantization_schemes=num_quantization_schemes,
            bitwidth_options=bitwidth_options,
            bitwidth_bin_size=None
        )

        # Log accuracy summary for uniform sampling (mean/std across schemes)
        if isinstance(test_results, list) and len(test_results) > 0:
            accs = np.array([r.get("accuracy", 0.0) for r in test_results], dtype=float)
            logging.info(f"Uniform random accuracy: {accs.mean():.4f}±{accs.std():.4f} (n={len(test_results)})")
            f1s = np.array([r.get("f1_macro", np.nan) for r in test_results], dtype=float)
            if np.isfinite(f1s).any():
                f1s = f1s[np.isfinite(f1s)]
                logging.info(f"Uniform random macro-F1: {f1s.mean():.4f}±{f1s.std():.4f} (n={len(f1s)})")
        
        raise ValueError(f"Uniform random sampling is not supported anymore")

        # Wrap results in a dictionary for consistency with multi-range case
        return {"uniform_random": test_results}
    
    # Otherwise, validate and iterate over bitwidth ranges
    assert isinstance(bitwidth_bin_size, list), "Bitwidth ranges should be a list of tuples"
    
    # Log test configuration
    logging.info(f"Running random bitwidth test with {num_quantization_schemes} schemes per range...")
    logging.info(f"  Bitwidth options: {bitwidth_options}")
    logging.info(f"  Number of quantization schemes per range: {num_quantization_schemes}")
    logging.info(f"  Number of bitwidth ranges: {len(bitwidth_bin_size)}")
    for i, (bin_min, bin_max) in enumerate(bitwidth_bin_size):
        logging.info(f"    Range {i+1}: [{bin_min}, {bin_max}]")
    
    # Run random bitwidth test for each range
    all_test_results = {}
    for range_idx, bitwidth_range in enumerate(bitwidth_bin_size):
        bin_min, bin_max = bitwidth_range
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing Range {range_idx + 1}/{len(bitwidth_bin_size)}: [{bin_min}, {bin_max}]")
        logging.info(f"{'='*80}")
        
        range_results = test_random_bitwidths(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation_fn,
            num_quantization_schemes=num_quantization_schemes,
            bitwidth_options=bitwidth_options,
            bitwidth_bin_size=bitwidth_range
        )

        # Log per-range accuracy summary (mean/std across schemes)
        if isinstance(range_results, list) and len(range_results) > 0:
            accs = np.array([r.get("accuracy", 0.0) for r in range_results], dtype=float)
            logging.info(f"Range accuracy [{bin_min}, {bin_max}]: {accs.mean():.4f}±{accs.std():.4f} (n={len(range_results)})")
            f1s = np.array([r.get("f1_macro", np.nan) for r in range_results], dtype=float)
            if np.isfinite(f1s).any():
                f1s = f1s[np.isfinite(f1s)]
                logging.info(f"Range macro-F1 [{bin_min}, {bin_max}]: {f1s.mean():.4f}±{f1s.std():.4f} (n={len(f1s)})")
        elif isinstance(range_results, dict) and range_results.get("status") == "generation_failed":
            logging.error(f"Range [{bin_min}, {bin_max}] failed: scheme generation failed.")
        
        # Store results with range as key
        range_key = f"range_{bin_min}_{bin_max}"
        all_test_results[range_key] = range_results
        
        logging.info(f"Completed testing for range [{bin_min}, {bin_max}]")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"All {len(bitwidth_bin_size)} ranges tested successfully")
    logging.info(f"{'='*80}")
    
    return all_test_results


