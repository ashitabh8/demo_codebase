import logging
from pathlib import Path
import yaml
import datetime
import numpy as np

from rel_cost import compute_conv_bitops_uniform


def find_latest_experiment(experiments_dir):
    """
    Find the most recent experiment directory.
    
    Args:
        experiments_dir: Base directory containing experiments
    
    Returns:
        experiment_path: Path to the latest experiment directory
    """
    experiments_path = Path(experiments_dir)
    
    if not experiments_path.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
    
    # Get all experiment directories (format: YYYYMMDD_HHMMSS_*)
    experiment_dirs = [d for d in experiments_path.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiments found in: {experiments_dir}")
    
    # Sort by directory name (which includes timestamp)
    experiment_dirs.sort(reverse=True)
    latest_experiment = experiment_dirs[0]
    
    logging.info(f"Found latest experiment: {latest_experiment.name}")
    
    return latest_experiment


def find_all_epoch_checkpoints(models_dir):
    """
    Find all checkpoint_epoch_*.pth files in a models directory.
    
    Args:
        models_dir: Path to the models directory
    
    Returns:
        list: Sorted list of checkpoint paths (by epoch number)
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Find all checkpoint_epoch_*.pth files
    checkpoint_files = list(models_path.glob("checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint_epoch_*.pth files found in: {models_dir}")
    
    # Sort by epoch number
    def get_epoch_num(path):
        # Extract number from checkpoint_epoch_10.pth -> 10
        return int(path.stem.split('_')[-1])
    
    checkpoint_files.sort(key=get_epoch_num)
    
    logging.info(f"Found {len(checkpoint_files)} checkpoint(s): {[f.name for f in checkpoint_files]}")
    
    return checkpoint_files


def get_checkpoint_by_epoch(models_dir, epoch_num):
    """
    Get a specific checkpoint by epoch number.
    
    Args:
        models_dir: Path to the models directory
        epoch_num: Epoch number to find
    
    Returns:
        Path: Path to the checkpoint file
    """
    models_path = Path(models_dir)
    checkpoint_path = models_path / f"checkpoint_epoch_{epoch_num}.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"Found checkpoint: {checkpoint_path.name}")
    
    return checkpoint_path


def load_config_from_experiment(experiment_dir):
    """
    Load configuration from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
    
    Returns:
        config: Configuration dictionary
    """
    config_path = Path(experiment_dir) / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found in experiment: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded config from: {config_path}")
    
    return config


def get_checkpoint_path(experiment_dir, use_best=True):
    """
    Get the path to a checkpoint in an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        use_best: If True, return best_model.pth; else last_epoch.pth
    
    Returns:
        checkpoint_path: Path to the checkpoint file
    """
    models_dir = Path(experiment_dir) / "models"
    
    if use_best:
        checkpoint_path = models_dir / "best_model.pth"
    else:
        checkpoint_path = models_dir / "last_epoch.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path


def setup_test_run_directory(experiment_dir, suffix_to_test_run_dir_name):
    """
    Create a test run directory under the experiment directory and setup file logging.
    
    Args:
        experiment_dir: Path to the experiment directory
        suffix_to_test_run_dir_name: Suffix to add to the test run directory name
    
    Returns:
        test_run_dir: Path to the created test run directory
        log_file_path: Path to the logs.txt file
    """
    experiment_path = Path(experiment_dir)
    
    # Create directory name with date and test function name
    # Format: test_run_YYYYMMDD_HHMMSS_{test_function_name}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_dir_name = f"test_run_{timestamp}_{suffix_to_test_run_dir_name}"
    test_run_dir = experiment_path / test_run_dir_name
    
    # Create the directory
    test_run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created test run directory: {test_run_dir}")
    
    # Setup file logging to logs.txt in the test run directory
    log_file_path = test_run_dir / "logs.txt"
    file_handler = logging.FileHandler(log_file_path, mode='w')  # 'w' mode overwrites, 'a' would append
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Add file handler to root logger
    # This ensures all subsequent logging.info(), logging.error(), etc. go to both console and file
    logging.getLogger().addHandler(file_handler)
    
    # Write a header to the log file to mark when file logging started
    logging.info("=" * 80)
    logging.info(f"TEST RUN LOG FILE - {suffix_to_test_run_dir_name}")
    logging.info(f"Test run directory: {test_run_dir}")
    logging.info(f"Log file: {log_file_path}")
    logging.info("=" * 80)
    
    return test_run_dir, log_file_path


def setup_checkpoint_from_path(test_args):
    """
    Setup testing from a specific checkpoint path.
    Infers experiment directory from checkpoint path structure.
    
    Args:
        checkpoint_path: Path to checkpoint file (str or Path)
    
    Returns:
        tuple: (experiment_dir, checkpoint_paths, config)
            - experiment_dir: Path to experiment directory
            - checkpoint_paths: List containing single checkpoint path
            - config: Configuration dictionary loaded from experiment
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    logging.info("Mode: Testing specific checkpoint")
    checkpoint_path = Path(test_args.checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint_paths = [checkpoint_path]
    
    # Infer experiment directory from checkpoint path
    # Expected structure: experiment_dir/models/checkpoint.pth
    experiment_dir = checkpoint_path.parent.parent
    
    # Load config from experiment
    config = load_config_from_experiment(experiment_dir)
    
    return experiment_dir, checkpoint_paths, config


def setup_checkpoints_from_experiment_dir(test_args):
    """
    Setup testing from an experiment directory with checkpoint selection.
    
    Args:
        experiments_dir: Path to experiment directory (str or Path)
        run_checkpoint: List of epoch numbers to test, or None
        run_all_checkpoints: If True, test all checkpoint_epoch_*.pth files
    
    Returns:
        tuple: (experiment_dir, checkpoint_paths, config)
            - experiment_dir: Path to experiment directory
            - checkpoint_paths: List of checkpoint paths to test
            - config: Configuration dictionary loaded from experiment
    
    Raises:
        FileNotFoundError: If experiment dir, models dir, or checkpoints don't exist
    """
    logging.info("Mode: Testing from experiment directory")
    experiment_dir = Path(test_args.experiments_dir)
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    models_dir = experiment_dir / "models"
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Load config from experiment
    config = load_config_from_experiment(experiment_dir)
    
    # Determine which checkpoints to test
    if test_args.run_checkpoint:
        # Run specific checkpoint(s) by epoch number(s)
        epoch_nums = test_args.run_checkpoint  # This is a list
        logging.info(f"Running checkpoints for epochs: {epoch_nums}")
        checkpoint_paths = []
        for epoch_num in epoch_nums:
            checkpoint_path = get_checkpoint_by_epoch(models_dir, epoch_num)
            checkpoint_paths.append(checkpoint_path)
    
    elif test_args.run_all_checkpoints:
        # Run all checkpoints
        logging.info("Running all checkpoints")
        checkpoint_paths = find_all_epoch_checkpoints(models_dir)
    elif test_args.use_best:
        # Run best model
        logging.info("Running best model")
        checkpoint_path = models_dir / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Best model not found: {checkpoint_path}")
        checkpoint_paths = [checkpoint_path]
    elif test_args.use_last_epoch:
        # Run last epoch
        logging.info("Running last epoch")
        checkpoint_path = models_dir / "last_epoch.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Last epoch not found: {checkpoint_path}")
        checkpoint_paths = [checkpoint_path]
    else:
        raise ValueError("No checkpoint specified, use --run_checkpoint, --run_all_checkpoints, --use_best, or --use_last_epoch")
    
    return experiment_dir, checkpoint_paths, config



def validate_test_args(test_args):
    """
    Validate test arguments.
    
    Args:
        test_args: Test arguments
    
    Returns:
        tuple: (experiment_dir, checkpoint_paths, config)
    """
    if test_args.run_checkpoint and test_args.run_all_checkpoints:
        raise ValueError("Cannot use both --run_checkpoint and --run_all_checkpoints, choose one")
    if test_args.use_best and test_args.use_last_epoch:
        raise ValueError("Cannot use both --use_best and --use_last_epoch, choose one")
    if test_args.override_single_bitwidth and test_args.num_test_configs != 4:
        raise ValueError("Cannot use both --override_single_bitwidth and --num_test_configs, choose one")
    if test_args.run_checkpoint and not test_args.experiments_dir:
        raise ValueError("Cannot use --run_checkpoint without --experiments_dir")
    if test_args.run_all_checkpoints and not test_args.experiments_dir:
        raise ValueError("Cannot use --run_all_checkpoints without --experiments_dir")
    if test_args.use_best and not test_args.experiments_dir:
        raise ValueError("Cannot use --use_best without --experiments_dir")
    if test_args.use_last_epoch and not test_args.experiments_dir:
        raise ValueError("Cannot use --use_last_epoch without --experiments_dir")
    if (test_args.run_checkpoint or test_args.run_all_checkpoints) and (test_args.use_best or test_args.use_last_epoch):
        raise ValueError("Cannot use --run_checkpoint or --run_all_checkpoints with --use_best or --use_last_epoch, choose one")
    if (test_args.test_function == "random_bitwidths") and (test_args.bitwidth_bin_size is None):
        raise ValueError("Cannot use --test_function random_bitwidths without --bitwidth_bin_size")
    return True



def setup_test_function_directory(experiment_dir, test_args):
    """
    Setup test function directory.
    
    Args:
        experiment_dir: Path to experiment directory
        test_args: Test arguments
    """
    test_function_suffix = ""
    if test_args.run_checkpoint:
        test_function_suffix = f"_run_checkpoint_{test_args.run_checkpoint}"
    elif test_args.run_all_checkpoints:
        test_function_suffix = f"_run_all_checkpoints"
    elif test_args.use_best:
        test_function_suffix = f"_use_best"
    elif test_args.use_last_epoch:
        test_function_suffix = f"_use_last_epoch"
    else:
        raise ValueError("No test function specified, use --run_checkpoint, --run_all_checkpoints, --use_best, or --use_last_epoch")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_function_directory_path = (
        experiment_dir
        / "test_functions"
        / f"test_run_{timestamp}_{test_args.test_function}_{test_function_suffix}"
    )
    test_function_directory_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created test function directory: {test_function_directory_path}")
    return test_function_directory_path


# =============================================================================
# Bitwidth Configuration Generation Functions
# =============================================================================

def validate_and_parse_bitwidth_ranges(bitwidth_values, parser=None):
    """
    Validate and parse bitwidth range values into list of (min, max) tuples.
    
    Args:
        bitwidth_values: List of float values from command line
        parser: Optional argparse.ArgumentParser for error reporting
    
    Returns:
        list of tuples: Each tuple is (min, max) for a bitwidth range
    
    Raises:
        ValueError or parser.error: If validation fails
    """
    # Check if we have an even number of values
    if len(bitwidth_values) % 2 != 0:
        error_msg = (
            f"--bitwidth_bin_size requires an even number of values (pairs of min/max), "
            f"but got {len(bitwidth_values)} values: {bitwidth_values}"
        )
        if parser:
            parser.error(error_msg)
        else:
            raise ValueError(error_msg)
    
    # Group into pairs and validate each pair
    ranges = []
    for i in range(0, len(bitwidth_values), 2):
        min_val = bitwidth_values[i]
        max_val = bitwidth_values[i + 1]
        
        # Validate: min <= max
        if min_val > max_val:
            error_msg = (
                f"--bitwidth_bin_size: min must be <= max, "
                f"but got range [{min_val}, {max_val}] at position {i//2 + 1}"
            )
            if parser:
                parser.error(error_msg)
            else:
                raise ValueError(error_msg)
        
        # Validate: both positive
        if min_val <= 0 or max_val <= 0:
            error_msg = (
                f"--bitwidth_bin_size: both min and max must be > 0, "
                f"but got range [{min_val}, {max_val}] at position {i//2 + 1}"
            )
            if parser:
                parser.error(error_msg)
            else:
                raise ValueError(error_msg)
        
        # Validate: reasonable upper bound (bitwidths typically <= 32)
        if min_val > 32 or max_val > 32:
            error_msg = (
                f"--bitwidth_bin_size: both min and max must be <= 32, "
                f"but got range [{min_val}, {max_val}] at position {i//2 + 1}"
            )
            if parser:
                parser.error(error_msg)
            else:
                raise ValueError(error_msg)
        
        ranges.append((min_val, max_val))
    
    return ranges


def validate_and_parse_relative_memory_ranges(relative_memory_values, parser=None):
    """
    Validate and parse relative memory range values into list of (min, max) tuples.

    Expected values are floats in [0, 1]. Values are provided as pairs:
      --relative_memory_consumption_bin_size min1 max1 min2 max2 ...

    Args:
        relative_memory_values: List of float values from command line
        parser: Optional argparse.ArgumentParser for error reporting

    Returns:
        list[tuple[float, float]]: Each tuple is (min, max) for a relative memory range
    """
    if len(relative_memory_values) % 2 != 0:
        error_msg = (
            f"--relative_memory_consumption_bin_size requires an even number of values "
            f"(pairs of min/max), but got {len(relative_memory_values)} values: {relative_memory_values}"
        )
        if parser:
            parser.error(error_msg)
        raise ValueError(error_msg)

    ranges = []
    for i in range(0, len(relative_memory_values), 2):
        min_val = float(relative_memory_values[i])
        max_val = float(relative_memory_values[i + 1])

        if min_val > max_val:
            error_msg = (
                f"--relative_memory_consumption_bin_size: min must be <= max, "
                f"but got range [{min_val}, {max_val}] at position {i//2 + 1}"
            )
            if parser:
                parser.error(error_msg)
            raise ValueError(error_msg)

        # Relative memory is a ratio; keep it in [0, 1]
        if min_val < 0.0 or max_val < 0.0 or min_val > 1.0 or max_val > 1.0:
            error_msg = (
                f"--relative_memory_consumption_bin_size: values must be in [0, 1], "
                f"but got range [{min_val}, {max_val}] at position {i//2 + 1}"
            )
            if parser:
                parser.error(error_msg)
            raise ValueError(error_msg)

        ranges.append((min_val, max_val))

    return ranges


def get_layer_weight_counts(model, conv_class):
    """
    Extract the number of weights for each quantized conv layer in the model.
    
    Args:
        model: PyTorch model containing quantized Conv layers
        conv_class: The quantized Conv class type
    
    Returns:
        list: Number of weights per layer (in module order)
    """
    weight_counts = []
    for module in model.modules():
        if isinstance(module, conv_class):
            weight_counts.append(module.weight.numel())
    return weight_counts


def compute_relative_memory(scheme, weight_counts, max_bitwidth=8):
    """
    Compute relative BitOps consumption for a quantization scheme.
    
    Relative BitOps = (Σ Nw_i × bw_i²) / (Σ Nw_i × max_bitwidth²)
    
    Assumes weight and activation bitwidths are equal (bw_w = bw_a = bw).
    
    Args:
        scheme: List of bitwidths, one per layer (quantization scheme)
        weight_counts: List of weight counts per layer
        max_bitwidth: Maximum bitwidth (default 8 for 8-bit reference)
    
    Returns:
        float: Relative BitOps consumption (0 to 1, where 1 = max_bitwidth for all)
    """
    assert len(scheme) == len(weight_counts), \
        f"Scheme length ({len(scheme)}) must match weight_counts length ({len(weight_counts)})"
    
    current_bitops = sum(nw * bw ** 2 for nw, bw in zip(weight_counts, scheme))
    max_bitops = sum(nw * max_bitwidth ** 2 for nw in weight_counts)
    
    if max_bitops == 0:
        return 0.0
    
    return current_bitops / max_bitops


def get_scheme_given_2bitops(num_layers, bitops_low, bitops_high, b_low, b_high, target_bitops):
    """
    Generate a quantization scheme for the 2-bitwidth case.
    
    Args:
        num_layers: Number of layers
        bitops_low: List of BitOPs for each layer at low bitwidth
        bitops_high: List of BitOPs for each layer at high bitwidth
        b_low: Low bitwidth value
        b_high: High bitwidth value
        target_bitops: Target total BitOPs to achieve
    
    Returns:
        scheme: List of bitwidths for each layer
        current_bitops: Actual total BitOPs achieved
    """
    scheme = [b_low] * num_layers
    current_bitops = sum(bitops_low)
    
    sorted_indices = list(range(num_layers))
    np.random.shuffle(sorted_indices)
    
    for idx in sorted_indices:
        if current_bitops >= target_bitops:
            break
        delta = bitops_high[idx] - bitops_low[idx]
        if current_bitops + delta <= target_bitops:
            scheme[idx] = b_high
            current_bitops += delta
        else:
            if abs(current_bitops + delta - target_bitops) < abs(current_bitops - target_bitops):
                scheme[idx] = b_high
                current_bitops += delta
    
    return scheme, current_bitops

def get_scheme_given_3bitops(num_layers, layers, input_shapes, output_shapes, 
                              options, target_bitops, max_bitops,
                              bin_min, bin_max, baseline_bitwidth=8):
    """
    Generate a quantization scheme for the 3+ bitwidth case.
    Greedy algorithm that can upgrade any layer through all bitwidths.
    
    Args:
        num_layers: Number of layers
        layers: List of layer modules
        input_shapes: List of input shapes for each layer
        output_shapes: List of output shapes for each layer
        options: Sorted list of bitwidth options (e.g., [2, 4, 8])
        target_bitops: Target total BitOPs to achieve
        max_bitops: Maximum possible BitOPs (all layers at highest bitwidth)
        bin_min: Minimum relative BitOPs for valid scheme
        bin_max: Maximum relative BitOPs for valid scheme
        baseline_bitwidth: Baseline bitwidth for comparison
    
    Returns:
        scheme: List of bitwidths for each layer
    """
    # Precompute BitOPs for all layers at all bitwidths
    all_bitops = {bw: [] for bw in options}
    for i in range(num_layers):
        layer, in_shape, out_shape = layers[i], input_shapes[i], output_shapes[i]
        for bw in options:
            all_bitops[bw].append(
                compute_conv_bitops_uniform(layer, in_shape, out_shape, bw, baseline_bitwidth)['bitops']
            )
    
    min_bitops = sum(all_bitops[options[0]])
    
    scheme = [options[0]] * num_layers
    current_bitops = min_bitops
    
    layer_order = np.random.permutation(num_layers).tolist()
    
    while current_bitops < target_bitops:
        upgraded = False
        for layer_idx in layer_order:
            if current_bitops >= target_bitops:
                break
            
            curr_bw = scheme[layer_idx]
            curr_bw_idx = options.index(curr_bw)
            
            if curr_bw_idx < len(options) - 1:
                best_new_bw = None
                for next_bw in options[curr_bw_idx + 1:]:
                    # Use precomputed bitops instead of weight_counts * bw²
                    delta = all_bitops[next_bw][layer_idx] - all_bitops[curr_bw][layer_idx]
                    if current_bitops + delta <= target_bitops:
                        best_new_bw = next_bw
                    else:
                        if best_new_bw is None:
                            if abs(current_bitops + delta - target_bitops) < abs(current_bitops - target_bitops):
                                best_new_bw = next_bw
                        break
                
                if best_new_bw is not None:
                    old_bw = scheme[layer_idx]
                    # Use precomputed bitops for delta
                    current_bitops += all_bitops[best_new_bw][layer_idx] - all_bitops[old_bw][layer_idx]
                    scheme[layer_idx] = best_new_bw
                    upgraded = True
        
        if not upgraded:
            break
    
    return scheme


def generate_scheme_for_target_relative_memory(model, conv_class, bitwidth_options, target_relative_memory, 
                                               bin_min, bin_max, 
                                               layers, input_shapes, output_shapes, baseline_bitwidth=8): # NEW ARGS
    """
    Generate a quantization scheme that achieves a target relative BitOps.
    
    Relative BitOps = (Σ Nw_i × bw_i²) / (Σ Nw_i × max_bw²)
    """
    options = sorted(bitwidth_options)
    max_bw = options[-1]
    min_bw = options[0]
    
    weight_counts = get_layer_weight_counts(model, conv_class) # THIS IS FILTER SIZE NOW
    # num_layers = len(weight_counts)
    num_layers = len(layers)

    if num_layers == 0:
        return []

    bitops_low = []
    bitops_high = []
    for layer, in_shape, out_shape in zip(layers, input_shapes, output_shapes):
        bitops_low.append(compute_conv_bitops_uniform(layer, in_shape, out_shape, min_bw, baseline_bitwidth)['bitops'])
        bitops_high.append(compute_conv_bitops_uniform(layer, in_shape, out_shape, max_bw, baseline_bitwidth)['bitops'])
    
    # BitOps bounds (squared bitwidths) TODO:THIS WILL CHANGE BECAUSE THIS BITCOST IS WRONG WE HAVE NEW COST FUNCTIONS
    # max_bitops = sum(nw * max_bw ** 2 for nw in weight_counts)
    # min_bitops = sum(nw * min_bw ** 2 for nw in weight_counts)

    min_bitops = sum(bitops_low)
    max_bitops = sum(bitops_high)
    
    target_bitops = target_relative_memory * max_bitops  # ✅ Fixed
    target_bitops = np.clip(target_bitops, min_bitops, max_bitops)
    
    if len(options) == 2:
        b_low, b_high = options

        
        scheme, current_bitops = get_scheme_given_2bitops(num_layers, bitops_low, bitops_high, b_low, b_high, target_bitops)

    else:
        # 3+ bitwidth case: fix some layers at bit_high, run 2-case on rest
        scheme = get_scheme_given_3bitops(
            num_layers, layers, input_shapes, output_shapes,
            options, target_bitops, max_bitops,
            bin_min, bin_max, baseline_bitwidth
        )
    
    return scheme


def generate_schemes_in_relative_memory_bin(model, conv_class, bitwidth_options, 
                                            bin_min, bin_max, num_schemes,
                                            layers, input_shapes, output_shapes,
                                            baseline_bitwidth=8,
                                            max_attempts_per_scheme=100):
    """
    Generate multiple quantization schemes with relative BitOps in [bin_min, bin_max].
    
    Returns:
        list: List of valid schemes, or None if generation failed
    """
    # weight_counts = get_layer_weight_counts(model, conv_class)
    # max_bw = max(bitwidth_options)
    
    schemes = []
    
    for i in range(num_schemes):
        scheme = None
        for attempt in range(max_attempts_per_scheme):
            target_relative_memory = np.random.uniform(bin_min, bin_max)
            scheme = generate_scheme_for_target_relative_memory(
                model, conv_class, bitwidth_options, target_relative_memory,bin_min, bin_max, layers, input_shapes, output_shapes, baseline_bitwidth=8
            )

            # actual_rel_memory = compute_relative_memory(scheme, weight_counts, max_bitwidth=max_bw)
            
            

                        # Compute total relative bitops across all layers
            total_bitops = 0
            total_bitops_baseline = 0
            for lyr, in_shape, out_shape, bw in zip(layers, input_shapes, output_shapes, scheme):
                result = compute_conv_bitops_uniform(lyr, in_shape, out_shape, bw, baseline_bitwidth)
                total_bitops += result['bitops']
                total_bitops_baseline += result['bitops_baseline']
            
            actual_rel_memory = total_bitops / total_bitops_baseline if total_bitops_baseline > 0 else 0

            if bin_min <= actual_rel_memory <= bin_max:
                schemes.append((actual_rel_memory, scheme)) 
                break
        else:
            # Max attempts exhausted - return None to signal failure

            raise ValueError(f"Invalid test function: {test_args.test_function}, "
                              f"Failed to generate scheme {i+1}/{num_schemes} within bin [{bin_min}, {bin_max}] "
                              f"after {max_attempts_per_scheme} attempts")
            # logging.warning(
            #     f"Failed to generate scheme {i+1}/{num_schemes} within bin [{bin_min}, {bin_max}] "
            #     f"after {max_attempts_per_scheme} attempts"
            # )
            # return None
    
    return schemes


def get_num_quantized_layers(model, conv_class):
    """
    Count the number of quantized convolutional layers in the model.
    
    Args:
        model: PyTorch model
        conv_class: The quantized Conv class type to count
    
    Returns:
        int: Number of quantized layers
    """
    return sum(1 for m in model.modules() if isinstance(m, conv_class))


# =============================================================================
# Results Saving Functions
# =============================================================================

def save_single_precision_results_to_csv(test_results, output_path, checkpoint_name=None):
    """
    Save single precision test results to CSV file.
    
    Args:
        test_results: Dictionary with bitwidth as keys and test metrics as values
                     Format: {4: {'accuracy': 0.85, 'loss': 0.23}, ...}
        output_path: Path to save CSV file (can be Path object or string)
        checkpoint_name: Optional checkpoint name to include in CSV
    
    Returns:
        Path: Path to saved CSV file
    """
    import csv
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    rows = []
    for bitwidth, metrics in sorted(test_results.items()):
        row = {
            'bitwidth': bitwidth,
            'accuracy': metrics.get('accuracy', 'N/A'),
            'loss': metrics.get('loss', 'N/A')
        }
        if checkpoint_name:
            row['checkpoint'] = checkpoint_name
        rows.append(row)
    
    # Write to CSV
    if rows:
        fieldnames = ['checkpoint', 'bitwidth', 'accuracy', 'loss'] if checkpoint_name else ['bitwidth', 'accuracy', 'loss']
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logging.info(f"Single precision results saved to: {output_path}")
    else:
        logging.warning("No test results to save")
    
    return output_path