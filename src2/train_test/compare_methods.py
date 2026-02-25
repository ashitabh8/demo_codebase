"""
Utility script to compare different quantization methods after training.

This script can be used to:
1. Load results from multiple training runs
2. Create comparison plots across different quantization methods
3. Generate comprehensive method comparison visualizations

Usage:
    python compare_methods.py --experiment_dirs dir1 dir2 dir3 --output_dir comparisons/
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Add src2 to path
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from train_test.quantization_train_test_utils import create_method_comparison_plot

logging.basicConfig(level=logging.INFO)


def load_experiment_results(experiment_dir):
    """
    Load validation results from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
    
    Returns:
        dict: Method results or None if not found
    """
    experiment_path = Path(experiment_dir)
    config_path = experiment_path / "config.yaml"
    
    if not config_path.exists():
        logging.warning(f"Config not found: {config_path}")
        return None
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    quantization_method = config.get('quantization_method', 'unknown')
    
    # Try to load from checkpoint or logs
    # For now, return method name - actual implementation would load from logs/checkpoints
    return {
        'method': quantization_method,
        'config': config
    }


def compare_multiple_methods(experiment_dirs, output_dir=None, writer=None):
    """
    Compare multiple quantization methods from different experiment directories.
    
    Args:
        experiment_dirs: List of experiment directory paths
        output_dir: Output directory for comparison plots
        writer: TensorBoard writer (optional)
    
    Returns:
        matplotlib figure with comparison
    """
    method_results = {}
    
    for exp_dir in experiment_dirs:
        result = load_experiment_results(exp_dir)
        if result:
            method_name = result['method']
            # In a full implementation, you would load actual validation stats
            # For now, this is a template
            method_results[method_name] = {
                'mean_acc': 0.0,  # Would load from actual results
                'std_acc': 0.0,
                'min_acc': 0.0,
                'max_acc': 0.0
            }
    
    if method_results:
        # Get bitwidth options from first config
        bitwidth_options = [4, 6, 8]  # Default, would load from config
        
        fig = create_method_comparison_plot(method_results, bitwidth_options, writer, epoch=0)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path / "method_comparison.png", dpi=300, bbox_inches='tight')
            logging.info(f"Comparison plot saved to {output_path / 'method_comparison.png'}")
        
        return fig
    
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compare quantization methods')
    parser.add_argument('--experiment_dirs', nargs='+', required=True,
                       help='List of experiment directories to compare')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for comparison plots')
    parser.add_argument('--tensorboard_dir', type=str, default=None,
                       help='TensorBoard directory to log comparison')
    
    args = parser.parse_args()
    
    writer = None
    if args.tensorboard_dir:
        writer = SummaryWriter(args.tensorboard_dir)
    
    compare_multiple_methods(args.experiment_dirs, args.output_dir, writer)
    
    if writer:
        writer.close()

