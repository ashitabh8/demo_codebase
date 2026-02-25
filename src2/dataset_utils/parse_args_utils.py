import argparse
import yaml
from pathlib import Path


def parse_args():
    """
    Parse command line arguments for model training/testing.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Load configuration from YAML file for model training/testing'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        help='Name of the experiment'
    )
    
    parser.add_argument(
        '--yaml_path',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        required=False,
        help='GPU to use'
    )
    
    return parser.parse_args()


def parse_test_args():
    """
    Parse command line arguments for vanilla supervised testing.
    
    Simplified version for distillation pipeline - no quantization support.
    Config is automatically loaded from experiment_dir/config.yaml.
    
    Returns:
        args: Parsed arguments with experiment_dir, checkpoint_path, and gpu
    """
    parser = argparse.ArgumentParser(
        description='Test a trained distillation model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with best checkpoint (default)
  python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet_early_exit --gpu 0
  
  # Test with specific checkpoint
  python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet_early_exit \\
                 --checkpoint_path ../experiments/.../models/checkpoint_epoch_10.pth --gpu 0
  
  # Test on CPU
  python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet_early_exit --gpu -1
"""
    )
    
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Path to experiment directory (contains config.yaml and models/)'
    )
    
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to specific checkpoint file (default: experiment_dir/models/best_model.pth)'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID to use for testing (-1 for CPU, default: 0)'
    )
    
    args = parser.parse_args()
    
    return args


def load_yaml_config(yaml_path):
    """
    Load YAML configuration file and return as dictionary.
    
    Args:
        yaml_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary loaded from YAML
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    yaml_file = Path(yaml_path)
    
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    
    with open(yaml_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")


def get_config():
    """
    Single function to parse arguments and load configuration.
    This is the main entry point for getting configuration in main.py.
    
    Returns:
        tuple: (args, config_dict)
            - args: Namespace containing model, dataset, and yaml_path
            - config_dict: Dictionary containing the full YAML configuration
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load YAML configuration
    config = load_yaml_config(args.yaml_path)
    
    # Add command line args to config for easy access
    config['experiment_name'] = args.experiment_name
    config['yaml_path'] = args.yaml_path
    config['device'] = f'cuda:{args.gpu}'
    
    return config

