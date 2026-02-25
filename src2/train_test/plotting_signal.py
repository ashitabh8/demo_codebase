import sys
import logging
import yaml
import torch
from pathlib import Path

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import (
    setup_experiment_dir, train, setup_optimizer, setup_scheduler
)
from train_test.normalize import setup_normalization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """Main training function."""
    
    # ========================================================================
    # 1. Load Configuration
    # ========================================================================
    logging.info("=" * 80)
    logging.info("TRAINING SCRIPT")
    logging.info("=" * 80)
    
    config = get_config()
    logging.info("Configuration loaded successfully")
    logging.info(f"  Model: {config.get('model', 'Unknown')}")
    logging.info(f"  Model variant: {config.get('model_variant', 'None')}")
    logging.info(f"  Dataset: {config.get('yaml_path', 'Unknown')}")
    logging.info(f"  Device: {config.get('device', 'cpu')}")
    
    # ========================================================================
    # 2. Create Dataloaders
    # ========================================================================
    logging.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info("  Train batches: {}".format(len(train_loader)))
    logging.info("  Val batches: {}".format(len(val_loader)))
    logging.info("  Test batches: {}".format(len(test_loader)))

    # plot one sample from train_loader
    for i, (data, labels, idx) in enumerate(train_loader):
        breakpoint()
        print(data.shape)
        print(labels.shape)
        break

if __name__ == "__main__":
    main()