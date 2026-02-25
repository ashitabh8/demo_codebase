import sys
import logging
from pathlib import Path
import torch

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    config = get_config()
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info("\nDataloaders created successfully!")
    
    augmenter = create_augmenter(config, augmentation_mode="no")
    logging.info("Augmenter created successfully!")
    
    # Create model
    model = create_model(config)
    
    # Move model to the appropriate device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")
    
    model.eval()  # Set to evaluation mode for testing
    
    logging.info("\nTesting model on batches...")
    
    for batch_idx, (data, labels, idx) in enumerate(train_loader):
        augmented_data, augmented_labels = apply_augmentation(augmenter, data, labels)
        logging.info(f"\nBatch {batch_idx}:")
        logging.info(f"  Data keys (locations): {list(augmented_data.keys())}")
        
        # Show data shapes for each location and modality
        for loc in augmented_data.keys():
            logging.info(f"  Location '{loc}':")
            for mod in augmented_data[loc].keys():
                logging.info(f"    Modality '{mod}': {augmented_data[loc][mod].shape}")
        
        logging.info(f"  Labels shape: {augmented_labels.shape}")
        
        # Forward pass through the model
        try:
            with torch.no_grad():
                logits = model(augmented_data)
                embeddings = model(augmented_data, return_embeddings=True)
            
            logging.info(f"  Model forward pass successful!")
            logging.info(f"    Output logits shape: {logits.shape}")
            logging.info(f"    Output embeddings shape: {embeddings.shape}")
            logging.info(f"    Predicted classes: {torch.argmax(logits, dim=1)}")
            logging.info(f"    True labels: {augmented_labels}")
            
        except Exception as e:
            logging.error(f"  Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        if batch_idx >= 0:  # Test on just one batch
            break
    
    logging.info("\n✓ All tests passed successfully!")


if __name__ == "__main__":
    main()

