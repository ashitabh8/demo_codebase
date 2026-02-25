import logging
from .Augmenter import Augmenter


class AugmenterConfig:
    def __init__(self, config):
        self.device = config.get("device", "cpu")
        self.model = config.get("model", "ResNet")
        self.train_mode = "supervised"
        self.stage = "train"
        self.learn_framework = None
        
        self.dataset_config = {
            "modality_names": config.get("modality_names", []),
            "location_names": config.get("location_names", []),
            "num_segments": config.get("num_segments", 1),
        }
        
        if self.model in config and "fixed_augmenters" in config[self.model]:
            self.dataset_config[self.model] = config[self.model]

        # Add band_limited_noise config if present
        if "band_limited_noise" in config:
            self.dataset_config["band_limited_noise"] = config["band_limited_noise"]
        
        # Add audio_noise_augmentation config if present
        if "audio_noise_augmentation" in config:
            self.dataset_config["audio_noise_augmentation"] = config["audio_noise_augmentation"]


def create_augmenter(config, augmentation_mode="no"):
    """
    Create an augmenter from configuration dictionary.
    
    Args:
        config (dict): Configuration dictionary
        augmentation_mode (str): Augmentation mode - "no", "fixed", or "random"
    
    Returns:
        Augmenter: Configured augmenter instance
    """
    args = AugmenterConfig(config)
    
    logging.info(f"Creating augmenter with mode: {augmentation_mode}")
    augmenter = Augmenter(args)
    augmenter.augmentation_mode = augmentation_mode
    
    return augmenter


def apply_augmentation(augmenter, data, labels=None):
    """
    Apply augmentation to data batch.
    
    Args:
        augmenter (Augmenter): Augmenter instance
        data (dict): Multi-modal data dict[location][modality]
        labels (Tensor, optional): Labels
    
    Returns:
        tuple: (augmented_data, labels)
    """
    mode = getattr(augmenter, "augmentation_mode", "no")
    return augmenter.forward(mode, data, labels)

