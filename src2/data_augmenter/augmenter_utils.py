import logging
from .Augmenter import Augmenter


class AugmenterConfig:
    def __init__(self, config, experiment_config):
        """
        Args:
            config: Full config dict (for modality_names, location_names, etc.).
            experiment_config: Resolved experiment block (experiments.<name>). Must contain
                "model" and "fixed_augmenters" (no fallbacks).
        """
        if experiment_config is None:
            raise ValueError(
                "experiment_config is required. Augmenter config is taken only from "
                "experiments.<name>; pass the resolved experiment block."
            )
        if "model" not in experiment_config:
            raise ValueError(
                "experiment_config must contain 'model' (model zoo name). "
                f"Keys present: {list(experiment_config.keys())}"
            )
        if "fixed_augmenters" not in experiment_config:
            raise ValueError(
                "experiment_config must contain 'fixed_augmenters' with "
                "time_augmenters and freq_augmenters. No fallback."
            )

        self.device = config.get("device", "cpu")
        self.model = experiment_config["model"]
        self.train_mode = "supervised"
        self.stage = "train"
        self.learn_framework = None

        self.dataset_config = {
            "modality_names": config.get("modality_names", []),
            "location_names": config.get("location_names", []),
            "num_segments": config.get("num_segments", 1),
            "fixed_augmenters": experiment_config["fixed_augmenters"],
        }

        # Add band_limited_noise config if present
        if "band_limited_noise" in config:
            self.dataset_config["band_limited_noise"] = config["band_limited_noise"]

        # Add audio_noise_augmentation config if present
        if "audio_noise_augmentation" in config:
            self.dataset_config["audio_noise_augmentation"] = config["audio_noise_augmentation"]

        # Per-augmenter and dataset shape config from top-level YAML (used by augmenter instances)
        _augmenter_config_keys = (
            "loc_mod_spectrum_len", "time_mask", "freq_mask", "phase_shift", "jitter",
            "mixup", "scaling", "permutation", "negation", "horizontal_flip", "channel_shuffle",
            "time_warp", "mag_warp", "MissModalityGenerator", "specmix", "num_sensors",
        )
        for key in _augmenter_config_keys:
            if key in config:
                self.dataset_config[key] = config[key]


def create_augmenter(config, augmentation_mode="no", experiment_config=None):
    """
    Create an augmenter from configuration dictionary.

    Args:
        config (dict): Configuration dictionary
        augmentation_mode (str): Augmentation mode - "no", "fixed", or "random"
        experiment_config (dict): Resolved experiment block (experiments.<name>). Required;
            must contain "model" and "fixed_augmenters". Single path only, no fallbacks.

    Returns:
        Augmenter: Configured augmenter instance
    """
    if experiment_config is None:
        raise ValueError(
            "experiment_config is required. Augmenter config comes only from "
            "experiments.<name>; pass the resolved experiment block."
        )
    args = AugmenterConfig(config, experiment_config)
    
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

