"""
Normalization utilities for training data.

This module provides various normalization methods (StandardScaler, MinMaxScaler, etc.)
that can be configured via YAML and applied to multi-modal sensor data.

The normalizers are fit on training data and then applied to train/val/test data.
"""

import torch
import logging
from typing import Dict, Optional, Tuple


class BaseNormalizer:
    """Base class for all normalizers."""
    
    def __init__(self):
        self.statistics = {}
        self.is_fitted = False
    
    def fit(self, data_loader):
        """
        Compute normalization statistics from training data.
        
        Args:
            data_loader: DataLoader for training data
        """
        raise NotImplementedError
    
    def transform(self, data):
        """
        Apply normalization to data.
        
        Args:
            data: Multi-modal data dict[location][modality] = Tensor
        
        Returns:
            Normalized data with same structure
        """
        raise NotImplementedError
    
    def fit_transform(self, data_loader):
        """Fit and transform in one step."""
        self.fit(data_loader)
        return self


class StandardScaler(BaseNormalizer):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Formula: z = (x - mean) / std
    """
    
    def __init__(self, epsilon=1e-8):
        """
        Args:
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def fit(self, data_loader):
        """
        Compute mean and std for each location and modality.
        
        Args:
            data_loader: DataLoader for training data
        """
        logging.info("Fitting StandardScaler on training data...")
        
        # Collect all data to compute statistics
        location_modality_data = {}
        
        for batch_idx, (data, labels, indices) in enumerate(data_loader):
            for location in data:
                if location not in location_modality_data:
                    location_modality_data[location] = {}
                
                for modality in data[location]:
                    if modality not in location_modality_data[location]:
                        location_modality_data[location][modality] = []
                    
                    # Flatten all dimensions except batch
                    # Shape: [batch, ...] -> [batch * ...]
                    batch_data = data[location][modality].reshape(-1)
                    location_modality_data[location][modality].append(batch_data)
            
            if (batch_idx + 1) % 50 == 0:
                logging.info(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Compute mean and std for each location-modality pair
        for location in location_modality_data:
            if location not in self.statistics:
                self.statistics[location] = {}
            
            for modality in location_modality_data[location]:
                # Concatenate all batches
                all_data = torch.cat(location_modality_data[location][modality])
                
                mean = all_data.mean().item()
                std = all_data.std().item()
                
                # Avoid division by zero
                if std < self.epsilon:
                    std = 1.0
                    logging.warning(f"  {location}/{modality}: std is very small ({std:.2e}), setting to 1.0")
                
                self.statistics[location][modality] = {
                    'mean': mean,
                    'std': std
                }
                
                logging.info(f"  {location}/{modality}: mean={mean:.4f}, std={std:.4f}")
        
        self.is_fitted = True
        logging.info("StandardScaler fitted successfully")
    
    def transform(self, data):
        """
        Apply standardization to data.
        
        Args:
            data: Multi-modal data dict[location][modality] = Tensor
        
        Returns:
            Normalized data with same structure
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        normalized_data = {}
        
        for location in data:
            normalized_data[location] = {}
            
            for modality in data[location]:
                if location not in self.statistics or modality not in self.statistics[location]:
                    logging.warning(f"No statistics for {location}/{modality}, skipping normalization")
                    normalized_data[location][modality] = data[location][modality]
                    continue
                
                mean = self.statistics[location][modality]['mean']
                std = self.statistics[location][modality]['std']
                
                # Apply standardization: (x - mean) / std
                normalized_data[location][modality] = (data[location][modality] - mean) / std
        
        return normalized_data


class MinMaxScaler(BaseNormalizer):
    """
    Scale features to a given range (default [0, 1]).
    
    Formula: x_scaled = (x - min) / (max - min) * (feature_range[1] - feature_range[0]) + feature_range[0]
    """
    
    def __init__(self, feature_range=(0, 1), epsilon=1e-8):
        """
        Args:
            feature_range: Desired range of transformed data (min, max)
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.feature_range = feature_range
        self.epsilon = epsilon
    
    def fit(self, data_loader):
        """
        Compute min and max for each location and modality.
        
        Args:
            data_loader: DataLoader for training data
        """
        logging.info(f"Fitting MinMaxScaler (range={self.feature_range}) on training data...")
        
        # Collect all data to compute statistics
        location_modality_data = {}
        
        for batch_idx, (data, labels, indices) in enumerate(data_loader):
            for location in data:
                if location not in location_modality_data:
                    location_modality_data[location] = {}
                
                for modality in data[location]:
                    if modality not in location_modality_data[location]:
                        location_modality_data[location][modality] = []
                    
                    # Flatten all dimensions except batch
                    batch_data = data[location][modality].reshape(-1)
                    location_modality_data[location][modality].append(batch_data)
            
            if (batch_idx + 1) % 50 == 0:
                logging.info(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Compute min and max for each location-modality pair
        for location in location_modality_data:
            if location not in self.statistics:
                self.statistics[location] = {}
            
            for modality in location_modality_data[location]:
                # Concatenate all batches
                all_data = torch.cat(location_modality_data[location][modality])
                
                data_min = all_data.min().item()
                data_max = all_data.max().item()
                
                # Avoid division by zero
                if abs(data_max - data_min) < self.epsilon:
                    data_range = 1.0
                    logging.warning(f"  {location}/{modality}: range is very small, setting to 1.0")
                else:
                    data_range = data_max - data_min
                
                self.statistics[location][modality] = {
                    'min': data_min,
                    'max': data_max,
                    'range': data_range
                }
                
                logging.info(f"  {location}/{modality}: min={data_min:.4f}, max={data_max:.4f}")
        
        self.is_fitted = True
        logging.info("MinMaxScaler fitted successfully")
    
    def transform(self, data):
        """
        Apply min-max scaling to data.
        
        Args:
            data: Multi-modal data dict[location][modality] = Tensor
        
        Returns:
            Normalized data with same structure
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        normalized_data = {}
        scale = self.feature_range[1] - self.feature_range[0]
        min_val = self.feature_range[0]
        
        for location in data:
            normalized_data[location] = {}
            
            for modality in data[location]:
                if location not in self.statistics or modality not in self.statistics[location]:
                    logging.warning(f"No statistics for {location}/{modality}, skipping normalization")
                    normalized_data[location][modality] = data[location][modality]
                    continue
                
                data_min = self.statistics[location][modality]['min']
                data_range = self.statistics[location][modality]['range']
                
                # Apply min-max scaling: (x - min) / range * scale + min_val
                normalized_data[location][modality] = \
                    (data[location][modality] - data_min) / data_range * scale + min_val
        
        return normalized_data


class MaxAbsScaler(BaseNormalizer):
    """
    Scale features by their maximum absolute value.
    
    This scaler scales data to [-1, 1] range by dividing by the maximum absolute value.
    Useful when data is already centered at zero or contains both positive and negative values.
    
    Formula: x_scaled = x / max(abs(x))
    """
    
    def __init__(self, epsilon=1e-8):
        """
        Args:
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def fit(self, data_loader):
        """
        Compute max absolute value for each location and modality.
        
        Args:
            data_loader: DataLoader for training data
        """
        logging.info("Fitting MaxAbsScaler on training data...")
        
        # Collect all data to compute statistics
        location_modality_data = {}
        
        for batch_idx, (data, labels, indices) in enumerate(data_loader):
            for location in data:
                if location not in location_modality_data:
                    location_modality_data[location] = {}
                
                for modality in data[location]:
                    if modality not in location_modality_data[location]:
                        location_modality_data[location][modality] = []
                    
                    # Flatten all dimensions except batch
                    batch_data = data[location][modality].reshape(-1)
                    location_modality_data[location][modality].append(batch_data)
            
            if (batch_idx + 1) % 50 == 0:
                logging.info(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Compute max absolute value for each location-modality pair
        for location in location_modality_data:
            if location not in self.statistics:
                self.statistics[location] = {}
            
            for modality in location_modality_data[location]:
                # Concatenate all batches
                all_data = torch.cat(location_modality_data[location][modality])
                
                max_abs = all_data.abs().max().item()
                
                # Avoid division by zero
                if max_abs < self.epsilon:
                    max_abs = 1.0
                    logging.warning(f"  {location}/{modality}: max_abs is very small, setting to 1.0")
                
                self.statistics[location][modality] = {
                    'max_abs': max_abs
                }
                
                logging.info(f"  {location}/{modality}: max_abs={max_abs:.4f}")
        
        self.is_fitted = True
        logging.info("MaxAbsScaler fitted successfully")
    
    def transform(self, data):
        """
        Apply max absolute value scaling to data.
        
        Args:
            data: Multi-modal data dict[location][modality] = Tensor
        
        Returns:
            Normalized data with same structure
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        normalized_data = {}
        
        for location in data:
            normalized_data[location] = {}
            
            for modality in data[location]:
                if location not in self.statistics or modality not in self.statistics[location]:
                    logging.warning(f"No statistics for {location}/{modality}, skipping normalization")
                    normalized_data[location][modality] = data[location][modality]
                    continue
                
                max_abs = self.statistics[location][modality]['max_abs']
                
                # Apply max absolute value scaling: x / max_abs
                normalized_data[location][modality] = data[location][modality] / max_abs
        
        return normalized_data


class NoNormalizer(BaseNormalizer):
    """
    No normalization - pass through the data unchanged.
    Useful as a default/baseline.
    """
    
    def fit(self, data_loader):
        """No-op."""
        logging.info("Using NoNormalizer (no normalization applied)")
        self.is_fitted = True
    
    def transform(self, data):
        """Return data unchanged."""
        return data


def create_normalizer(config: dict) -> BaseNormalizer:
    """
    Create a normalizer based on configuration.
    
    Args:
        config: Configuration dictionary from YAML
    
    Returns:
        Normalizer instance
    
    Example config:
        normalization:
            method: "standard"  # Options: "standard", "minmax", "maxabs", "none"
            minmax_range: [0, 1]  # Only for minmax
    """
    norm_config = config.get("normalization", {})
    method = norm_config.get("method", "none").lower()
    
    if method == "standard" or method == "standardscaler":
        normalizer = StandardScaler()
        logging.info("Created StandardScaler")
    
    elif method == "minmax" or method == "minmaxscaler":
        feature_range = norm_config.get("minmax_range", [0, 1])
        normalizer = MinMaxScaler(feature_range=tuple(feature_range))
        logging.info(f"Created MinMaxScaler with range {feature_range}")
    
    elif method == "maxabs" or method == "maxabsscaler":
        normalizer = MaxAbsScaler()
        logging.info("Created MaxAbsScaler")
    
    elif method == "none" or method is None:
        normalizer = NoNormalizer()
        logging.info("No normalization will be applied")
    
    else:
        logging.warning(f"Unknown normalization method: {method}. Using no normalization.")
        normalizer = NoNormalizer()
    
    return normalizer


class NormalizingDataLoader:
    """
    Wrapper around DataLoader that applies normalization on-the-fly.
    
    This allows seamless integration with existing training code while applying
    normalization to each batch as it's loaded.
    """
    
    def __init__(self, data_loader, normalizer: BaseNormalizer):
        """
        Args:
            data_loader: Original PyTorch DataLoader
            normalizer: Fitted normalizer instance
        """
        self.data_loader = data_loader
        self.normalizer = normalizer
        self.dataset = data_loader.dataset
    
    def __iter__(self):
        """Iterate over batches, applying normalization."""
        for data, labels, indices in self.data_loader:
            normalized_data = self.normalizer.transform(data)
            yield normalized_data, labels, indices
    
    def __len__(self):
        """Return number of batches."""
        return len(self.data_loader)


def setup_normalization(train_loader, val_loader, test_loader, config: dict) -> Tuple:
    """
    Setup normalization for all data loaders.
    
    This function:
    1. Creates a normalizer based on config
    2. Fits it on training data
    3. Wraps all loaders to apply normalization
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration dictionary
    
    Returns:
        Tuple of (normalized_train_loader, normalized_val_loader, normalized_test_loader)
    """
    # Create normalizer
    normalizer = create_normalizer(config)
    
    # Fit on training data
    normalizer.fit(train_loader)
    
    # Wrap all loaders
    normalized_train_loader = NormalizingDataLoader(train_loader, normalizer)
    normalized_val_loader = NormalizingDataLoader(val_loader, normalizer)
    normalized_test_loader = NormalizingDataLoader(test_loader, normalizer)
    
    return normalized_train_loader, normalized_val_loader, normalized_test_loader

