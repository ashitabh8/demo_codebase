"""
Loss Functions Module

This module provides a flexible interface for different loss functions.
Currently supports CrossEntropyLoss, with easy extension for future loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rel_cost import compute_conv_bitops_uniform

import logging


# =============================================================================
# Loss Classes
# =============================================================================

class CrossEntropyLossForDictOutput(nn.Module):
    """
    CrossEntropyLoss that handles model outputs in dict format.
    
    Single-modal models (ResNet, ConvOnly) used in distillation return:
        {'logits': tensor, 'exits': [...], 'features': tensor}
    
    This loss extracts the 'logits' key and applies CrossEntropyLoss.
    Hardcoded to use CrossEntropyLoss - designed specifically for distillation training.
    
    Args:
        None - uses default CrossEntropyLoss settings
    
    Forward Args:
        model_output: Either a tensor (B, num_classes) or dict with 'logits' key
        target: Ground truth labels (B,)
    
    Returns:
        loss: Scalar loss value
    
    Example:
        >>> loss_fn = CrossEntropyLossForDictOutput()
        >>> outputs = {'logits': torch.randn(32, 10), 'exits': [], 'features': torch.randn(32, 512)}
        >>> labels = torch.randint(0, 10, (32,))
        >>> loss = loss_fn(outputs, labels)
    """
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, model_output, target):
        # Extract logits if output is a dictionary
        if isinstance(model_output, dict):
            logits = model_output['logits']
        else:
            logits = model_output
        
        return self.ce_loss(logits, target)


class CrossEntropyLossWithEarlyExits(nn.Module):
    """
    CrossEntropyLoss for models with early exits.
    
    Calculates CE loss for each early exit and the final exit,
    then returns weighted sum. This allows the model to learn
    at multiple depths simultaneously.
    
    Model output format:
        {'logits': final_tensor, 'exits': [exit1_tensor, exit2_tensor, ...], 'features': ...}
    
    Args:
        exit_weights (list, optional): Weights for each exit [exit1, exit2, ..., final].
                                       Should sum to 1.0 for balanced training.
                                       If None, uses equal weights for all exits.
    
    Forward Args:
        model_output (dict): Model output with 'logits' and 'exits' keys
        target (tensor): Ground truth labels (B,)
    
    Returns:
        loss (tensor): Weighted sum of losses from all exits
    
    Example:
        >>> # Model with 2 early exits + 1 final exit
        >>> loss_fn = CrossEntropyLossWithEarlyExits(exit_weights=[0.3, 0.3, 0.4])
        >>> outputs = {
        ...     'logits': torch.randn(32, 10),  # Final exit
        ...     'exits': [torch.randn(32, 10), torch.randn(32, 10)],  # 2 early exits
        ...     'features': torch.randn(32, 512)
        ... }
        >>> labels = torch.randint(0, 10, (32,))
        >>> loss = loss_fn(outputs, labels)
    """
    def __init__(self, exit_weights=None):
        super().__init__()
        self.exit_weights = exit_weights
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, model_output, target):
        # Extract exits and final logits
        if not isinstance(model_output, dict):
            raise ValueError("Model output must be a dict with 'logits' and 'exits' keys")
        
        final_logits = model_output['logits']
        early_exits = model_output['exits']  # List of tensors
        
        num_early_exits = len(early_exits)
        num_total_exits = num_early_exits + 1  # +1 for final exit
        
        # Use equal weights if not provided
        if self.exit_weights is None:
            weights = [1.0 / num_total_exits] * num_total_exits
        else:
            weights = self.exit_weights
            # Validate weights
            if len(weights) != num_total_exits:
                raise ValueError(f"Expected {num_total_exits} weights, got {len(weights)}")
        
        # Calculate loss for each early exit
        total_loss = 0.0
        for idx, exit_logits in enumerate(early_exits):
            loss = self.ce_loss(exit_logits, target)
            total_loss += weights[idx] * loss
        
        # Add final exit loss
        final_loss = self.ce_loss(final_logits, target)
        total_loss += weights[-1] * final_loss
        
        return total_loss


# =============================================================================
# Loss Factory Function
# =============================================================================

def get_loss_function(stage_config, has_early_exits=False):
    """
    Factory function to get the loss function for distillation training.
    
    Simplified for distillation pipelines - reads loss configuration from stage config
    and returns the appropriate loss function.
    
    Args:
        stage_config (dict): Stage configuration dictionary containing:
            - 'loss_name': Name of the loss function (e.g., 'cross_entropy')
            - 'exit_weights' (optional): Weights for early exits if model has them
        has_early_exits (bool): Whether the model has early exits (auto-detected from model config)
    
    Returns:
        tuple: (loss_fn, loss_name)
            - loss_fn: The loss function instance
            - loss_name: String name of the loss
    
    Example:
        >>> # Standard model (no early exits)
        >>> stage_config = {'loss_name': 'cross_entropy', 'epochs': 50}
        >>> loss_fn, loss_name = get_loss_function(stage_config, has_early_exits=False)
        
        >>> # Model with early exits (detected from model config)
        >>> stage_config = {'loss_name': 'cross_entropy', 'exit_weights': [0.3, 0.3, 0.4]}
        >>> loss_fn, loss_name = get_loss_function(stage_config, has_early_exits=True)
    """
    # Extract loss name from stage config
    loss_name = stage_config['loss_name']
    
    # Log loss function details
    logging.info(f"Loss function: {loss_name}")
    
    # Create loss function based on name
    if loss_name == "cross_entropy":
        # Automatically use early exit loss if model has early exits
        if has_early_exits:
            exit_weights = stage_config.get('exit_weights', None)
            logging.info("  Using CrossEntropyLossWithEarlyExits")
            if exit_weights:
                logging.info(f"  Exit weights: {exit_weights}")
            else:
                logging.info("  Using equal weights for all exits")
            return CrossEntropyLossWithEarlyExits(exit_weights), loss_name
        else:
            logging.info("  Using CrossEntropyLossForDictOutput (handles dict model outputs)")
            return CrossEntropyLossForDictOutput(), loss_name
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Currently only 'cross_entropy' is supported for distillation.")


def convert_to_one_hot(labels, num_classes):
    """
    Convert class indices to one-hot encoded labels.
    
    Args:
        labels: Tensor of shape (batch_size,) containing class indices
        num_classes: Number of classes
    
    Returns:
        one_hot: Tensor of shape (batch_size, num_classes)
    
    Note:
        PyTorch CrossEntropyLoss expects class indices, not one-hot vectors.
        This function is provided for compatibility with other loss functions.
    """
    return F.one_hot(labels, num_classes=num_classes).float()


def convert_from_one_hot(one_hot_labels):
    """
    Convert one-hot encoded labels back to class indices.
    
    Args:
        one_hot_labels: Tensor of shape (batch_size, num_classes)
    
    Returns:
        labels: Tensor of shape (batch_size,) containing class indices
    """
    return torch.argmax(one_hot_labels, dim=1)


class LossWrapper:
    """
    Wrapper class for loss functions that handles label format conversions.
    
    This is useful when your data pipeline outputs one-hot labels but
    your loss function expects class indices (or vice versa).
    """
    
    def __init__(self, loss_fn, expects_one_hot=False):
        """
        Args:
            loss_fn: The underlying loss function
            expects_one_hot: If True, converts class indices to one-hot.
                           If False (default), converts one-hot to class indices.
        """
        self.loss_fn = loss_fn
        self.expects_one_hot = expects_one_hot
    
    def __call__(self, logits, labels):
        """
        Compute loss, handling label format conversion if needed.
        
        Args:
            logits: Model output logits (batch_size, num_classes)
            labels: Labels (either class indices or one-hot)
        
        Returns:
            loss: Scalar loss value
        """
        # Check if labels are one-hot encoded (2D with more than 1 column)
        is_one_hot = len(labels.shape) == 2 and labels.shape[1] > 1
        
        if self.expects_one_hot and not is_one_hot:
            # Convert class indices to one-hot
            num_classes = logits.shape[1]
            labels = convert_to_one_hot(labels, num_classes)
        
        elif not self.expects_one_hot and is_one_hot:
            # Convert one-hot to class indices
            labels = convert_from_one_hot(labels)
        
        return self.loss_fn(logits, labels)




