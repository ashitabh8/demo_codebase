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



# =============================================================================
# Loss Factory Function
# =============================================================================

def get_loss_function(training_config):
    """
    Factory function to get the loss function for training.

    Reads loss configuration from training config and returns the appropriate
    loss function.

    Args:
        training_config (dict): Training configuration dictionary containing:
            - 'loss_name': Name of the loss function (e.g., 'cross_entropy')

    Returns:
        tuple: (loss_fn, loss_name)
            - loss_fn: The loss function instance
            - loss_name: String name of the loss

    Example:
        >>> training_config = {'loss_name': 'cross_entropy', 'epochs': 50}
        >>> loss_fn, loss_name = get_loss_function(training_config)
    """
    # Extract loss name from training config
    loss_name = training_config['loss_name']

    # Log loss function details
    logging.info(f"Loss function: {loss_name}")

    # Create loss function based on name
    if loss_name == "cross_entropy":
        logging.info("  Using CrossEntropyLossForDictOutput (handles dict model outputs)")
        return CrossEntropyLossForDictOutput(), loss_name

    if loss_name == "ce_supcon":
        # Supervised contrastive learning on embeddings with CE on logits.
        temperature = float(training_config.get("supcon_temperature", 0.07))
        supcon_weight = float(training_config.get("supcon_weight", 1.0))
        logging.info(
            "  Using CrossEntropyPlusSupConLoss "
            f"(temperature={temperature}, supcon_weight={supcon_weight})"
        )
        return CrossEntropyPlusSupConLoss(
            temperature=temperature,
            supcon_weight=supcon_weight,
        ), loss_name

    raise ValueError(
        f"Unknown loss function: {loss_name}. "
        f"Supported: 'cross_entropy', 'ce_supcon'."
    )


class CrossEntropyPlusSupConLoss(nn.Module):
    """
    Combined objective for supervised contrastive learning:
      - CE loss on classifier logits (average across 2 views)
      - Supervised Contrastive (SupCon) loss on embedding features (2 views)

    Expected model outputs during training:
      forward((out_view1, out_view2), labels)
    Each out_* should be a dict with:
      - 'logits': [B, C]
      - 'features': [B, D]

    During testing, we support forward(out, labels) and only compute CE
    (SupCon term is not computed because there is only one view).
    """

    def __init__(self, temperature: float = 0.07, supcon_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.supcon_weight = supcon_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def _extract_logits_and_features(self, outputs):
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            features = outputs["features"]
        else:
            # Fallback: treat outputs as logits only.
            logits = outputs
            features = None
        return logits, features

    def _supcon_loss_two_views(self, features1, features2, labels):
        """
        Two-view SupCon loss (Khosla et al.).

        Args:
          features1: [B, D]
          features2: [B, D]
          labels: [B] class indices
        """
        if features1 is None or features2 is None:
            # If features aren't available, contrastive term is undefined.
            device = labels.device
            return torch.zeros((), device=device, dtype=torch.float32) + 0.0

        # [2B, D]
        z = torch.cat([features1, features2], dim=0)
        z = F.normalize(z, dim=1)

        # [2B]
        labels_2v = torch.cat([labels, labels], dim=0)

        # Similarity matrix: [2B, 2B]
        sim = torch.matmul(z, z.T) / self.temperature

        # Stability: subtract row-wise max from logits before exp/log.
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Mask out self-contrast.
        n = z.shape[0]
        device = z.device
        self_mask = torch.eye(n, dtype=torch.bool, device=device)

        exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
        denom = exp_sim.sum(dim=1, keepdim=True)  # [2B, 1]
        log_prob = sim - torch.log(denom + 1e-12)  # [2B, 2B]

        # Positive mask: same label, exclude self.
        labels_row = labels_2v.unsqueeze(0)  # [1, 2B]
        labels_col = labels_2v.unsqueeze(1)  # [2B, 1]
        positive_mask = (labels_col == labels_row) & (~self_mask)  # [2B, 2B]

        positive_count = positive_mask.sum(dim=1)  # [2B]
        positive_count_f = positive_count.to(log_prob.dtype)

        # Mean over positives for each anchor i.
        mean_log_prob_pos = (log_prob * positive_mask.to(log_prob.dtype)).sum(dim=1) / torch.clamp(
            positive_count_f, min=1.0
        )  # [2B]

        loss_per_anchor = -mean_log_prob_pos

        valid = (positive_count > 0).to(loss_per_anchor.dtype)
        denom_valid = valid.sum().clamp(min=1.0)
        return (loss_per_anchor * valid).sum() / denom_valid

    def forward(self, model_output, target):
        # Training mode: (out1, out2)
        if isinstance(model_output, (tuple, list)) and len(model_output) == 2:
            out1, out2 = model_output
            logits1, features1 = self._extract_logits_and_features(out1)
            logits2, features2 = self._extract_logits_and_features(out2)

            ce = 0.5 * (self.ce_loss(logits1, target) + self.ce_loss(logits2, target))
            supcon = self._supcon_loss_two_views(features1, features2, target)
            return ce + self.supcon_weight * supcon

        # Testing mode: only one set of outputs => CE only.
        logits, _ = self._extract_logits_and_features(model_output)
        return self.ce_loss(logits, target)


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




