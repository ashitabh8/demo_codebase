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
        class_weights: Optional list of length C (same order as class indices). Passed
            through training_config['ce_class_weights'] via get_loss_function.
    
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
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, model_output, target):
        # Extract logits if output is a dictionary
        if isinstance(model_output, dict):
            logits = model_output['logits']
        else:
            logits = model_output

        weight = None
        if self.class_weights is not None:
            weight = torch.tensor(
                self.class_weights,
                device=logits.device,
                dtype=logits.dtype,
            )
        return F.cross_entropy(logits, target, weight=weight)


class BCEWithLogitsMultilabelForDictOutput(nn.Module):
    """
    BCEWithLogitsLoss on classifier logits for multi-label targets.

    Model output: dict with 'logits' [B, C] or raw tensor [B, C].
    Target: float tensor [B, C] with values in {0.0, 1.0}.

    Optional pos_weight (length C) is passed through to nn.BCEWithLogitsLoss
    for per-class positive imbalance; supply via training_config bce_pos_weight.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, model_output, target):
        if isinstance(model_output, dict):
            logits = model_output["logits"]
        else:
            logits = model_output
        return self.bce(logits, target.float())


# =============================================================================
# NT-Xent Loss (SimCLR-style Self-Supervised Contrastive Loss)
# =============================================================================

class NTXentLoss(nn.Module):
    """
    NT-Xent Loss — Normalized Temperature-scaled Cross Entropy Loss.

    This is the contrastive loss used in SimCLR (Chen et al., 2020) for
    self-supervised representation learning.

    Core idea
    ---------
    During self-supervised pretraining we don't have class labels. Instead we
    teach the model to recognize that two different augmented views of the SAME
    sample are similar, while views from DIFFERENT samples are dissimilar.

    Given a batch of B samples:
      - We augment each sample twice → proj1 [B, D] and proj2 [B, D]
      - These form B "positive pairs": (proj1[i], proj2[i]) for each i
      - All other 2B-2 items in the batch are "negatives" for that pair

    The loss pulls each positive pair's embeddings together in the projected
    space and pushes them away from all negatives.

    Why project to a lower-dim space first?
    ----------------------------------------
    Applying contrastive loss directly on the backbone features would force
    them to be *completely* invariant to augmentation, which hurts fine-tuning.
    Instead we add a small projection MLP on top of the backbone, apply NT-Xent
    in that projected space, and then discard the projection head after
    pretraining. This lets the backbone features stay rich and informative.

    Args:
        temperature (float): Softmax temperature τ. Lower = sharper distribution,
                             harder negatives. Default 0.5 is typical for SimCLR.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        # Store temperature as a plain Python float.
        # We don't register it as a buffer because it's a scalar hyperparameter,
        # not something that needs to move between devices automatically.
        self.temperature = temperature

    def forward(self, proj1, proj2):
        """
        Compute NT-Xent loss for a batch of projection pairs.

        Args:
            proj1 (Tensor): Projected embeddings for augmentation view 1. Shape: [B, D]
            proj2 (Tensor): Projected embeddings for augmentation view 2. Shape: [B, D]
                            proj1[i] and proj2[i] are two views of the same sample.

        Returns:
            loss (Tensor): Scalar loss value.
        """

        # --- Step 1: Get the batch size ---
        # B is how many original samples are in this batch.
        B = proj1.shape[0]

        # --- Step 2: Concatenate both views and L2-normalize ---
        # We stack all 2B projections into one matrix.
        # torch.cat([proj1, proj2], dim=0) → shape [2B, D]
        #
        # F.normalize(..., dim=1) makes every row a unit vector (length = 1).
        # After normalization, the dot product of two rows equals their
        # cosine similarity, which is what we want to compare.
        #
        # Result: z has shape [2B, D], each row is a unit vector.
        z = F.normalize(torch.cat([proj1, proj2], dim=0), dim=1)  # [2B, D]

        # --- Step 3: Compute all pairwise cosine similarities ---
        # z @ z.T multiplies each row of z against every other row.
        # Since rows are unit vectors, this gives the cosine similarity
        # for every pair (i, j).
        #
        # We then divide by temperature τ. A small τ (e.g. 0.1) makes the
        # distribution very peaked — the model has to be very confident about
        # which pair is the positive. A large τ (e.g. 1.0) softens it.
        #
        # Result: sim has shape [2B, 2B]. sim[i, j] = cos_sim(z[i], z[j]) / τ
        sim = torch.matmul(z, z.T) / self.temperature              # [2B, 2B]

        # --- Step 4: Mask out the diagonal (self-similarity) ---
        # The diagonal of sim[i, i] is always 1.0 / τ (a vector with itself).
        # We don't want the model to "cheat" by treating itself as a positive,
        # so we set the diagonal to a very large negative number (-1e9).
        # After softmax this will become ~0, effectively removing those entries.
        #
        # torch.eye(2*B) makes the identity matrix: 1s on the diagonal, 0s elsewhere.
        # We cast it to bool so masked_fill_ treats it as a binary mask.
        #
        # mask shape: [2B, 2B], True on the diagonal.
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)  # sim is still [2B, 2B]

        # --- Step 5: Build the positive-pair labels ---
        # This is the elegant trick that avoids writing an explicit loop.
        #
        # After concatenation:
        #   Rows 0 .. B-1 in z  → come from proj1 (view 1)
        #   Rows B .. 2B-1 in z → come from proj2 (view 2)
        #
        # For row i in [0, B):     its positive pair is row i+B   (the other view of sample i)
        # For row i in [B, 2B):    its positive pair is row i-B   (the other view of sample i)
        #
        # So the correct "class" for each row is:
        #   [B, B+1, ..., 2B-1,   0, 1, ..., B-1]
        #
        # torch.arange(B, 2*B) produces [B, B+1, ..., 2B-1]  — positives for the first half
        # torch.arange(0, B)   produces [0, 1, ..., B-1]      — positives for the second half
        # cat them together to get a length-2B label vector.
        #
        # labels shape: [2B]
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)

        # --- Step 6: Cross-entropy loss ---
        # F.cross_entropy(sim, labels) treats each row of sim as unnormalized
        # logits over 2B "classes" and applies softmax + negative log-likelihood.
        # The "correct class" for each row is given by labels (the positive pair index).
        #
        # Intuitively: for each anchor row i, out of all 2B rows, the model must
        # assign the highest score to the one true positive. The loss is low when
        # the positive pair has much higher similarity than all the negatives.
        #
        # The diagonal entries are already masked to -1e9 so they don't contribute
        # to the softmax denominator.
        #
        # Returns a scalar — the mean loss over all 2B anchor rows.
        return F.cross_entropy(sim, labels)


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
        ce_w = None
        if "ce_class_weights" in training_config:
            ce_w = training_config["ce_class_weights"]
            if not isinstance(ce_w, list):
                raise ValueError("ce_class_weights must be a list of length C")
            logging.info(f"  CE class weights (len={len(ce_w)}): {ce_w}")
        logging.info("  Using CrossEntropyLossForDictOutput (handles dict model outputs)")
        return CrossEntropyLossForDictOutput(class_weights=ce_w), loss_name

    if loss_name == "ce_supcon":
        # Supervised contrastive learning on embeddings with CE on logits.
        temperature = float(training_config.get("supcon_temperature", 0.07))
        supcon_weight = float(training_config.get("supcon_weight", 1.0))
        ce_w = None
        if "ce_class_weights" in training_config:
            ce_w = training_config["ce_class_weights"]
            if not isinstance(ce_w, list):
                raise ValueError("ce_class_weights must be a list of length C")
            logging.info(f"  CE class weights (len={len(ce_w)}): {ce_w}")
        logging.info(
            "  Using CrossEntropyPlusSupConLoss "
            f"(temperature={temperature}, supcon_weight={supcon_weight})"
        )
        return CrossEntropyPlusSupConLoss(
            temperature=temperature,
            supcon_weight=supcon_weight,
            class_weights=ce_w,
        ), loss_name

    if loss_name == "nt_xent":
        # NT-Xent is used for self-supervised pretraining (SimCLR-style).
        # The temperature controls how sharply the softmax peaks around the
        # positive pair — lower values make the task harder and typically
        # yield better representations.
        # We look up the temperature from the training config; if not set,
        # we fall back to 0.5 (the SimCLR default).
        temperature = float(training_config.get("nt_xent_temperature", 0.5))
        logging.info(f"  Using NTXentLoss (temperature={temperature})")
        return NTXentLoss(temperature=temperature), loss_name

    if loss_name == "bce_multilabel":
        pos_weight = None
        if "bce_pos_weight" in training_config:
            lst = training_config["bce_pos_weight"]
            if not isinstance(lst, list):
                raise ValueError("bce_pos_weight must be a list of length C")
            pos_weight = torch.tensor(lst, dtype=torch.float32)
            logging.info(f"  Using BCEWithLogitsMultilabelForDictOutput (pos_weight len={len(lst)})")
        else:
            logging.info("  Using BCEWithLogitsMultilabelForDictOutput (no pos_weight)")
        return BCEWithLogitsMultilabelForDictOutput(pos_weight=pos_weight), loss_name

    raise ValueError(
        f"Unknown loss function: {loss_name}. "
        f"Supported: 'cross_entropy', 'ce_supcon', 'nt_xent', 'bce_multilabel'."
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

    def __init__(
        self,
        temperature: float = 0.07,
        supcon_weight: float = 1.0,
        class_weights=None,
    ):
        super().__init__()
        self.temperature = temperature
        self.supcon_weight = supcon_weight
        self.class_weights = class_weights

    def _ce_weight_tensor(self, logits):
        if self.class_weights is None:
            return None
        return torch.tensor(
            self.class_weights,
            device=logits.device,
            dtype=logits.dtype,
        )

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

            w1 = self._ce_weight_tensor(logits1)
            w2 = self._ce_weight_tensor(logits2)
            ce = 0.5 * (
                F.cross_entropy(logits1, target, weight=w1)
                + F.cross_entropy(logits2, target, weight=w2)
            )
            supcon = self._supcon_loss_two_views(features1, features2, target)
            return ce + self.supcon_weight * supcon

        # Testing mode: only one set of outputs => CE only.
        logits, _ = self._extract_logits_and_features(model_output)
        w = self._ce_weight_tensor(logits)
        return F.cross_entropy(logits, target, weight=w)


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




