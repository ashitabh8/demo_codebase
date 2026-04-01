"""
QuantBase — shared fixed-scale fake-quantization primitives.

Contains the parametric layer classes (QuantDWConv2d, QuantDWConv1d) and
all calibration / export helpers shared by W8A8Quant and W8A16Quant.

Do NOT import this directly in model code. Use:
  - W8A8Quant  for INT8 weights + INT8 activations
  - W8A16Quant for INT8 weights + INT16 activations

Both provide QuantDWConv2d / QuantDWConv1d with the correct bit-width
locked in, plus re-export all helpers from this module.

Design
------
Weights:     per-output-channel symmetric INT8 [-128, 127].
             Scale is computed dynamically from max(|w|) per channel each
             forward pass — always in sync with the current weight values.

Activations: per-tensor symmetric INT8 or INT16 depending on act_bits.
             Scale is computed from a calibration pass (running max over
             N batches), then frozen for the rest of training.

STE (Straight-Through Estimator) passes gradients through both round()
and clamp(), so the model trains end-to-end despite the discrete grid.

BN and GELU stay float32 throughout (standard QAT practice; BN is
folded into the preceding conv at export time).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INT8_MAX = 127
INT8_MIN = -128

INT16_MAX = 32767
INT16_MIN = -32768


# ---------------------------------------------------------------------------
# Core fake-quant primitive
# ---------------------------------------------------------------------------

def _fake_quant_symmetric(
    x: torch.Tensor,
    scale: torch.Tensor,
    int_min: int = INT8_MIN,
    int_max: int = INT8_MAX,
) -> torch.Tensor:
    """
    Symmetric fixed-point fake-quantization with STE.

    Forward:  x_q = clamp(round(x / scale), int_min, int_max) * scale
    Backward: gradient passes straight through (STE).

    Args:
        x:       input tensor
        scale:   broadcastable to x; must be > 0
        int_min: lower integer clamp bound (e.g. -128 for INT8, -32768 for INT16)
        int_max: upper integer clamp bound (e.g.  127 for INT8,  32767 for INT16)
    """
    s = scale.to(device=x.device, dtype=x.dtype)
    x_q = (x / s).clamp(int_min, int_max).round() * s
    return x + (x_q - x).detach()


def _per_channel_weight_scale(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute per-output-channel symmetric scale for a weight tensor.

    Returns a tensor of shape [out_ch, 1, 1, ...] (same ndim as weight,
    extra dims set to 1 for broadcasting).
    """
    out_ch = weight.shape[0]
    w_flat = weight.detach().view(out_ch, -1)
    scale = w_flat.abs().max(dim=1).values.clamp(min=1e-8) / INT8_MAX
    extra_dims = weight.dim() - 1
    return scale.view(out_ch, *([1] * extra_dims))


# ---------------------------------------------------------------------------
# Quantized building blocks
# ---------------------------------------------------------------------------

class QuantDWConv2d(nn.Module):
    """
    Quantized depthwise-separable 2D conv layer.
    Supports W8A8 (act_bits=8) and W8A16 (act_bits=16).
    Structurally identical to DSDWConvLayer; replaces it when w8a8/w8a16=True.

    Quantization
    ------------
    - Input activation: fake-quantized with a frozen per-tensor scale
      (set during calibration). Bit-width controlled by act_bits.
    - Depthwise weights: fake-quantized per-output-channel INT8 dynamically.
    - Pointwise weights: fake-quantized per-output-channel INT8 dynamically.
    - BN + GELU: float32, unchanged.

    States (Python attrs, not buffers — they drive Python-level branching)
    -------
    calibrating:      bool  — collect running max of |activations|
    quantize_enabled: bool  — apply fake-quant in forward()
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        dropout_ratio: float = 0.0,
        act_bits: int = 8,
    ):
        super().__init__()
        kernel_size = tuple(kernel_size)
        stride = tuple(stride)
        padding = "same" if max(stride) == 1 else "valid"

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, padding_mode="zeros",
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 1), stride=1,
            padding=0, bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p=dropout_ratio)

        # Activation quantization range (INT8 or INT16)
        self._act_int_max = 2 ** (act_bits - 1) - 1   # 127 or 32767
        self._act_int_min = -(2 ** (act_bits - 1))     # -128 or -32768
        self.act_bits = act_bits

        # Frozen activation scale (set during calibration, then fixed)
        self.register_buffer("act_scale", torch.tensor(1.0))

        # Mode flags (plain Python bools — no device movement needed)
        self.calibrating: bool = False
        self.quantize_enabled: bool = False
        self._calib_max: float = 0.0

    # -- Mode controls --------------------------------------------------

    def enable_calibration(self) -> None:
        self.calibrating = True
        self.quantize_enabled = False
        self._calib_max = 0.0

    def freeze_calibration(self) -> None:
        """Lock in the activation scale and switch to QAT mode.
        Only enables fake-quant if calibration collected real data.
        Layers with calib_max=0 stay in float32 to avoid destroying features.
        """
        if self._calib_max > 0.0:
            scale_val = max(self._calib_max / self._act_int_max, 1e-8)
            self.act_scale.fill_(scale_val)
            self.quantize_enabled = True
        # else: calibration saw no data for this layer — leave quantize_enabled=False
        self.calibrating = False

    def disable_quantization(self) -> None:
        self.calibrating = False
        self.quantize_enabled = False

    # -- Forward --------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Activation fake-quant / calibration
        if self.calibrating:
            cur_max = x.detach().abs().max().item()
            if cur_max > self._calib_max:
                self._calib_max = cur_max
        elif self.quantize_enabled:
            x = _fake_quant_symmetric(x, self.act_scale, self._act_int_min, self._act_int_max)

        # 2. Depthwise conv (INT8 quantized weights)
        dw_w = self.depthwise.weight
        if self.quantize_enabled:
            dw_scale = _per_channel_weight_scale(dw_w)
            dw_w = _fake_quant_symmetric(dw_w, dw_scale)
        x = F.conv2d(
            x, dw_w, None,
            self.depthwise.stride, self.depthwise.padding,
            self.depthwise.dilation, self.depthwise.groups,
        )

        # 3. Pointwise conv (INT8 quantized weights)
        pw_w = self.pointwise.weight
        if self.quantize_enabled:
            pw_scale = _per_channel_weight_scale(pw_w)
            pw_w = _fake_quant_symmetric(pw_w, pw_scale)
        x = F.conv2d(x, pw_w, self.pointwise.bias, 1, 0)

        return self.drop(self.act(self.bn(x)))

    # -- Introspection --------------------------------------------------

    def get_scales(self) -> dict:
        """Return current scales for TensorBoard logging."""
        dw_scale = _per_channel_weight_scale(self.depthwise.weight)
        pw_scale = _per_channel_weight_scale(self.pointwise.weight)
        return {
            "act_scale": self.act_scale.item(),
            "dw_weight_scale_mean": dw_scale.mean().item(),
            "dw_weight_scale_max": dw_scale.max().item(),
            "pw_weight_scale_mean": pw_scale.mean().item(),
            "pw_weight_scale_max": pw_scale.max().item(),
        }


class QuantDWConv1d(nn.Module):
    """
    Quantized depthwise-separable 1D conv layer.
    Supports W8A8 (act_bits=8) and W8A16 (act_bits=16).
    Structurally identical to DSTemporalDWLayer; replaces it when w8a8/w8a16=True.

    Quantization
    ------------
    Same scheme as QuantDWConv2d (per-channel INT8 weight scales, per-tensor
    activation scale at act_bits precision).  BN1d + GELU stay float32.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout_ratio: float = 0.0,
        act_bits: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2  # "same" for odd kernels

        self.depthwise = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size, padding=padding,
            groups=channels, bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout_ratio)

        # Activation quantization range (INT8 or INT16)
        self._act_int_max = 2 ** (act_bits - 1) - 1   # 127 or 32767
        self._act_int_min = -(2 ** (act_bits - 1))     # -128 or -32768
        self.act_bits = act_bits

        self.register_buffer("act_scale", torch.tensor(1.0))

        self.calibrating: bool = False
        self.quantize_enabled: bool = False
        self._calib_max: float = 0.0

    def enable_calibration(self) -> None:
        self.calibrating = True
        self.quantize_enabled = False
        self._calib_max = 0.0

    def freeze_calibration(self) -> None:
        if self._calib_max > 0.0:
            scale_val = max(self._calib_max / self._act_int_max, 1e-8)
            self.act_scale.fill_(scale_val)
            self.quantize_enabled = True
        self.calibrating = False

    def disable_quantization(self) -> None:
        self.calibrating = False
        self.quantize_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, I]
        if self.calibrating:
            cur_max = x.detach().abs().max().item()
            if cur_max > self._calib_max:
                self._calib_max = cur_max
        elif self.quantize_enabled:
            x = _fake_quant_symmetric(x, self.act_scale, self._act_int_min, self._act_int_max)

        dw_w = self.depthwise.weight
        if self.quantize_enabled:
            dw_scale = _per_channel_weight_scale(dw_w)
            dw_w = _fake_quant_symmetric(dw_w, dw_scale)
        x = F.conv1d(
            x, dw_w, None,
            self.depthwise.stride, self.depthwise.padding,
            self.depthwise.dilation, self.depthwise.groups,
        )

        pw_w = self.pointwise.weight
        if self.quantize_enabled:
            pw_scale = _per_channel_weight_scale(pw_w)
            pw_w = _fake_quant_symmetric(pw_w, pw_scale)
        x = F.conv1d(x, pw_w, self.pointwise.bias, 1, 0)

        return self.drop(self.act(self.bn(x)))

    def get_scales(self) -> dict:
        dw_scale = _per_channel_weight_scale(self.depthwise.weight)
        pw_scale = _per_channel_weight_scale(self.pointwise.weight)
        return {
            "act_scale": self.act_scale.item(),
            "dw_weight_scale_mean": dw_scale.mean().item(),
            "dw_weight_scale_max": dw_scale.max().item(),
            "pw_weight_scale_mean": pw_scale.mean().item(),
            "pw_weight_scale_max": pw_scale.max().item(),
        }


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _iter_quant_layers(model: nn.Module):
    """Yield (name, layer) for all QuantDWConv2d / QuantDWConv1d in model."""
    for name, module in model.named_modules():
        if isinstance(module, (QuantDWConv2d, QuantDWConv1d)):
            yield name, module


def calibrate_w8a8(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_batches: int = 50,
    augmenter=None,
    apply_augmentation_fn=None,
) -> None:
    """
    Collect activation statistics over `n_batches` from `dataloader`.

    Sets every QuantDWConv2d/QuantDWConv1d layer to calibration mode,
    runs the model in eval (no grad), then calls freeze_calibration() on
    each layer to lock in the activation scales.

    After this call the model is ready for QAT finetuning.

    Args:
        model:                  model containing QuantDW* layers
        dataloader:             DataLoader that yields (inputs, labels)
        device:                 torch device to run calibration on
        n_batches:              number of batches to use (50 is usually sufficient)
        augmenter:              augmenter object — MUST be provided if preprocessing
                                (e.g. mel conversion) is done by the augmenter, otherwise
                                the model receives raw FFT data and the spectrum_proj will
                                have a shape mismatch.
        apply_augmentation_fn:  function with signature (augmenter, inputs, labels) → (inputs, labels)
    """
    import logging
    logger = logging.getLogger("w8a8_calibration")

    if augmenter is None or apply_augmentation_fn is None:
        logger.warning(
            "[W8A8] calibrate_w8a8 called without augmenter/apply_augmentation_fn. "
            "If mel conversion is done by the augmenter (preprocess_mode='mel'), "
            "the model will receive raw FFT data and calibration will fail."
        )

    # Enable calibration mode on all quant layers
    for _, layer in _iter_quant_layers(model):
        layer.enable_calibration()

    model.eval()
    model.to(device)

    batches_seen = 0
    with torch.no_grad():
        for batch in dataloader:
            if batches_seen >= n_batches:
                break
            # Unpack — DataLoader yields (inputs_dict, labels)
            if len(batch) == 2:
                inputs, labels = batch
            elif len(batch) == 3:
                inputs, _, labels = batch
            else:
                inputs = batch[0]
                labels = None

            if isinstance(inputs, dict):
                inputs = {
                    loc: {mod: t.to(device) for mod, t in mods.items()}
                    for loc, mods in inputs.items()
                }
            else:
                inputs = inputs.to(device)

            # Apply augmenter (mel conversion + augmentation).
            # Mel conversion is essential when preprocess_mode='mel'.
            if augmenter is not None and apply_augmentation_fn is not None and labels is not None:
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                labels = labels.to(device)
                inputs, labels = apply_augmentation_fn(augmenter, inputs, labels)

            try:
                model(inputs)
            except Exception as e:
                logger.warning(
                    "[W8A8] Calibration batch %d raised an exception — "
                    "layers after this point may not be calibrated: %s",
                    batches_seen, e,
                )
            batches_seen += 1

    # Freeze scales and enable fake-quant
    n_layers = 0
    n_quantized = 0
    for name, layer in _iter_quant_layers(model):
        layer.freeze_calibration()
        n_layers += 1
        status = "QUANTIZED" if layer.quantize_enabled else "FLOAT32 (no calib data)"
        logger.info(
            "  [W8A8] %-50s  act_scale=%.6f  calib_max=%.4f  [%s]",
            name,
            layer.act_scale.item(),
            layer._calib_max,
            status,
        )
        if layer.quantize_enabled:
            n_quantized += 1

    logger.info(
        "[W8A8] Calibration complete: %d/%d layers quantized, %d batches used",
        n_quantized, n_layers, batches_seen,
    )


def freeze_w8a8(model: nn.Module) -> None:
    """Enable fake-quant on all quant layers (call after calibrate_w8a8)."""
    for _, layer in _iter_quant_layers(model):
        layer.quantize_enabled = True
        layer.calibrating = False


def disable_w8a8(model: nn.Module) -> None:
    """Turn off all fake-quant (useful for float32 baseline eval)."""
    for _, layer in _iter_quant_layers(model):
        layer.disable_quantization()


def has_w8a8_layers(model: nn.Module) -> bool:
    """Return True if the model contains any QuantDW* layers."""
    return any(True for _ in _iter_quant_layers(model))


# ---------------------------------------------------------------------------
# TensorBoard logging
# ---------------------------------------------------------------------------

def export_quantized_checkpoint(
    model: nn.Module,
    best_model_path: str,
    output_path: str,
) -> None:
    """
    Export a deployment-ready checkpoint from a QAT-trained model.

    Loads the best model weights, then saves a new state dict where:
      - QuantDW* layer weights are stored as their true integer dtype
        (torch.int8 for act_bits=8, torch.int16 for act_bits=16)
      - A matching <key>_scale tensor (float32, per output channel) is added
        alongside each weight tensor
      - act_scale buffers are kept as-is (float32 scalar)
      - All other tensors (BN, spectrum_proj, fc, etc.) are kept as float32

    Args:
        model:           the QAT model (used to identify QuantDW* layers)
        best_model_path: path to best_model.pth saved during training
        output_path:     where to write best_model_quantized.pth
    """
    import logging
    import os
    logger = logging.getLogger("w8a8_export")

    state_dict = torch.load(best_model_path, map_location="cpu")

    quant_layer_names = {
        name for name, module in model.named_modules()
        if isinstance(module, (QuantDWConv2d, QuantDWConv1d))
    }

    export_dict = {}
    for key, tensor in state_dict.items():
        # Identify which quant layer this key belongs to, if any
        matched_layer = None
        for layer_name in quant_layer_names:
            if key.startswith(layer_name + "."):
                matched_layer = layer_name
                break

        if matched_layer is None:
            # Not a quantized layer — keep as float32
            export_dict[key] = tensor
            continue

        suffix = key[len(matched_layer) + 1:]  # e.g. "depthwise.weight"

        if suffix in ("depthwise.weight", "pointwise.weight"):
            # Get the act_bits for this layer
            layer_module = dict(model.named_modules())[matched_layer]
            act_bits = layer_module.act_bits
            int_max = layer_module._act_int_max  # 127 or 32767 — but weights always INT8
            weight_int_max = INT8_MAX

            # Compute per-channel scale: max(|w|) / 127
            w = tensor.float()
            out_ch = w.shape[0]
            w_flat = w.view(out_ch, -1)
            weight_scale = w_flat.abs().max(dim=1).values.clamp(min=1e-8) / weight_int_max
            extra_dims = w.dim() - 1
            scale_broadcast = weight_scale.view(out_ch, *([1] * extra_dims))

            # Convert to int8
            w_int = (w / scale_broadcast).round().clamp(INT8_MIN, INT8_MAX).to(torch.int8)

            export_dict[key] = w_int
            export_dict[key + "_scale"] = weight_scale  # float32 [out_channels]

            logger.info(
                "  [export] %-60s  int8  scale=[%.4e, %.4e]  shape=%s",
                key, weight_scale.min().item(), weight_scale.max().item(), list(w_int.shape),
            )

        else:
            # act_scale, bn params, bias — keep as float32
            export_dict[key] = tensor

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(export_dict, output_path)
    logger.info("[export] Quantized checkpoint saved to: %s", output_path)


def log_w8a8_scales(
    model: nn.Module,
    writer,
    epoch: int,
    prefix: str = "W8A8",
) -> None:
    """
    Write per-layer W8A8 scale statistics to TensorBoard.

    Called once per epoch from the training loop.  Logs:
      - Activation scale (per-tensor, frozen)
      - Depthwise weight scale: mean and max over output channels
      - Pointwise weight scale: mean and max over output channels

    Tags:  W8A8/<layer_name>/act_scale
           W8A8/<layer_name>/dw_weight_scale_mean  (and _max)
           W8A8/<layer_name>/pw_weight_scale_mean  (and _max)

    Aggregates across all quant layers:
           W8A8/summary/mean_act_scale
           W8A8/summary/mean_dw_weight_scale
           W8A8/summary/mean_pw_weight_scale
    """
    all_act = []
    all_dw_mean = []
    all_pw_mean = []

    for name, layer in _iter_quant_layers(model):
        if not layer.quantize_enabled:
            continue
        scales = layer.get_scales()
        tag_base = f"{prefix}/{name}"
        writer.add_scalar(f"{tag_base}/act_scale",         scales["act_scale"],          epoch)
        writer.add_scalar(f"{tag_base}/dw_weight_scale_mean", scales["dw_weight_scale_mean"], epoch)
        writer.add_scalar(f"{tag_base}/dw_weight_scale_max",  scales["dw_weight_scale_max"],  epoch)
        writer.add_scalar(f"{tag_base}/pw_weight_scale_mean", scales["pw_weight_scale_mean"], epoch)
        writer.add_scalar(f"{tag_base}/pw_weight_scale_max",  scales["pw_weight_scale_max"],  epoch)

        all_act.append(scales["act_scale"])
        all_dw_mean.append(scales["dw_weight_scale_mean"])
        all_pw_mean.append(scales["pw_weight_scale_mean"])

    if all_act:
        writer.add_scalar(f"{prefix}/summary/mean_act_scale",        sum(all_act) / len(all_act),    epoch)
        writer.add_scalar(f"{prefix}/summary/mean_dw_weight_scale",  sum(all_dw_mean) / len(all_dw_mean), epoch)
        writer.add_scalar(f"{prefix}/summary/mean_pw_weight_scale",  sum(all_pw_mean) / len(all_pw_mean), epoch)
