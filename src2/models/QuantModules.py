import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


"""
Quantization Modules for Neural Networks
=========================================

This module provides quantization support for convolutional layers with the following features:
- Multiple quantization methods: DoReFa, PACT, LSQ, LSQPlus
- Adaptive bitwidth support
- Base class architecture for extensibility

Architecture Overview
--------------------

BaseQuanConv: Abstract base class providing common functionality
├── Constructor: Initializes conv parameters, weights, bias
├── setup_quantize_funcs(): Setup from config
├── _setup_quantizers(): Initialize quantizers (DoReFa/PACT/LSQ/LSQPlus)
├── alpha_setup(): Configure PACT alpha parameters
├── Bitwidth management: set/get/random bitwidth methods
├── Helper methods:
│   ├── _quantize_weight(): Quantize weights
│   ├── _quantize_activation(): Quantize activations
│   └── _apply_convolution(): Apply conv2d operation
└── _custom_init(): Hook for subclass-specific initialization

QuanConv: Standard implementation (original behavior)
└── forward(): Weight quant -> Weight norm -> Activation quant -> Conv

QuanConvActivationFirst: Alternative implementation
└── forward(): Activation quant -> Weight quant -> Weight norm -> Conv

QuanConvNoWeightNorm: Simplified implementation
└── forward(): Weight quant -> Activation quant -> Conv (no normalization)

Creating Custom Implementations
-------------------------------

To create a new quantized convolution variant:

1. Inherit from BaseQuanConv
2. Override forward() to implement your quantization strategy
3. Optionally override _custom_init() for additional setup
4. Use the provided helper methods:
   - self._quantize_weight()
   - self._quantize_activation(inp)
   - self._apply_convolution(inp, weight, bias)

Example:
    class MyCustomQuanConv(BaseQuanConv):
        def _custom_init(self):
            # Custom initialization
            self.my_custom_param = nn.Parameter(torch.ones(1))
        
        def forward(self, inp):
            if self.float_mode or self.curr_bitwidth == 32:
                return self._apply_convolution(inp, self.weight)
            
            # Your custom quantization logic here
            w = self._quantize_weight()
            x = self._quantize_activation(inp)
            # ... custom processing ...
            return self._apply_convolution(x, w)
"""


def quantize(x, nbit, alpha=None):
    """Quantization function used by DoReFa and PACT"""
    class DynamicQuantizer(torch.autograd.Function):
        """Simple quantizer for DoReFa (no learnable clipping)"""
        @staticmethod
        def forward(ctx, x, nbit):
            if nbit == 32:
                return x
            scale = 2**nbit - 1
            return torch.round(scale * x) / scale

        @staticmethod
        def backward(ctx, grad_output):
            # STE: pass gradient through unchanged
            return grad_output.clone(), None
        
    return DynamicQuantizer.apply(x, nbit)

import random

class BitMixerController:
    """
    Exact Bit-Mixer controller from ICCV 2021.

    Responsibilities:
    - Implements Stage I / II / III scheduling
    - Sets per-layer or global bitwidths
    - Enables/disables weight quantization
    - Works with TransitionalBatchNorm automatically via conv.curr_bitwidth
    """

    def __init__(
        self,
        model,
        bitwidth_options,
        stage,
        sigma_start=1.0,
        sigma_end=0.25,
        total_epochs=1,
    ):
        """
        Args:
            model: PyTorch model containing BaseQuanConv layers
            bitwidth_options: e.g. [2, 3, 4, 8]
            stage: 1, 2, or 3
            sigma_start: initial σ for Stage III
            sigma_end: final σ for Stage III
            total_epochs: Stage III length (epochs)
        """
        assert stage in [1, 2, 3]
        self.model = model
        self.bitwidth_options = list(bitwidth_options)
        self.stage = stage

        # Stage III parameters
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.total_epochs = max(total_epochs, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_global_bitwidth(self, bw, quantize_weights):
        for m in self.model.modules():
            if isinstance(m, BaseQuanConv):
                m.quantize_weights_enabled = quantize_weights
                m.set_bitwidth(bw)

    def _set_layerwise_random_bitwidth(self, quantize_weights):
        for m in self.model.modules():
            if isinstance(m, BaseQuanConv):
                m.quantize_weights_enabled = quantize_weights
                bw = random.choice(self.bitwidth_options)
                m.set_bitwidth(bw)

    def _sigma(self, epoch):
        """Linear σ schedule (paper uses decreasing σ)."""
        # Avoid divide-by-zero when total_epochs == 1 (common during quick runs)
        denom = max(self.total_epochs - 1, 1)
        t = min(epoch / denom, 1.0)
        return self.sigma_start + t * (self.sigma_end - self.sigma_start)

    # ------------------------------------------------------------------
    # Public API (CALL THIS EACH ITERATION)
    # ------------------------------------------------------------------

    def step(self, epoch):
        """
        Apply Bit-Mixer logic for ONE training iteration.

        Call this once per batch BEFORE forward().
        """

        # ------------------------------
        # Stage I
        # ------------------------------
        if self.stage == 1:
            bw = random.choice(self.bitwidth_options)
            self._set_global_bitwidth(bw, quantize_weights=False)
            return

        # ------------------------------
        # Stage II
        # ------------------------------
        if self.stage == 2:
            bw = random.choice(self.bitwidth_options)
            self._set_global_bitwidth(bw, quantize_weights=True)
            return

        # ------------------------------
        # Stage III (Bit-Mixer)
        # ------------------------------
        sigma = self._sigma(epoch)

        if random.random() < sigma:
            # Global bitwidth (Stage II behavior)
            bw = random.choice(self.bitwidth_options)
            self._set_global_bitwidth(bw, quantize_weights=True)
        else:
            # Layer-wise random bitwidths (true Bit-Mixer)
            self._set_layerwise_random_bitwidth(quantize_weights=True)



class LSQQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step_size, nbit, is_activation=False):
        qn = 0 if is_activation else -(2 ** (nbit - 1))
        qp = 2 ** (nbit - 1) - 1 if not is_activation else 2 ** nbit - 1
        step_size = step_size.to(x.device)

        ctx.save_for_backward(x, step_size)
        ctx.other = (qn, qp, is_activation)

        x = x / step_size
        x_clipped = torch.clamp(x.round(), qn, qp)
        return x_clipped * step_size

    @staticmethod
    def backward(ctx, grad_output):
        x, step_size = ctx.saved_tensors
        qn, qp, is_activation = ctx.other

        x_div = x / step_size
        x_clipped = torch.clamp(x_div.round(), qn, qp)
        mask = ((x_div >= qn) & (x_div <= qp)).float()

        grad_x = grad_output * mask
        grad_s = ((grad_output * (x_div - x_clipped)).sum()).view_as(step_size)
        grad_s /= ((qp * x.numel()) ** 0.5)

        return grad_x, grad_s, None, None


import math

class LSQ(nn.Module):
    def __init__(self, is_activation=False, bitwidth=8, shape=1, bitwidth_opts=None, signed=None):
        super().__init__()
        self.is_activation = is_activation

        # default signed rule: activations unsigned, weights signed
        if signed is None:
            signed = (not is_activation)
        self.signed = signed

        # multi-precision setup
        if bitwidth_opts is None:
            bitwidth_opts = [bitwidth]
        self.bitwidth_opts = list(bitwidth_opts)
        self.bitwidth = bitwidth if bitwidth in self.bitwidth_opts else self.bitwidth_opts[0]

        # ensure shape is tuple
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)

        # bank of step sizes: [B, *shape]
        self.step_size = nn.Parameter(torch.ones(len(self.bitwidth_opts), *self.shape))
        self.register_buffer("initialized", torch.zeros(len(self.bitwidth_opts), dtype=torch.bool))

    def _idx(self, bw: int) -> int:
        return self.bitwidth_opts.index(int(bw))

    def update_bitwidth(self, bw):
        self.bitwidth = int(bw)

    def _qrange(self, bw: int):
        if self.signed:
            qn = -(2 ** (bw - 1))
            qp =  (2 ** (bw - 1)) - 1
        else:
            qn = 0
            qp = (2 ** bw) - 1
        return qn, qp

    def _init_step(self, x, bw, idx):
        _, qp = self._qrange(bw)
        # LSQ init: alpha0 = 2*E|x| / sqrt(qp)  (common LSQ init)
        alpha0 = (2.0 * x.detach().abs().mean() / math.sqrt(max(qp, 1)))
        with torch.no_grad():
            self.step_size[idx].copy_(alpha0.expand_as(self.step_size[idx]))
        self.initialized[idx] = True

    def forward(self, x):
        bw = int(self.bitwidth)
        idx = self._idx(bw)

        if not bool(self.initialized[idx].item()):
            self._init_step(x, bw, idx)

        qn, qp = self._qrange(bw)

        s = self.step_size[idx].to(device=x.device, dtype=x.dtype)

        # gradient scaling factor
        # N per alpha supports per-channel step sizes too
        N_per_alpha = x.numel() / s.numel()
        g = 1.0 / math.sqrt(max(qp, 1) * N_per_alpha)

        # grad scale trick: forward identity; backward scaled
        s_scaled = (s.detach() - (s.detach() * g).detach() + s * g)

        # broadcast s_scaled to x if needed
        s_b = s_scaled
        while s_b.dim() < x.dim():
            s_b = s_b.view(*s_b.shape, 1)

        x_div = x / s_b
        x_clamped = torch.clamp(x_div, qn, qp)

        # round STE
        x_rounded = (torch.round(x_clamped) - x_clamped).detach() + x_clamped
        return x_rounded * s_b



def grad_scale(x, scale: float):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

class LSQPlus(nn.Module):
    """
    LSQ+ Quantizer (Learned Step Size Quantization Plus)
    
    Paper: "LSQ+: Improving low-bit quantization through learnable offsets and better initialization"
    
    Quantization formula: x_int = clamp(round(x/s - zp), qn, qp)
    Dequantization formula: x_q = (x_int + zp) * s
    
    For weights: symmetric quantization (zp = 0, signed)
    For activations: asymmetric quantization (learned zp, unsigned)
    """
    
    def __init__(self, is_activation=False, bitwidth=8, shape=1,
                 bitwidth_opts=None, signed=None, batch_init: int = 20):
        super().__init__()
        self.is_activation = is_activation
        
        # Signed convention: weights signed, activations unsigned (unless overridden)
        if signed is None:
            signed = not is_activation
        self.signed = signed
        
        # Multi-precision support
        if bitwidth_opts is None:
            bitwidth_opts = [bitwidth]
        self.bitwidth_opts = list(bitwidth_opts)
        self.bitwidth = bitwidth if bitwidth in self.bitwidth_opts else self.bitwidth_opts[0]
        
        # Shape for per-channel or per-tensor quantization
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        
        B = len(self.bitwidth_opts)
        
        # Learnable step size (scale)
        self.step_size = nn.Parameter(torch.ones(B, *self.shape))
        
        # Learnable zero-point: only for activations in asymmetric mode
        # For weights, we use symmetric quantization (zp fixed at 0)
        self.use_zero_point = is_activation and not signed
        if self.use_zero_point:
            self.zp = nn.Parameter(torch.zeros(B, *self.shape))
        else:
            self.register_buffer('zp', torch.zeros(B, *self.shape))
        
        # Initialization tracking
        self.register_buffer("initialized", torch.zeros(B, dtype=torch.bool))
        
        # Batch-wise initialization for activations (EMA over first few batches)
        self.batch_init = int(batch_init) if is_activation else 0
        if self.batch_init > 0:
            self.register_buffer("init_count", torch.zeros(B, dtype=torch.long))
            self.register_buffer("running_min", torch.zeros(B, *self.shape))
            self.register_buffer("running_max", torch.zeros(B, *self.shape))

    def _idx(self, bw: int) -> int:
        return self.bitwidth_opts.index(int(bw))

    def update_bitwidth(self, bw):
        self.bitwidth = int(bw)

    def _qrange(self, bw: int):
        """Get quantization range [qn, qp] based on bitwidth and signedness."""
        if self.signed:
            qn = -(2 ** (bw - 1))
            qp = (2 ** (bw - 1)) - 1
        else:
            qn = 0
            qp = (2 ** bw) - 1
        return qn, qp

    def _init_params(self, x, bw, idx):
        """
        Initialize step_size and zero_point based on input statistics.
        
        For activations (asymmetric): use min/max to set scale and zero-point
        For weights (symmetric): use 2*mean(|x|)/sqrt(qp) like LSQ
        """
        qn, qp = self._qrange(bw)
        
        with torch.no_grad():
            if self.is_activation:
                # Asymmetric initialization for activations
                x_min = x.detach().min()
                x_max = x.detach().max()
                
                # Scale to cover the full range
                scale = ((x_max - x_min) / (qp - qn)).clamp(min=1e-8)
                
                # Zero-point: maps x_min to qn
                # x_int = round(x/s - zp), so zp = x_min/s - qn
                zero_point = (x_min / scale) - qn
                
                self.step_size[idx].fill_(scale.item())
                if self.use_zero_point:
                    self.zp[idx].fill_(zero_point.item())
            else:
                # Symmetric initialization for weights (same as LSQ)
                scale = (2.0 * x.detach().abs().mean() / math.sqrt(qp)).clamp(min=1e-8)
                self.step_size[idx].fill_(scale.item())
                # zp stays at 0 for symmetric quantization
        
        self.initialized[idx] = True

    def _batch_init_update(self, x, bw, idx):
        """
        Update running statistics during batch initialization phase.
        Uses exponential moving average of min/max values.
        """
        qn, qp = self._qrange(bw)
        count = int(self.init_count[idx].item())
        
        with torch.no_grad():
            x_min = x.detach().min()
            x_max = x.detach().max()
            
            if count == 0:
                self.running_min[idx].fill_(x_min.item())
                self.running_max[idx].fill_(x_max.item())
            else:
                # EMA with momentum 0.9
                momentum = 0.9
                self.running_min[idx].mul_(momentum).add_((1 - momentum) * x_min)
                self.running_max[idx].mul_(momentum).add_((1 - momentum) * x_max)
            
            # Update scale and zero-point based on running stats
            r_min = self.running_min[idx]
            r_max = self.running_max[idx]
            
            scale = ((r_max - r_min) / (qp - qn)).clamp(min=1e-8)
            zero_point = (r_min / scale) - qn
            
            self.step_size[idx].copy_(scale)
            if self.use_zero_point:
                self.zp[idx].copy_(zero_point)
        
        self.init_count[idx] += 1

    def forward(self, x):
        bw = int(self.bitwidth)
        idx = self._idx(bw)
        qn, qp = self._qrange(bw)
        
        # Initialize on first forward pass
        if not self.initialized[idx]:
            self._init_params(x, bw, idx)
        
        # Batch initialization phase for activations
        if self.is_activation and self.batch_init > 0:
            if self.init_count[idx] < self.batch_init:
                self._batch_init_update(x, bw, idx)
        
        # Get parameters
        s = self.step_size[idx]
        zp = self.zp[idx]
        
        # Move to correct device/dtype
        s = s.to(device=x.device, dtype=x.dtype)
        zp = zp.to(device=x.device, dtype=x.dtype)
        
        # Gradient scaling (LSQ+ paper recommendation)
        # Scale gradients by 1/sqrt(n_elements * qp) for stability
        n_elements = x.numel() / s.numel()
        grad_scale_factor = 1.0 / math.sqrt(n_elements * max(qp, 1))
        
        # Apply gradient scaling via the detach trick
        s = (s - s * grad_scale_factor).detach() + s * grad_scale_factor
        if self.use_zero_point and self.zp.requires_grad:
            zp = (zp - zp * grad_scale_factor).detach() + zp * grad_scale_factor
        
        # Broadcast to input dimensions
        s_b = s
        zp_b = zp
        while s_b.dim() < x.dim():
            s_b = s_b.unsqueeze(-1)
            zp_b = zp_b.unsqueeze(-1)
        
        # Quantize: x_int = clamp(round(x/s - zp), qn, qp)
        x_scaled = x / s_b - zp_b
        
        # STE for rounding: forward uses round, backward passes gradient through
        x_rounded = (torch.round(x_scaled) - x_scaled).detach() + x_scaled
        
        # Clamp to valid range
        x_int = torch.clamp(x_rounded, qn, qp)
        
        # Dequantize: x_q = (x_int + zp) * s
        x_q = (x_int + zp_b) * s_b
        
        return x_q




class DoReFaW(nn.Module):
    def __init__(self):
        super(DoReFaW, self).__init__()

    def forward(self, inp, nbit_w, *args, **kwargs):
        """forward pass"""
        w = torch.tanh(inp)
        maxv = torch.abs(w).max() + 1e-8  # Add epsilon to prevent division by zero
        w = w / (2 * maxv) + 0.5  # This is the quantization step in DoreFa
        w = 2 * quantize(w, nbit_w) - 1  # This is the dequantization step in DoreFa
        return w


class DoReFaA(nn.Module):
    def __init__(self):
        super(DoReFaA, self).__init__()

    def forward(self, inp, nbit_a, *args, **kwargs):
        """forward pass"""
        if nbit_a == 32:
            return torch.clamp(inp, 0, 1)
        return quantize(torch.clamp(inp, 0, 1), nbit_a)


class PACTFunction(torch.autograd.Function):
    """
    PACT: Parameterized Clipping Activation for Quantized Neural Networks
    Reference: https://arxiv.org/abs/1805.06085
    
    Forward:
        y = clamp(x, 0, α)
        y_q = round(y * (2^k - 1) / α) * α / (2^k - 1)
    
    Backward (using STE):
        ∂L/∂x = ∂L/∂y  if 0 ≤ x ≤ α, else 0
        ∂L/∂α = Σ(∂L/∂y * 1_{x ≥ α})
    """
    @staticmethod
    def forward(ctx, x, alpha, nbit):
        # Save ORIGINAL x before clamping - needed for correct alpha gradient
        ctx.save_for_backward(x, alpha)
        ctx.nbit = nbit
        
        # Clamp to [0, alpha]
        y = torch.clamp(x, min=0.0, max=alpha.item())
        
        # Quantize: scale to [0, 2^k - 1], round, scale back
        scale = (2 ** nbit - 1) / alpha
        y_q = torch.round(y * scale) / scale
        
        return y_q

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve ORIGINAL x (before clamping) and alpha
        x, alpha = ctx.saved_tensors
        
        # Gradient for x: pass through only where 0 <= x <= alpha (STE)
        x_lower = (x >= 0).float()
        x_upper = (x <= alpha).float()
        grad_x = grad_output * x_lower * x_upper
        
        # Gradient for alpha: accumulate where x >= alpha (clipped from above)
        # This is the key fix - we check ORIGINAL x, not clamped x
        grad_alpha = (grad_output * (x >= alpha).float()).sum().view(-1)
        
        return grad_x, grad_alpha, None


class PACT(nn.Module):
    """
    PACT activation quantization module.
    Replaces ReLU with parameterized clipping: y = clamp(x, 0, α)
    then quantizes to k bits.
    """
    def __init__(self):
        super(PACT, self).__init__()

    def forward(self, inp, nbit_a, alpha, *args, **kwargs):
        """
        Forward pass
        
        Args:
            inp: Input tensor
            nbit_a: Number of bits for activation quantization
            alpha: Learnable clipping parameter (nn.Parameter)
        
        Returns:
            Quantized activation tensor
        """
        if nbit_a == 32:
            # Full precision mode - just clamp, no quantization
            return torch.clamp(inp, min=0, max=alpha.item())
        
        return PACTFunction.apply(inp, alpha, nbit_a)


class BaseQuanConv(nn.Module):
    """
    Base class for quantized convolutional layers.
    
    This class provides common functionality for quantized convolutions:
    - Convolution parameter setup (channels, kernel size, stride, etc.)
    - Weight and bias initialization
    - Quantizer setup (DoReFa, PACT, LSQ, LSQPlus)
    - Alpha parameter management for PACT
    - Bitwidth management (adaptive quantization)
    - Float mode support
    - Config-based setup
    
    Subclasses should override the forward() method to implement
    different quantization strategies.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(BaseQuanConv, self).__init__()

        
        # Basic conv parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Weight and bias
        # For grouped convolutions, weight shape is (out_channels, in_channels // groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        nn.init.kaiming_uniform_(
            self.weight, mode="fan_out", nonlinearity="relu"
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # Current bitwidth (adaptive quantization)
        self.curr_bitwidth = None  # Default to full precision
        
        # Initialize quantizer placeholders (will be set by setup_quantize_funcs)
        self.quantize_w = None
        self.quantize_a = None
        self.quantization_enabled = False
        # Used by BitMixer Stage I to disable weight quantization while keeping
        # activation quantization / bitwidth setting active.
        self.quantize_weights_enabled = True
        self.alpha_setup_flag = False
        self.weight_quantization = None
        self.activation_quantization = None
        self.bitwidth_opts = None
        self.switchable_clipping = None
        self.sat_weight_normalization = None
        
        # Setup quantization functions if parameters provided directly
        self.quantization_config = None
        # Call custom initialization hook for subclasses
        self._custom_init()
    
    def _custom_init(self):
        """
        Custom initialization hook for subclasses.
        Override this method to add custom initialization logic.
        """
        pass
    
    def setup_quantize_funcs(self, quantization_method_config):
        """
        Setup quantization functions from quantization configuration dictionary
        
        Args:
            quantization_config: Quantization configuration dictionary
        """
       #self.quantization_config = quantization_config
        
        
        self.weight_quantization =  quantization_method_config["weight_quantization"]
        self.activation_quantization = quantization_method_config["activation_quantization"]
        self.bitwidth_opts =  quantization_method_config["bitwidth_options"]
        self.switchable_clipping =  quantization_method_config["switchable_clipping"]
        self.sat_weight_normalization =  quantization_method_config["sat_weight_normalization"]
        
        # Store for _setup_quantizers
        
        # Setup quantizers
        self._setup_quantizers()
        
        # Alpha setup for PACT
        if self.activation_quantization == "pact":
            self.alpha_setup_flag = True
            if self.switchable_clipping:
                # Alpha shape: [len(bitwidth_opts), 1, 1, 1]
                self.alpha = nn.Parameter(torch.ones(len(self.bitwidth_opts), 1, 1, 1), requires_grad=True)
            else:
                # Single alpha with shape [1, 1, 1, 1]
                self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
            nn.init.constant_(self.alpha, 10.0)
        elif self.activation_quantization in ["lsq", "lsqplus"]:
            self.alpha_setup_flag = False  # LSQ doesn't use alpha the same way
        
        # Enable quantization after setup is complete
        self.quantization_enabled = True
    
    def _setup_quantizers(self):
        """Setup weight and activation quantizers"""
        # Weight quantization
        if self.weight_quantization == "dorefa":
            self.quantize_w = DoReFaW()
        elif self.weight_quantization == "lsq":
            # weights are signed, often better per-channel step: (out_ch,1,1,1)
            self.quantize_w = LSQ(
                is_activation=False,
                bitwidth=self.bitwidth_opts[0],
                bitwidth_opts=self.bitwidth_opts,
                shape=(self.out_channels, 1, 1, 1),
                signed=True
            )
        elif self.weight_quantization == "lsqplus":
            self.quantize_w = LSQPlus(
                is_activation=False,
                bitwidth=self.bitwidth_opts[0],
                bitwidth_opts=self.bitwidth_opts,
                shape=(self.out_channels, 1, 1, 1),
                signed=True
            )
        else:
            raise NotImplementedError(
                f"Weight Quantization '{self.weight_quantization}' not supported. "
                "Use 'dorefa', 'lsq', or 'lsqplus'"
            )
        
        # Activation quantization
        if self.activation_quantization == "dorefa":
            self.quantize_a = DoReFaA()
        elif self.activation_quantization == "pact":
            self.quantize_a = PACT()
        elif self.activation_quantization == "lsq":
            # activations are unsigned (post-ReLU), usually per-tensor step: shape=1
            self.quantize_a = LSQ(
                is_activation=True,
                bitwidth=self.bitwidth_opts[0],
                bitwidth_opts=self.bitwidth_opts,
                shape=1,
                signed=False
            )
        elif self.activation_quantization == "lsqplus":
            self.quantize_a = LSQPlus(
                is_activation=True,
                bitwidth=self.bitwidth_opts[0],
                bitwidth_opts=self.bitwidth_opts,
                shape=1,
                signed=False
            )

        else:
            raise NotImplementedError(
                f"Activation Quantization '{self.activation_quantization}' not supported. "
                "Use 'dorefa', 'pact', 'lsq', or 'lsqplus'"
            )
    
    def get_alpha(self, bitwidth=None):
        """
        Get alpha value for PACT quantization
        
        Args:
            bitwidth: Specific bitwidth to get alpha for. If None, uses current bitwidth.
        
        Returns:
            Alpha parameter value
        """
        assert self.alpha_setup_flag, "Alpha not setup. Call alpha_setup() first."
        bitwidth = self.curr_bitwidth if bitwidth is None else bitwidth
        if self.switchable_clipping:
            if bitwidth in self.bitwidth_opts:
                idx = self.bitwidth_opts.index(bitwidth)
                return self.alpha[idx]
            else:
                # If bitwidth not in options, return first alpha
                return self.alpha[0]
        else:
            return self.alpha[0]
    
    def set_bitwidth(self, bitwidth):
        """
        Set the current bitwidth for adaptive quantization
        
        Args:
            bitwidth: Target bitwidth (must be in bitwidth_opts or 32 for full precision)
        """
        assert bitwidth <= 32 and bitwidth > 1, "bitwidth should be between 2 and 32"
        if bitwidth != 32 and self.bitwidth_opts is not None:
            assert bitwidth in self.bitwidth_opts, (
                f"bitwidth {bitwidth} not in bitwidth_options {self.bitwidth_opts}"
            )
        self.curr_bitwidth = bitwidth
        
        # Update LSQ/LSQPlus bitwidth dynamically (only for quantized modes)
        if bitwidth != 32 and isinstance(self.quantize_w, (LSQ, LSQPlus)):
            self.quantize_w.update_bitwidth(bitwidth)
        if bitwidth != 32 and isinstance(self.quantize_a, (LSQ, LSQPlus)):
            self.quantize_a.update_bitwidth(bitwidth)
    
    def get_bitwidth(self):
        """Get the current bitwidth"""
        return self.curr_bitwidth
    
    def set_random_bitwidth(self):
        """Randomly select a bitwidth from available options"""
        assert self.bitwidth_opts is not None, "bitwidth options not set"
        bw = self.bitwidth_opts[torch.randint(0, len(self.bitwidth_opts), (1,)).item()]
        self.set_bitwidth(bw)
    
    def set_highest_bitwidth(self):
        """Set bitwidth to the maximum available option"""
        assert self.bitwidth_opts is not None, "bitwidth options not set"
        self.curr_bitwidth = max(self.bitwidth_opts)
    
    def set_lowest_bitwidth(self):
        """Set bitwidth to the minimum available option"""
        assert self.bitwidth_opts is not None, "bitwidth options not set"
        self.curr_bitwidth = min(self.bitwidth_opts)
    
    def forward(self, inp):
        """
        Forward pass - to be implemented by subclasses.
        
        This is the main method that subclasses should override to implement
        different quantization strategies.
        
        Args:
            inp: Input tensor of shape [batch_size, in_channels, height, width]
        
        Returns:
            Output tensor of shape [batch_size, out_channels, out_height, out_width]
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def _quantize_weight(self):
        """
        Helper method to quantize weights according to current settings.
        
        Returns:
            Quantized weight tensor
        """
        if not getattr(self, "quantize_weights_enabled", True):
            return self.weight
        if isinstance(self.quantize_w, (LSQ, LSQPlus)):
            return self.quantize_w(self.weight)
        else:
            return self.quantize_w(self.weight, self.curr_bitwidth)
    
    def _quantize_activation(self, inp):
        """
        Helper method to quantize activations according to current settings.
        
        Args:
            inp: Input activation tensor
            
        Returns:
            Quantized activation tensor
        """
        if isinstance(self.quantize_a, PACT):
            alpha = self.get_alpha()
            return self.quantize_a(inp, self.curr_bitwidth, alpha)
        elif isinstance(self.quantize_a, DoReFaA):
            return self.quantize_a(inp, self.curr_bitwidth)
        elif isinstance(self.quantize_a, (LSQ, LSQPlus)):
            return self.quantize_a(inp)
        else:
            raise ValueError(
                f"Unsupported activation quantization type: {type(self.quantize_a)}"
            )
    
    def _apply_convolution(self, inp, weight, bias=None):
        """
        Helper method to apply the convolution operation.
        
        Args:
            inp: Input tensor
            weight: Weight tensor
            bias: Optional bias tensor (defaults to self.bias if None)
            
        Returns:
            Output tensor after convolution
        """
        if bias is None:
            bias = self.bias
        return nn.functional.conv2d(
            inp,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuanConv(BaseQuanConv):
    """
    Standard quantized convolution implementation.
    """
    
    def forward(self, inp):
        """
        Forward pass with adaptive quantization
        """
        if not self.quantization_enabled:
            return self._apply_convolution(inp, self.weight)
        
        # Validate setup
        if isinstance(self.quantize_a, PACT):
            assert self.alpha_setup_flag, "Alpha not setup for PACT quantization"
        assert self.curr_bitwidth is not None, "bitwidth is None"
        assert self.quantize_w is not None, "quantize_w is None"
        assert self.quantize_a is not None, "quantize_a is None"
        
        # Quantize weights
        w = self._quantize_weight()
        
        # Weight normalization (optional)
        if self.sat_weight_normalization:
            std = w.detach().std()
            if std > 0:
                w = w / std
        
        # Quantize activations
        x = self._quantize_activation(inp)
        
        # Convolution
        output = self._apply_convolution(x, w)
        
        return output


class QuanConvImportance(BaseQuanConv):
    """
    Quantized convolution with learnable per-layer bit-width importance vectors.
    
    Uses Gumbel-Softmax to sample bit-widths from learned importance distributions,
    enabling automatic discovery of optimal per-layer quantization precision.
    
    Features:
    - Learnable importance vector for bit-width selection
    - Gumbel-Softmax sampling with temperature annealing
    - Soft mode (training): Differentiable weighted combination of all bitwidth outputs
    - Hard mode (inference): Argmax selection of single best bitwidth
    - Unique layer identification for tracking and visualization
    
    Training vs Inference:
    - During training, use 'soft' strategy: outputs from all bitwidths are weighted
      by importance probabilities, enabling gradient flow to importance vectors
    - During inference, use 'hard' strategy: argmax selects single best bitwidth
    """
    
    # Class variable for unique layer identification
    _layer_counter = 0
    
    def _custom_init(self):
        """
        Custom initialization for importance vector quantization.
        
        Sets up:
        - Unique layer ID and name
        - Learnable importance vector (initialized uniformly)
        """
        # Assign unique layer ID
        QuanConvImportance._layer_counter += 1
        self.layer_id = QuanConvImportance._layer_counter
        self.layer_name = f"conv_{self.layer_id}"
        
        # Importance vector will be initialized after setup_quantize_funcs is called
        # because we need to know the number of bitwidth options
        self.importance_vector = None
        self.sampling_strategy = None  # Default to soft for training (differentiable)
        self.current_temperature = 1.0  # Default temperature (will be updated during training)
    
    def setup_quantize_funcs(self, quantization_config):
        """
        Override to initialize importance vector after bitwidth options are known.
        Also extracts sampling strategy from config.
        """
        super().setup_quantize_funcs(quantization_config)
        
        # Extract sampling strategy from config (default to 'soft' for training)
        iv_config = quantization_config.get('importance_vector', {})
        self.sampling_strategy = iv_config.get('sampling_strategy', 'soft')
        
        # Initialize importance vector uniformly: [1/N, 1/N, ..., 1/N]
        num_bitwidths = len(self.bitwidth_opts)
        # Use logits (unnormalized), which will be passed through softmax
        # Initialize to zeros so softmax gives uniform distribution initially
        self.importance_vector = nn.Parameter(
            torch.zeros(num_bitwidths),
            requires_grad=True
        )

        # Register as buffer so it moves with model.to(device) but doesn't require gradients
        self.register_buffer(
            'bitwidth_opts_tensor',
            torch.tensor(self.bitwidth_opts, dtype=torch.float32)
        )
        

    
    
    def set_temperature(self, temperature):
        """
        Set the current temperature for Gumbel-Softmax sampling.
        
        This is typically called by the TemperatureScheduler at the end of each epoch.
        See models/Temperature_Scheduler.py for scheduler implementations.
        
        Args:
            temperature: Temperature value (higher = more exploration, lower = more exploitation)
        """
        self.current_temperature = temperature

    def get_temperature(self):
        """
        Get the current temperature for Gumbel-Softmax sampling.
        """
        return self.current_temperature
    
    def get_best_bitwidth(self):
        """
        Get the bitwidth with highest importance (argmax).
        Used during inference/hard mode.
        
        Returns:
            int: Best bitwidth according to learned importance
        """
        assert self.importance_vector is not None, "Importance vector not initialized"
        best_idx = torch.argmax(self.importance_vector).item()
        return self.bitwidth_opts[best_idx]
    
    def _forward_with_bitwidth(self, inp, bitwidth):
        """
        Compute forward pass with a specific bitwidth.
        
        Args:
            inp: Input tensor [batch_size, in_channels, height, width]
            bitwidth: Specific bitwidth to use for quantization
        
        Returns:
            Output tensor [batch_size, out_channels, out_height, out_width]
        """
        # Set bitwidth for quantization functions
        self.set_bitwidth(bitwidth)
        
        # Quantize weights
        w = self._quantize_weight()
        
        # Weight normalization (optional)
        if self.sat_weight_normalization:
            std = w.detach().std()
            if std > 0:
                w = w / std
        
        # Quantize activations
        x = self._quantize_activation(inp)
        
        # Convolution
        output = self._apply_convolution(x, w)
        
        return output
    
    def get_importance_distribution(self):
        """
        Get the current importance distribution (softmax of importance vector).
        
        Returns:
            dict: {'layer_name': str, 'distribution': tensor, 'bitwidth_options': list}
        """
        if self.importance_vector is None:
            return None
        
        distribution = torch.softmax(self.importance_vector, dim=0)
        return {
            'layer_name': self.layer_name,
            'distribution': distribution.detach().cpu(),
            'bitwidth_options': self.bitwidth_opts
        }
    
    @classmethod
    def set_all_layers_mode(cls, model, mode):
        """
        Set the sampling strategy for all QuanConvImportance layers in a model.
        
        Args:
            model: PyTorch model containing QuanConvImportance layers
            mode: Sampling strategy:
                - 'best_bitwidth': Use argmax of importance vector (inference/validation)
                - 'soft_gumbel_softmax': Sample from importance distribution (training)
                - 'uniform_sampling': Sample uniformly at random (validation baseline)
        """
        valid_modes = ['best_bitwidth', 'soft_gumbel_softmax', 'uniform_sampling']
        assert mode in valid_modes, f"Mode must be one of {valid_modes}, got {mode}"
        
        for module in model.modules():
            if isinstance(module, cls):
                module.sampling_strategy = mode
    
    @classmethod
    def get_all_best_bitwidths(cls, model):
        """
        Get the best bitwidth (argmax of importance) for all layers.
        
        Useful for logging and visualization during inference.
        
        Args:
            model: PyTorch model containing QuanConvImportance layers
        
        Returns:
            dict: {layer_name: best_bitwidth}
        """
        best_bitwidths = {}
        for module in model.modules():
            if isinstance(module, cls):
                best_bitwidths[module.layer_name] = module.get_best_bitwidth()
        return best_bitwidths
    
    @classmethod
    def get_all_importance_distributions(cls, model):
        """
        Get importance distributions for all QuanConvImportance layers in a model.
        
        This is a class method that collects softmax distributions from all instances
        of QuanConvImportance in the given model, useful for visualization and analysis.
        
        Args:
            model: PyTorch model containing QuanConvImportance layers
        
        Returns:
            dict: {layer_name: {'distribution': tensor, 'bitwidth_options': list}}
        """
        distributions = {}
        for module in model.modules():
            if isinstance(module, cls):
                dist_info = module.get_importance_distribution()
                if dist_info is not None:
                    distributions[dist_info['layer_name']] = {
                        'distribution': dist_info['distribution'],
                        'bitwidth_options': dist_info['bitwidth_options']
                    }
        
        return distributions
    
    @classmethod
    def collect_all_importance_vectors(cls, model):
        """
        Collect all importance vectors from QuanConvImportance layers in a model.
        
        Args:
            model: PyTorch model containing QuanConvImportance layers
        
        Returns:
            dict: {layer_name: importance_vector_tensor}
        """
        importance_vectors = {}
        for module in model.modules():
            if isinstance(module, cls):
                if hasattr(module, 'importance_vector') and module.importance_vector is not None:
                    importance_vectors[module.layer_name] = module.importance_vector.detach().cpu()
        
        return importance_vectors
    
    
    @staticmethod
    def plot_importance_distributions(importance_dists, bitwidth_options, epoch):
        """
        Create a bar plot showing learned importance distributions for each layer.
        
        Args:
            importance_dists: Dict of {layer_name: {'distribution': tensor, 'bitwidth_options': list}}
            bitwidth_options: List of bitwidth options (for reference)
            epoch: Current epoch number (for title)
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_layers = len(importance_dists)
        if num_layers == 0:
            # Create empty figure if no importance vectors
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No importance vectors found', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'Importance Distributions (Epoch {epoch + 1})')
            return fig
        
        # Create subplots: arrange in grid
        cols = min(3, num_layers)
        rows = (num_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        
        # Flatten axes for easier indexing
        if num_layers == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each layer's distribution
        for idx, (layer_name, dist_info) in enumerate(sorted(importance_dists.items())):
            ax = axes[idx]
            
            distribution = dist_info['distribution'].numpy()
            bw_opts = dist_info['bitwidth_options']
            
            # Create bar plot
            x_pos = np.arange(len(bw_opts))
            bars = ax.bar(x_pos, distribution, alpha=0.7, color='steelblue', edgecolor='navy')
            
            # Highlight the maximum probability
            max_idx = np.argmax(distribution)
            bars[max_idx].set_color('orange')
            bars[max_idx].set_alpha(0.9)
            
            # Labels and formatting
            ax.set_xlabel('Bitwidth', fontsize=10, fontweight='bold')
            ax.set_ylabel('Probability', fontsize=10, fontweight='bold')
            ax.set_title(f'{layer_name}', fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{bw}' for bw in bw_opts])
            ax.set_ylim([0, 1.0])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add text annotation for max probability
            ax.text(max_idx, distribution[max_idx] + 0.05, 
                   f'{distribution[max_idx]:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'Learned Importance Distributions (Epoch {epoch + 1})', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        return fig

    
    def sample_bitwidth_uniform(self):
        """
        Sample a bitwidth uniformly at random from bitwidth options.
        """
        idx = torch.randint(0, len(self.bitwidth_opts), (1,)).item()
        return self.bitwidth_opts[idx]
    
    def forward(self, inp):
        """
        Forward pass with importance-based bitwidth selection.
        
        Two modes:
        - 'hard' (inference): Use argmax to select single best bitwidth.
        
        Args:
            inp: Input tensor [batch_size, in_channels, height, width]
        
        Returns:
            Output tensor [batch_size, out_channels, out_height, out_width]
        """
        if not self.quantization_enabled:
            return self._apply_convolution(inp, self.weight)
        
        # Validate setup
        if isinstance(self.quantize_a, PACT):
            assert self.alpha_setup_flag, "Alpha not setup for PACT quantization"
        assert self.importance_vector is not None, "Importance vector not initialized"
        assert self.quantize_w is not None, "quantize_w is None"
        assert self.quantize_a is not None, "quantize_a is None"
        
        # Validate mode vs model.training state
        assert self.sampling_strategy is not None, "sampling_strategy is None"
        if self.sampling_strategy == 'best_bitwidth':
            # ================================================================
            # BEST BITWIDTH MODE: Use argmax to select single best bitwidth
            # ================================================================
            best_bitwidth = self.get_best_bitwidth()
            output = self._forward_with_bitwidth(inp, best_bitwidth)
        elif self.sampling_strategy == 'soft_gumbel_softmax':
            # ================================================================
            # HARD GUMBEL SOFTMAX MODE: Sample from importance vector
            # Temperature is managed by TemperatureScheduler.step()
            # ================================================================
            soft_Weights = F.gumbel_softmax(self.importance_vector,
             tau=self.current_temperature, hard=False)

            outputs = []
            for bitwidth in self.bitwidth_opts:
                output = self._forward_with_bitwidth(inp, bitwidth)
                outputs.append(output)

            stacked_outputs = torch.stack(outputs, dim=0)
            probs_expanded = soft_Weights.view(-1, 1, 1, 1,1)
            output = (stacked_outputs * probs_expanded).sum(dim=0)

        elif self.sampling_strategy == 'uniform_sampling':
            # ================================================================
            # UNIFORM SAMPLING MODE (Validation): Sample uniformly at random
            # ================================================================
            selected_bitwidth = self.sample_bitwidth_uniform()
            output = self._forward_with_bitwidth(inp, selected_bitwidth)
        else:
            raise ValueError(f"Invalid sampling strategy: {self.sampling_strategy}, \
             must be one of: best_bitwidth, hard_gumbel_softmax, uniform_sampling")
        return output

class SwitchableBatchNorm(nn.Module):
    """
    Switchable BatchNorm for AdaBits/Any-Precision style training.
    
    Maintains separate BatchNorm statistics for each bitwidth.
    The active BN is selected based on the input conv layer's current bitwidth.
    """
    layer_registry = {}
    layer_counter = 0
    
    def __init__(self, out_channels):
        super(SwitchableBatchNorm, self).__init__()
        self.out_channels = out_channels
        self.bitwidth_options = None
        self.bns = None
        self.bn_float = nn.BatchNorm2d(out_channels)  # Fallback for 32-bit
        
        # Layer identification
        self.layer_name = f"switch_bn_{SwitchableBatchNorm.layer_counter}"
        SwitchableBatchNorm.layer_counter += 1
        SwitchableBatchNorm.layer_registry[self.layer_name] = self
        
        # Associated conv layer
        self.input_conv = None
        
        # Config storage
        self.quantization_config = None
    
    def set_input_conv(self, input_conv):
        """Set the corresponding input conv layer for bitwidth lookup."""
        self.input_conv = input_conv
    
    def setup_quantize_funcs(self, quant_config):
        """
        Initialize switchable BN layers for each bitwidth option.
        
        Args:
            quant_config: Quantization configuration dictionary with 'bitwidth_options'
        """
        self.quantization_config = quant_config
        self.bitwidth_options = list(quant_config.get('bitwidth_options', [8]))
        
        # Ensure 32-bit (float) is always available
        if 32 not in self.bitwidth_options:
            self.bitwidth_options.append(32)
        
        # Create separate BN for each bitwidth
        self.bns = nn.ModuleDict({
            str(bw): nn.BatchNorm2d(self.out_channels) 
            for bw in self.bitwidth_options
        })
    
    def get_current_bitwidth(self):
        """Get the current bitwidth from the associated input conv."""
        if self.input_conv is not None and hasattr(self.input_conv, 'curr_bitwidth'):
            return self.input_conv.curr_bitwidth or 32
        return 32
    
    def forward(self, x):
        if self.bns is None:
            # Fallback to float BN if not configured
            return self.bn_float(x)
        
        bitwidth = self.get_current_bitwidth()
        
        # Use the BN corresponding to current bitwidth
        if str(bitwidth) in self.bns:
            return self.bns[str(bitwidth)](x)
        else:
            # Fallback to float BN for unknown bitwidths
            return self.bn_float(x)


class TransitionalBatchNorm(nn.Module):
    """
    Transitional BatchNorm for BitMixer-style training.
    
    Maintains separate BatchNorm statistics for each (input_bitwidth, output_bitwidth) pair.
    This captures the transition between different precisions at layer boundaries.
    """
    layer_registry = {}
    layer_counter = 0
    
    def __init__(self, out_channels):
        super(TransitionalBatchNorm, self).__init__()
        self.out_channels = out_channels
        self.bitwidth_options = None
        self.bns = None
        self.bn_float = nn.BatchNorm2d(out_channels)  # Fallback
        
        # Layer identification
        self.layer_name = f"trans_bn_{TransitionalBatchNorm.layer_counter}"
        TransitionalBatchNorm.layer_counter += 1
        TransitionalBatchNorm.layer_registry[self.layer_name] = self
        
        # Associated conv layers (input and output)
        self.input_conv = None
        self.output_conv = None
        
        # Config storage
        self.quantization_config = None
    
    def set_input_conv(self, input_conv):
        """Set the corresponding input conv layer."""
        self.input_conv = input_conv
    
    def set_output_conv(self, output_conv):
        """Set the corresponding output conv layer."""
        self.output_conv = output_conv
    
    def set_convs(self, input_conv, output_conv):
        """Set both input and output conv layers."""
        self.input_conv = input_conv
        self.output_conv = output_conv
    
    def setup_quantize_funcs(self, quant_config):
        """
        Initialize transitional BN layers for each (input_bw, output_bw) pair.
        
        Args:
            quant_config: Quantization configuration dictionary with 'bitwidth_options'
        """
        self.quantization_config = quant_config
        self.bitwidth_options = list(quant_config.get('bitwidth_options', [8]))
        
        # Ensure 32-bit (float) is always available
        if 32 not in self.bitwidth_options:
            self.bitwidth_options.append(32)
        
        # Create BN for each (input_bitwidth, output_bitwidth) pair
        self.bns = nn.ModuleDict()
        for bw_in in self.bitwidth_options:
            for bw_out in self.bitwidth_options:
                key = f"{bw_in}_{bw_out}"
                self.bns[key] = nn.BatchNorm2d(self.out_channels)
    
    def get_current_bitwidths(self):
        """Get current bitwidths from associated input and output convs."""
        input_bw = 32
        output_bw = 32
        
        if self.input_conv is not None and hasattr(self.input_conv, 'curr_bitwidth'):
            input_bw = self.input_conv.curr_bitwidth or 32
        
        if self.output_conv is not None and hasattr(self.output_conv, 'curr_bitwidth'):
            output_bw = self.output_conv.curr_bitwidth or 32
        
        return input_bw, output_bw
    
    def forward(self, x):
        if self.bns is None:
            # Fallback to float BN if not configured
            return self.bn_float(x)
        
        input_bw, output_bw = self.get_current_bitwidths()
        key = f"{input_bw}_{output_bw}"
        
        # Use the BN corresponding to current transition
        if key in self.bns:
            return self.bns[key](x)
        else:
            # Fallback to float BN for unknown transitions
            return self.bn_float(x)