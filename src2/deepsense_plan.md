# SingleModalDeepSense Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SingleModalDeepSense` to the training pipeline following the same pattern as `SingleModalResNet`: backbone takes a raw tensor, returns `{'logits', 'features'}`. `SingleModalDeepSense` is a thin wrapper that unpacks the dict input.

**Architecture:** `DeepSenseLatest.py` defines three classes:
1. `DSConvLayer` — single `Conv2d + BN + GELU + Dropout2d` building block.
2. `DeepSenseBackbone` — stacks N `DSConvLayer`s in a `nn.ModuleList` (one per entry in `channels`/`kernel_sizes`/`strides`), pools the spectrum dim, runs a bidirectional GRU, then a linear head.
3. `SingleModalDeepSense` — thin wrapper (identical pattern to `SingleModalResNet`).

`create_deepsense()` is added to `create_models.py` and dispatched from `create_single_modal_model()` on `model_type: "deepsense"`.

**Tech Stack:** PyTorch, `RecurrentBlock` (`models/RecurrentModule.py`). No `ConvBlock`/`ConvModules` dependency.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src2/models/DeepSenseLatest.py` | **Create** | `DSConvLayer` + `DeepSenseBackbone` + `SingleModalDeepSense` |
| `src2/models/create_models.py` | **Modify** | Add `create_deepsense()` + dispatch in `create_single_modal_model()` |
| `src2/data/Parkland.yaml` | **Modify** | Add DeepSense model configs to `models:` zoo + experiment entries |
| `src2/models/test_deepsense_creation.py` | **Create** | Model creation + single-batch inference test |

---

## Key Design Decisions

**Three parallel lists** — `channels`, `kernel_sizes`, `strides` each have one entry per conv layer. `len(channels)` determines the number of layers. All three must be the same length (ValueError otherwise).

**Channel progression** — `channels[i]` is the output channels for layer `i`. Input to layer 0 is `in_channels` (from dataset config). Input to layer `i > 0` is `channels[i-1]`. The GRU input dim is `channels[-1]`.

**Example:** `channels: [64, 128, 32]`, `in_channels=2`:
- Layer 0: in=2, out=64
- Layer 1: in=64, out=128
- Layer 2: in=128, out=32
- GRU input: 32

**Padding** — `"same"` when `max(stride) == 1`, `"valid"` otherwise. Follows the convention in `ConvLayer2D`.

**Spectrum collapse** — `AdaptiveAvgPool2d((None, 1))` after the conv stack, giving `[B, channels[-1], intervals']`.

**Configurable GRU** — `recurrent_layers` is explicit in YAML (was hardcoded to 2 in old `DeepSense.py`).

**Output** — `{'logits': [B, num_classes], 'features': [B, fc_dim]}`.

---

## YAML config shape

```yaml
student_audio_deepsense:
    model_source: "create_single_modal_model"
    model_type: "deepsense"
    active_modality: "audio"
    # Conv stack — all three lists must be the same length
    channels: [64, 128, 128]            # output channels per layer; input auto-determined
    kernel_sizes: [[1, 80], [1, 5], [1, 5]]
    strides: [[1, 80], [1, 1], [1, 1]]
    # GRU
    recurrent_dim: 256
    recurrent_layers: 2
    # Head
    fc_dim: 512
    dropout_ratio: 0.2
```

---

## Task 1: Create `DeepSenseLatest.py`

**Files:**
- Create: `src2/models/DeepSenseLatest.py`

- [ ] **Step 1: Write the file**

```python
# src2/models/DeepSenseLatest.py
"""
DeepSense backbone for single-modality classification.

Classes
-------
DSConvLayer          : single Conv2d + BN + GELU + Dropout2d building block
DeepSenseBackbone    : configurable conv stack → pool → GRU → linear head
SingleModalDeepSense : thin wrapper unpacking freq_x[location][modality]
"""
import torch
import torch.nn as nn

from models.RecurrentModule import RecurrentBlock


class DSConvLayer(nn.Module):
    """
    Single 2-D conv layer: Conv2d → BatchNorm2d → GELU → Dropout2d.

    Padding convention (same as ConvLayer2D in ConvModules.py):
        max(stride) == 1  →  "same"   (output spatial dims unchanged)
        max(stride) >  1  →  "valid"  (no padding; dims shrink by kernel/stride)

    Args:
        in_channels:   input channels
        out_channels:  output channels
        kernel_size:   (freq_kernel, time_kernel)
        stride:        (freq_stride, time_stride)
        dropout_ratio: Dropout2d probability
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_ratio=0.0):
        super().__init__()
        kernel_size = tuple(kernel_size)
        stride = tuple(stride)
        padding = "same" if max(stride) == 1 else "valid"
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, padding_mode="zeros", bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p=dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class DeepSenseBackbone(nn.Module):
    """
    Configurable DeepSense backbone for a single input tensor.

    Architecture
    ------------
    Conv stack (N = len(channels) layers):
        Layer i: DSConvLayer(in=channels[i-1], out=channels[i])
                 Layer 0 input is in_channels (from data).
        Input:  [B, in_channels, intervals, spectrum]
        Output: [B, channels[-1], intervals', spectrum']

    Spectrum pool (AdaptiveAvgPool2d → squeeze):
        → [B, channels[-1], intervals']

    RecurrentBlock (bidirectional GRU):
        → [B, recurrent_dim * 2]

    sample_embd_layer (Linear + ReLU):
        → [B, fc_dim]   ← 'features'

    class_layer (Linear):
        → [B, num_classes]  ← 'logits'

    Args
    ----
    in_channels:      input channels (from dataset config)
    num_classes:      number of output classes
    channels:         list of output channels per conv layer, e.g. [64, 128, 128]
    kernel_sizes:     list of (freq_k, time_k) per conv layer
    strides:          list of (freq_s, time_s) per conv layer
                      len(channels) == len(kernel_sizes) == len(strides) required
    recurrent_dim:    GRU hidden size (output dim is recurrent_dim * 2, bidirectional)
    recurrent_layers: number of GRU layers
    fc_dim:           feature embedding dimension
    dropout_ratio:    dropout probability for conv layers and GRU

    Returns
    -------
    dict: {'logits': [B, num_classes], 'features': [B, fc_dim]}
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels,
        kernel_sizes,
        strides,
        recurrent_dim: int,
        recurrent_layers: int,
        fc_dim: int,
        dropout_ratio: float = 0.0,
    ):
        super().__init__()

        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(
                f"channels, kernel_sizes, and strides must all have the same length; "
                f"got {len(channels)}, {len(kernel_sizes)}, {len(strides)}"
            )

        # Build conv stack: channels[i] is output for layer i
        conv_layers = []
        in_ch = in_channels
        for out_ch, k, s in zip(channels, kernel_sizes, strides):
            conv_layers.append(DSConvLayer(in_ch, out_ch, k, s, dropout_ratio))
            in_ch = out_ch
        self.conv_stack = nn.ModuleList(conv_layers)

        # Pool spectrum dim to 1 → [B, channels[-1], intervals']
        self.spectrum_pool = nn.AdaptiveAvgPool2d((None, 1))

        # Bidirectional GRU; input dim = channels[-1]
        self.recurrent_layer = RecurrentBlock(
            in_channel=channels[-1],
            out_channel=recurrent_dim,
            num_layers=recurrent_layers,
            dropout_ratio=dropout_ratio,
        )

        # Embedding projection
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(recurrent_dim * 2, fc_dim),  # *2 for bidirectional GRU
            nn.ReLU(),
        )

        # Classifier
        self.class_layer = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, in_channels, intervals, spectrum]
        Returns:
            {'logits': [B, num_classes], 'features': [B, fc_dim]}
        """
        # Conv stack: [B, C, I, S] → [B, channels[-1], I', S']
        for layer in self.conv_stack:
            x = layer(x)

        # Pool spectrum: → [B, channels[-1], I']
        x = self.spectrum_pool(x).squeeze(-1)

        # GRU: → [B, recurrent_dim * 2]
        recurrent_out, _ = self.recurrent_layer(x)

        # Embed: → [B, fc_dim]
        features = self.sample_embd_layer(recurrent_out)

        # Classify: → [B, num_classes]
        logits = self.class_layer(features)

        return {'logits': logits, 'features': features}


class SingleModalDeepSense(nn.Module):
    """
    Thin wrapper around DeepSenseBackbone — mirrors SingleModalResNet.

    Unpacks freq_x[location_name][modality_name] and forwards the tensor
    to the backbone. No architecture logic lives here.

    Input:  freq_x[location_name][modality_name] = [B, C, intervals, spectrum]
    Output: {'logits': [B, num_classes], 'features': [B, fc_dim]}
    """

    def __init__(
        self,
        modality_name: str,
        location_name: str,
        in_channels: int,
        num_classes: int,
        channels,
        kernel_sizes,
        strides,
        recurrent_dim: int,
        recurrent_layers: int,
        fc_dim: int,
        dropout_ratio: float = 0.0,
    ):
        super().__init__()
        self.modality_name = modality_name
        self.location_name = location_name

        self.backbone = DeepSenseBackbone(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            recurrent_dim=recurrent_dim,
            recurrent_layers=recurrent_layers,
            fc_dim=fc_dim,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, freq_x: dict) -> dict:
        x = freq_x[self.location_name][self.modality_name]
        return self.backbone(x)
```

---

## Task 2: Add `create_deepsense()` to `create_models.py`

**Files:**
- Modify: `src2/models/create_models.py`

- [ ] **Step 1: Add import at top (after line 13)**

```python
from models.DeepSenseLatest import SingleModalDeepSense
```

- [ ] **Step 2: Add `create_deepsense()` after `create_single_modal_model()` (after line 170)**

```python
def create_deepsense(config, model_config_key):
    """
    Create a SingleModalDeepSense from YAML config.

    Required keys under config["models"][model_config_key]:
        model_type:       "deepsense"
        active_modality:  str
        channels:         list[int]           — output channels per conv layer
        kernel_sizes:     list[[int, int]]    — (freq_k, time_k) per layer
        strides:          list[[int, int]]    — (freq_s, time_s) per layer
                          len(channels) == len(kernel_sizes) == len(strides) required
        recurrent_dim:    int
        recurrent_layers: int
        fc_dim:           int
        dropout_ratio:    float

    in_channels looked up from:
        config["loc_mod_in_freq_channels"][location][modality]
    """
    model_cfg = config["models"][model_config_key]
    active_modality = model_cfg["active_modality"]

    location_names = config["location_names"]
    if len(location_names) != 1:
        raise ValueError(
            f"SingleModalDeepSense expects exactly one location, got {location_names}"
        )
    location_name = location_names[0]

    in_channels = config["loc_mod_in_freq_channels"][location_name][active_modality]
    num_classes = config["vehicle_classification"]["num_classes"]

    channels = model_cfg["channels"]
    kernel_sizes = model_cfg["kernel_sizes"]
    strides = model_cfg["strides"]

    if not (len(channels) == len(kernel_sizes) == len(strides)):
        raise ValueError(
            f"[{model_config_key}] channels, kernel_sizes, and strides must have equal length; "
            f"got {len(channels)}, {len(kernel_sizes)}, {len(strides)}"
        )

    logging.info(f"Creating SingleModalDeepSense ({model_config_key})")
    logging.info(f"  modality={active_modality}, location={location_name}")
    logging.info(f"  in_channels={in_channels}, conv_layers={len(channels)}")
    logging.info(f"  channels={channels}")
    logging.info(f"  kernel_sizes={kernel_sizes}, strides={strides}")

    model = SingleModalDeepSense(
        modality_name=active_modality,
        location_name=location_name,
        in_channels=in_channels,
        num_classes=num_classes,
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        recurrent_dim=model_cfg["recurrent_dim"],
        recurrent_layers=model_cfg["recurrent_layers"],
        fc_dim=model_cfg["fc_dim"],
        dropout_ratio=model_cfg["dropout_ratio"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  Parameters: {total_params:,} ({total_params / 1e6:.4f}M)")
    return model
```

- [ ] **Step 3: Add dispatch in `create_single_modal_model()`**

Insert before the `else` block (after the `convonly` elif, around line 143):
```python
    elif model_type == "deepsense":
        return create_deepsense(config, model_config_key)
```

Update the error message:
```python
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: 'student_resnet', 'resnet', 'student_convonly', 'convonly', 'deepsense'"
        )
```

---

## Task 3: Add DeepSense configs to `Parkland.yaml`

**Files:**
- Modify: `src2/data/Parkland.yaml`

- [ ] **Step 1: Add two DeepSense model configs to `models:` section**

Insert after `student_audio_convonly` block (after line 134), before the training_configs comment:

```yaml
        # Student 3: Audio-only DeepSense (small — 1 GRU layer)
        student_audio_deepsense_small:
                model_source: "create_single_modal_model"
                model_type: "deepsense"
                active_modality: "audio"
                channels: [32, 64, 64]
                kernel_sizes: [[1, 80], [1, 5], [1, 5]]
                strides: [[1, 80], [1, 1], [1, 1]]
                recurrent_dim: 64
                recurrent_layers: 1
                fc_dim: 128
                dropout_ratio: 0.2

        # Student 4: Audio-only DeepSense (standard — mirrors original config)
        student_audio_deepsense:
                model_source: "create_single_modal_model"
                model_type: "deepsense"
                active_modality: "audio"
                channels: [64, 128, 128]
                kernel_sizes: [[1, 80], [1, 5], [1, 5]]
                strides: [[1, 80], [1, 1], [1, 1]]
                recurrent_dim: 256
                recurrent_layers: 2
                fc_dim: 512
                dropout_ratio: 0.2
```

- [ ] **Step 2: Add experiment entries**

In the `experiments:` section, add after `cascade_stage2`:
```yaml
        only_audio_deepsense_small:
                model: "student_audio_deepsense_small"
                training: "vanilla_supervised_ce"
                fixed_augmenters:
                        time_augmenters: ["no"]
                        freq_augmenters: ["no"]

        only_audio_deepsense:
                model: "student_audio_deepsense"
                training: "vanilla_supervised_ce"
                fixed_augmenters:
                        time_augmenters: ["no"]
                        freq_augmenters: ["no"]
```

---

## Task 4: Create and run test script

**Files:**
- Create: `src2/models/test_deepsense_creation.py`

- [ ] **Step 1: Write the test script**

```python
"""
Test: SingleModalDeepSense model creation and single-batch inference.

Run from repo root:
    cd /home/misra8/demo_codebase
    python src2/models/test_deepsense_creation.py

Expected: all three models pass shape checks and print "All tests passed!"
"""
import sys
from pathlib import Path

src2_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src2_dir))

import torch
import yaml

from models.create_models import create_single_modal_model


def make_dummy_batch(config, modality, location, batch_size=4):
    C = config["loc_mod_in_freq_channels"][location][modality]
    S = config["loc_mod_spectrum_len"][location][modality]
    I = config["num_segments"]
    return {location: {modality: torch.randn(batch_size, C, I, S)}}


def test_model(config, model_key, batch_size=4):
    print(f"\n{'='*60}\nTesting: {model_key}\n{'='*60}")

    model = create_single_modal_model(config, model_key)
    model.eval()

    model_cfg = config["models"][model_key]
    modality = model_cfg["active_modality"]
    location = config["location_names"][0]

    dummy_input = make_dummy_batch(config, modality, location, batch_size)
    tensor_shape = dummy_input[location][modality].shape
    print(f"Input: {location}/{modality} -> {tuple(tensor_shape)}")

    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, dict), f"Expected dict, got {type(output)}"
    assert "logits" in output and "features" in output, f"Missing keys: {output.keys()}"

    num_classes = config["vehicle_classification"]["num_classes"]
    fc_dim = model_cfg["fc_dim"]

    assert output["logits"].shape == (batch_size, num_classes), \
        f"logits shape {output['logits'].shape} != ({batch_size}, {num_classes})"
    assert output["features"].shape == (batch_size, fc_dim), \
        f"features shape {output['features'].shape} != ({batch_size}, {fc_dim})"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.3f}M)")
    print(f"logits:   {tuple(output['logits'].shape)}   ✓")
    print(f"features: {tuple(output['features'].shape)}  ✓")
    print("PASSED")


def main():
    yaml_path = src2_dir / "data" / "Parkland.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    test_model(config, "student_audio_deepsense_small")
    test_model(config, "student_audio_deepsense")
    test_model(config, "student_audio_resnet")   # sanity: existing model unbroken

    print(f"\n{'='*60}\nAll tests passed!\n{'='*60}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the test**

```bash
cd /home/misra8/demo_codebase
python src2/models/test_deepsense_creation.py
```

Expected:
```
Testing: student_audio_deepsense_small
Input: shake/audio -> (4, 2, 10, 1600)
Parameters: X,XXX (X.XXXM)
logits:   (4, 7)   ✓
features: (4, 128)  ✓
PASSED

Testing: student_audio_deepsense
...PASSED

Testing: student_audio_resnet
...PASSED

All tests passed!
```

---

## Verification

1. **Import check** (from `src2/`):
   ```bash
   cd src2 && python -c "from models.DeepSenseLatest import SingleModalDeepSense; print('import ok')"
   ```
2. **Shape test:** `python src2/models/test_deepsense_creation.py` — all three pass.
3. **Training smoke test (optional):**
   ```bash
   python src2/train_test/train.py -experiment_name only_audio_deepsense_small \
       -yaml_path src2/data/Parkland.yaml -gpu -1
   ```
