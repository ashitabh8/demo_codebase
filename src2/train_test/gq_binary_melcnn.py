"""
Train a binary near/far audio classifier on GQ .pt samples.

Pipeline decisions implemented here:
- Audio tensor is loaded from sample["data"][location]["audio"], expected [1, C, T]
- Leading singleton dim is squeezed -> [C, T]
- Mel frontend via torchaudio MelSpectrogram:
    n_fft=400, hop_length=160, n_mels=64, sample_rate=16000
- log1p compression applied to mel magnitude
- Binary target: 1 if min(distance) < 10m, else 0
- Model: 3-block CNN with AdaptiveAvgPool2d + MLP head
- Loss: weighted CrossEntropyLoss with positive class upweighting
- Training: Adam + cosine annealing LR
- Best checkpoint selected by validation F1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


DISTANCE_THRESHOLD_M = 10.0
EPS = 1e-8


def hz_to_mel(freq_hz: float) -> float:
    return 2595.0 * math.log10(1.0 + (freq_hz / 700.0))


def mel_to_hz(mel: float) -> float:
    return 700.0 * ((10.0 ** (mel / 2595.0)) - 1.0)


def build_mel_filterbank(
    n_fft: int, n_mels: int, sample_rate: int, f_min: float, f_max: float
) -> torch.Tensor:
    n_freqs = (n_fft // 2) + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, steps=n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(float(m)) for m in mel_points], dtype=torch.float32)
    bin_freqs = torch.arange(n_freqs, dtype=torch.float32) * (sample_rate / n_fft)

    fb = torch.zeros((n_mels, n_freqs), dtype=torch.float32)
    for m in range(n_mels):
        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]

        rising = (bin_freqs >= left) & (bin_freqs <= center)
        falling = (bin_freqs > center) & (bin_freqs <= right)

        left_denom = max(float(center - left), EPS)
        right_denom = max(float(right - center), EPS)

        fb[m, rising] = (bin_freqs[rising] - left) / left_denom
        fb[m, falling] = (right - bin_freqs[falling]) / right_denom
    return fb


class TorchMelSpectrogram(nn.Module):
    """Pure-PyTorch MelSpectrogram fallback when torchaudio is unavailable."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: float | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = float(sample_rate / 2.0) if f_max is None else float(f_max)

        fb = build_mel_filterbank(
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [C, T]
        returns:  [C, n_mels, n_frames]
        """
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform [C, T], got {tuple(waveform.shape)}")

        window = torch.hann_window(
            self.n_fft, device=waveform.device, dtype=waveform.dtype
        )
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )  # [C, n_freqs, n_frames]
        power = stft.abs() ** 2
        fb = self.fb.to(device=power.device, dtype=power.dtype)
        mel = torch.einsum("mf,cft->cmt", fb, power)
        return mel


def create_mel_transform(
    sample_rate: int, n_fft: int, hop_length: int, n_mels: int
) -> nn.Module:
    try:
        import torchaudio

        print("Using torchaudio MelSpectrogram frontend")
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    except ModuleNotFoundError:
        print("torchaudio not found; falling back to pure-PyTorch mel frontend")
        return TorchMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )


def read_index_paths(index_file: Path) -> list[str]:
    with open(index_file) as f:
        return [line.strip() for line in f if line.strip()]


def resolve_sample_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    # Common mount translation on shared servers.
    prefix_src = "/home/tkimura4/data/"
    prefix_dst = "/data/tkimura4/data/"
    if path_str.startswith(prefix_src):
        remapped = Path(prefix_dst + path_str[len(prefix_src) :])
        if remapped.exists():
            return remapped
    return p


def sample_has_audio_flag(sample: dict) -> bool:
    flag = sample["flag"]
    for _location, modalities in flag.items():
        if not isinstance(modalities, dict):
            continue
        if "audio" not in modalities:
            continue
        if modalities["audio"] is True:
            return True
    return False


def iter_distance_meters(obj: object) -> list[float]:
    out: list[float] = []
    if not isinstance(obj, dict):
        return out
    for value in obj.values():
        if isinstance(value, dict):
            out.extend(iter_distance_meters(value))
        elif isinstance(value, (int, float)):
            out.append(float(value))
    return out


def min_distance_meters(sample: dict) -> float | None:
    if "distance" not in sample:
        return None
    distances = iter_distance_meters(sample["distance"])
    if not distances:
        return None
    return min(distances)


def extract_audio_waveform(sample: dict) -> torch.Tensor:
    """
    Returns audio waveform with shape [channels, time].
    Expected stored audio shape is typically [1, channels, time].
    """
    for location, flag_modalities in sample["flag"].items():
        if not isinstance(flag_modalities, dict):
            continue
        if "audio" not in flag_modalities:
            continue
        if flag_modalities["audio"] is not True:
            continue
        audio = sample["data"][location]["audio"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        if audio.dim() == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        if audio.dim() != 2:
            raise ValueError(f"Expected audio shape [C, T], got {tuple(audio.shape)}")
        return audio.float()
    raise ValueError("No audio modality with true flag found in sample")


def random_time_shift(x: torch.Tensor, frac: float) -> torch.Tensor:
    """Random circular shift over time axis by +/- frac * T samples."""
    time_len = int(x.shape[-1])
    max_shift = int(round(frac * time_len))
    if max_shift <= 0:
        return x
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(x, shifts=shift, dims=-1)


@dataclass
class SampleEntry:
    path: str
    label: int
    distance_m: float


class GQBinaryAudioDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        mel_transform: Any,
        train_mode: bool,
        shift_frac: float,
    ) -> None:
        self.mel_transform = mel_transform
        self.train_mode = train_mode
        self.shift_frac = shift_frac
        self.samples: list[SampleEntry] = []

        kept = 0
        skipped_no_audio = 0
        skipped_no_distance = 0
        skipped_load_error = 0

        for p in paths:
            resolved_path = resolve_sample_path(p)
            try:
                sample = torch.load(resolved_path, weights_only=False)
            except Exception:
                skipped_load_error += 1
                continue

            if not sample_has_audio_flag(sample):
                skipped_no_audio += 1
                continue

            dmin = min_distance_meters(sample)
            if dmin is None:
                skipped_no_distance += 1
                continue

            label = 1 if dmin < DISTANCE_THRESHOLD_M else 0
            self.samples.append(
                SampleEntry(path=str(resolved_path), label=label, distance_m=float(dmin))
            )
            kept += 1

        print(
            f"Dataset init ({'train' if train_mode else 'eval'}): kept={kept}, "
            f"skipped_no_audio={skipped_no_audio}, skipped_no_distance={skipped_no_distance}, "
            f"skipped_load_error={skipped_load_error}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, float]:
        item = self.samples[index]
        sample = torch.load(item.path, weights_only=False)
        audio = extract_audio_waveform(sample)  # [C, T]
        if self.train_mode:
            audio = random_time_shift(audio, self.shift_frac)

        mel = self.mel_transform(audio)  # [C, n_mels, n_frames]
        mel = torch.log1p(mel)
        y = torch.tensor(item.label, dtype=torch.long)
        return mel, y, item.path, item.distance_m


class MelCNN(nn.Module):
    def __init__(self, in_channels: int = 10, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def compute_class_weight(train_ds: GQBinaryAudioDataset) -> torch.Tensor:
    labels = [entry.label for entry in train_ds.samples]
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = sum(1 for label in labels if label == 0)
    if n_pos == 0:
        raise ValueError("No positive samples found in train split")
    pos_weight = float(n_neg) / float(n_pos)
    return torch.tensor([1.0, pos_weight], dtype=torch.float32)


def batch_confusion(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
    return tp, tn, fp, fn


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, EPS)
    return precision, recall, f1


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    model.eval()
    total_loss = 0.0
    total_count = 0

    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob_pos: list[float] = []
    all_paths: list[str] = []
    all_distance_m: list[float] = []

    with torch.no_grad():
        for x, y, paths, distance_m in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            batch_size = int(y.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_prob_pos.extend(probs[:, 1].cpu().tolist())
            all_paths.extend(list(paths))
            all_distance_m.extend([float(d) for d in distance_m])

    y_true = torch.tensor(all_true)
    y_pred = torch.tensor(all_pred)
    tp, tn, fp, fn = batch_confusion(y_true, y_pred)
    precision, recall, f1 = compute_prf(tp, fp, fn)
    accuracy = float((y_true == y_pred).float().mean().item()) if len(all_true) else 0.0

    unique = set(all_true)
    if len(unique) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(all_true, all_prob_pos))

    fn_records: list[dict[str, float | int | str]] = []
    for idx, (yt, yp) in enumerate(zip(all_true, all_pred)):
        if yt == 1 and yp == 0:
            fn_records.append(
                {
                    "path": all_paths[idx],
                    "y_true": int(yt),
                    "y_pred": int(yp),
                    "true_distance_m": float(all_distance_m[idx]),
                }
            )

    fn_distance_values = [float(record["true_distance_m"]) for record in fn_records]
    if fn_distance_values:
        fn_distance_mean_m = float(statistics.mean(fn_distance_values))
        fn_distance_variance_m2 = (
            float(statistics.variance(fn_distance_values))
            if len(fn_distance_values) > 1
            else 0.0
        )
    else:
        fn_distance_mean_m = float("nan")
        fn_distance_variance_m2 = float("nan")

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "fn_distance_mean_m": fn_distance_mean_m,
        "fn_distance_variance_m2": fn_distance_variance_m2,
    }
    return metrics, fn_records


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y, _paths, _distance_m in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def create_dataloaders(
    index_dir: Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    batch_size: int,
    num_workers: int,
    shift_frac: float,
) -> tuple[DataLoader, DataLoader, DataLoader, GQBinaryAudioDataset]:
    train_paths = read_index_paths(index_dir / "train_index.txt")
    val_paths = read_index_paths(index_dir / "val_index.txt")
    test_paths = read_index_paths(index_dir / "test_index.txt")

    mel = create_mel_transform(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    train_ds = GQBinaryAudioDataset(
        paths=train_paths,
        mel_transform=mel,
        train_mode=True,
        shift_frac=shift_frac,
    )
    val_ds = GQBinaryAudioDataset(
        paths=val_paths,
        mel_transform=mel,
        train_mode=False,
        shift_frac=0.0,
    )
    test_ds = GQBinaryAudioDataset(
        paths=test_paths,
        mel_transform=mel,
        train_mode=False,
        shift_frac=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, train_ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary near/far classifier with mel frontend + CNN"
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory with train_index.txt, val_index.txt, test_index.txt",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument(
        "--shift_frac",
        type=float,
        default=0.125,
        help="Random train-time circular time shift fraction of waveform length",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    index_dir = Path(args.index_dir)
    out_root = Path(args.out_dir)
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S_gq_binary_melcnn")
    run_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        index_dir=index_dir,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shift_frac=args.shift_frac,
    )

    class_weight = compute_class_weight(train_ds).to(device)
    print(f"Class weight [neg, pos]: {class_weight.tolist()}")

    first_x, _, _, _ = train_ds[0]
    print(f"Example model input shape [C, n_mels, n_frames]: {tuple(first_x.shape)}")

    model = MelCNN(in_channels=int(first_x.shape[0]), num_classes=2).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params}")

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_f1 = -math.inf
    history: list[dict[str, float]] = []
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics, _ = evaluate(
            model=model, loader=val_loader, criterion=criterion, device=device
        )
        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": float(epoch),
            "lr": lr_now,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc_roc": val_metrics["auc_roc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_auc={val_metrics['auc_roc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), last_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    best_val_metrics, best_val_fn_records = evaluate(
        model=model, loader=val_loader, criterion=criterion, device=device
    )
    test_metrics, test_fn_records = evaluate(
        model=model, loader=test_loader, criterion=criterion, device=device
    )

    results = {
        "config": {
            "index_dir": str(index_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sample_rate": args.sample_rate,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "n_mels": args.n_mels,
            "shift_frac": args.shift_frac,
            "distance_threshold_m": DISTANCE_THRESHOLD_M,
        },
        "model": {
            "name": "MelCNN",
            "num_parameters": n_params,
            "input_channels": int(first_x.shape[0]),
            "input_n_mels": int(first_x.shape[1]),
            "input_n_frames": int(first_x.shape[2]),
        },
        "class_weight": [float(class_weight[0].item()), float(class_weight[1].item())],
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    save_json(run_dir / "results.json", results)
    save_json(run_dir / "false_negatives_val.json", {"false_negatives": best_val_fn_records})
    save_json(run_dir / "false_negatives_test.json", {"false_negatives": test_fn_records})
    save_json(
        run_dir / "false_negative_summary.json",
        {
            "val": {
                "count": len(best_val_fn_records),
                "distance_mean_m": best_val_metrics["fn_distance_mean_m"],
                "distance_variance_m2": best_val_metrics["fn_distance_variance_m2"],
            },
            "test": {
                "count": len(test_fn_records),
                "distance_mean_m": test_metrics["fn_distance_mean_m"],
                "distance_variance_m2": test_metrics["fn_distance_variance_m2"],
            },
        },
    )
    print(f"Saved run artifacts to: {run_dir}")
    print("Best-by-val-F1 metrics:")
    print(json.dumps({"val": best_val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
