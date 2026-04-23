"""
Testing script variant that evaluates on a merged giant test index.

This replicates the current working test pipeline and hardcodes three source
index files, then merges them into one combined test index used for evaluation.
"""

import sys
import logging
import torch
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import parse_test_args
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_single_modal_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import load_checkpoint
from train_test.normalize import setup_normalization


# Hardcoded input index files to merge for giant test set.
MERGE_INDEX_PATHS = [
    "/data/misra8/GracesQuarters/index_files/2024-08-07-GQ-split-multiclass/train_index.txt",
    "/data/misra8/GracesQuarters/index_files/2024-08-07-GQ-split-multiclass/val_index.txt",
    "/data/misra8/GracesQuarters/index_files/2024-08-07-GQ-split-multiclass/test_index.txt",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def create_merged_test_index(test_dir: Path) -> Path:
    """Merge hardcoded index files into a single deduplicated test index."""
    ordered_unique_lines = []
    seen = set()

    for src_path_str in MERGE_INDEX_PATHS:
        src_path = Path(src_path_str)
        if not src_path.exists():
            raise FileNotFoundError(f"Merged index source not found: {src_path}")

        with open(src_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line == "":
                    continue
                if line not in seen:
                    seen.add(line)
                    ordered_unique_lines.append(line)

    merged_path = test_dir / "merged_giant_test_index.txt"
    with open(merged_path, "w") as f:
        for line in ordered_unique_lines:
            f.write(f"{line}\n")

    logging.info(f"Merged test index created: {merged_path}")
    logging.info(f"  Source files: {len(MERGE_INDEX_PATHS)}")
    logging.info(f"  Unique samples: {len(ordered_unique_lines)}")
    return merged_path


def main():
    logging.info("=" * 80)
    logging.info("TEST_NEXT SCRIPT - MERGED GIANT TEST SET")
    logging.info("=" * 80)

    args = parse_test_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logging.info("\nLoading configuration...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"  Config loaded from: {config_path}")
    logging.info(f"  Experiment name: {config['experiment_name']}")
    logging.info(f"  Dataset config: {config['yaml_path']}")

    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        logging.info(f"\nUsing specified checkpoint: {args.checkpoint_path}")
    else:
        checkpoint_path = experiment_dir / "models" / "best_model.pth"
        logging.info(f"\nUsing default best checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logging.info(f"Device: GPU {args.gpu}")
    else:
        device = torch.device("cpu")
        logging.info("Device: CPU")
    config["device"] = str(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = experiment_dir / f"test_next_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = test_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / "test.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logging.info(f"\nTest directory: {test_dir}")
    logging.info(f"Log file: {log_file}")

    merged_test_index_path = create_merged_test_index(test_dir)
    task_name = config["task_name"]
    config[task_name]["test_index_file"] = str(merged_test_index_path)
    logging.info(f"Overriding {task_name}.test_index_file with merged index")

    logging.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info(f"  Test batches: {len(test_loader)}")

    logging.info("\nExtracting model information...")
    experiment_name = config["experiment_name"]
    if "distillation" in config and experiment_name in config["distillation"]:
        experiment_config = config["distillation"][experiment_name]
        model_name = experiment_config["models"][0]
        loss_source_config = experiment_config["stages"][0]
    else:
        experiment_config = config["experiments"][experiment_name]
        model_name = experiment_config["model"]
        training_config_name = experiment_config["training"]
        loss_source_config = config["training_configs"][training_config_name]

    model_config = config["models"][model_name]
    logging.info(f"  Model: {model_name}")
    logging.info(f"  Architecture: {model_config['model_type']}")
    if "active_modality" in model_config:
        logging.info(f"  Modality: {model_config['active_modality']}")
    else:
        logging.info("  Modality: N/A")

    skip_normalization = False
    if "type" in loss_source_config and loss_source_config["type"] == "finetune":
        skip_normalization = True

    if skip_normalization:
        logging.info("\nSkipping normalization setup to match finetune.py behavior")
    else:
        logging.info("\nSetting up normalization...")
        train_loader, val_loader, test_loader = setup_normalization(
            train_loader, val_loader, test_loader, config
        )
        logging.info("Normalization setup complete")

    logging.info("\nCreating augmenter (test mode: disabled)...")
    augmenter = create_augmenter(config, augmentation_mode="no", experiment_config=experiment_config)
    logging.info("Augmenter created successfully (no augmentations will be applied)")

    logging.info("\nCreating model...")
    config["models"][model_name]["pretrain_mode"] = False
    model = create_single_modal_model(config, model_name)
    logging.info("Model created successfully")

    logging.info("\nLoading checkpoint...")
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()
    logging.info("Model loaded and set to eval mode")

    logging.info("\nSetting up loss function...")
    loss_fn, loss_fn_name = get_loss_function(loss_source_config)
    logging.info(f"  Loss function: {loss_fn_name}")

    logging.info("\n" + "=" * 80)
    logging.info("STARTING TESTING")
    logging.info("=" * 80)

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                data, labels, _ = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]

            if augmenter is not None:
                data, labels = apply_augmentation(augmenter, data, labels)

            labels = labels.to(device)
            if isinstance(data, dict):
                for loc in data:
                    for mod in data[loc]:
                        data[loc][mod] = data[loc][mod].to(device)
            else:
                data = data.to(device)

            outputs = model(data)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels

            loss = loss_fn(outputs, loss_labels)
            test_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(logits, dim=1)
            test_correct += (predictions == loss_labels).sum().item()
            test_total += labels.size(0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(loss_labels.cpu().numpy())

    test_loss /= test_total
    test_acc = test_correct / test_total

    if "task_name" in config and config["task_name"] in config and "class_names" in config[config["task_name"]]:
        class_names = config[config["task_name"]]["class_names"]
    else:
        class_names = []

    inferred_num_classes = len(class_names)
    if inferred_num_classes == 0:
        inferred_num_classes = int(max(max(all_labels), max(all_preds))) + 1
        class_names = [f"class_{i}" for i in range(inferred_num_classes)]

    cm = np.zeros((inferred_num_classes, inferred_num_classes), dtype=np.int64)
    for true_label, pred_label in zip(all_labels, all_preds):
        cm[int(true_label), int(pred_label)] += 1

    per_class_accuracy = []
    for class_idx in range(inferred_num_classes):
        class_total = int(cm[class_idx].sum())
        class_correct = int(cm[class_idx, class_idx])
        class_acc = (class_correct / class_total) if class_total > 0 else 0.0
        per_class_accuracy.append(class_acc)

    logging.info("\n" + "-" * 80)
    logging.info("TEST RESULTS (Merged Giant Test Set)")
    logging.info("-" * 80)
    logging.info(f"Loss: {test_loss:.4f}")
    logging.info(f"Accuracy: {test_acc:.4f}")
    logging.info("Per-class accuracy:")
    for class_idx, class_name in enumerate(class_names):
        logging.info(f"  {class_name}: {per_class_accuracy[class_idx]:.4f}")
    logging.info("Confusion matrix (rows=true, cols=pred):")
    logging.info(f"\n{cm}")
    logging.info("-" * 80)

    results_file = test_dir / "test_results.txt"
    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TEST_NEXT RESULTS (MERGED GIANT TEST SET)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Merged test index: {merged_test_index_path}\n")
        f.write(f"Source indices:\n")
        for p in MERGE_INDEX_PATHS:
            f.write(f"  - {p}\n")
        f.write(f"Test samples: {test_total}\n")
        f.write("\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write("\nPer-class accuracy:\n")
        for class_idx, class_name in enumerate(class_names):
            f.write(f"  {class_name}: {per_class_accuracy[class_idx]:.4f}\n")
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(f"{cm}\n")
        f.write("\n" + "=" * 80 + "\n")

    logging.info(f"  Results saved to: {results_file}")
    logging.info("\n" + "=" * 80)
    logging.info("TESTING COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)
    logging.info(f"Test directory: {test_dir}")
    logging.info(f"  - Log file: {log_file}")
    logging.info(f"  - Results file: {results_file}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
