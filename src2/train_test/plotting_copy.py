import sys
import logging
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import (
    setup_experiment_dir, train, setup_optimizer, setup_scheduler
)
from train_test.normalize import setup_normalization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """Main training function."""
    
    # ========================================================================
    # 1. Load Configuration
    # ========================================================================
    logging.info("=" * 80)
    logging.info("TRAINING SCRIPT")
    logging.info("=" * 80)
    
    config = get_config()
    logging.info("Configuration loaded successfully")
    logging.info(f"  Model: {config.get('model', 'Unknown')}")
    logging.info(f"  Model variant: {config.get('model_variant', 'None')}")
    logging.info(f"  Dataset: {config.get('yaml_path', 'Unknown')}")
    logging.info(f"  Device: {config.get('device', 'cpu')}")
    
    # ========================================================================
    # 2. Create Dataloaders
    # ========================================================================
    logging.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info("  Train batches: {}".format(len(train_loader)))
    logging.info("  Val batches: {}".format(len(val_loader)))
    logging.info("  Test batches: {}".format(len(test_loader)))
    
    
    # ========================================================================
    # 3. Create Augmenter
    # ========================================================================
    logging.info("\nCreating augmenter...")
    # Get augmentation mode from config or use default
    model_name = config.get("model", "ResNet")
    augmenters_config = config.get(model_name, {}).get("fixed_augmenters", {})
    learning_rate = config.get(model_name, {}).get("lr_scheduler", {}).get("train_epochs", "Unknown")
    optimizer_name = config.get(model_name, {}).get("optimizer", {}).get("name", "Unknown")
    scheduler_name = config.get(model_name, {}).get("lr_scheduler", {}).get("name", "Unknown")
    
    # For training, we typically want augmentation enabled
    augmenter = create_augmenter(config, augmentation_mode="fixed")
    logging.info("Augmenter created successfully")
    
    # ========================================================================
    # 4. Create Model
    # ========================================================================
    logging.info("\nCreating model...")
    model = create_model(config)
    
    # ========================================================================
    # 5. Setup Experiment Directory
    # ========================================================================
    logging.info("\nSetting up experiment directory...")
    experiment_dir, tensorboard_dir = setup_experiment_dir(config)
    
    # ========================================================================
    # 8. Train Model
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TRAINING")
    logging.info("=" * 80 + "\n")

            # Use standard training
    logging.info("Using standard training (quantization disabled)")
    
    model, train_history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment_dir=experiment_dir,
        augmenter=augmenter,
        apply_augmentation_fn=apply_augmentation
    )

def train(model, train_loader, val_loader, config, experiment_dir, 
        loss_fn=None, val_fn=None,
        augmenter=None, apply_augmentation_fn=None):
    """
    Train the model with comprehensive logging and checkpointing.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        experiment_dir: Path to experiment directory
        loss_fn: Loss function (if None, uses CrossEntropyLoss)
        val_fn: Custom validation function (optional)
        augmenter: Data augmenter object (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
    
    Returns:
        model: Trained model
        train_history: Dictionary with training history
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    
    # Setup loss function
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    # Setup optimizer
    if optimizer is None:
        lr = config.get(config['model'], {}).get('optimizer', {}).get('start_lr', 0.001)
        weight_decay = config.get(config['model'], {}).get('optimizer', {}).get('weight_decay', 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup directories
    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    models_dir = experiment_path / "models"
    tensorboard_dir = experiment_path / "tensorboard"
    
    # Setup logging
    log_file = logs_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('train')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    # Setup TensorBoard
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Training parameters
    num_epochs = config.get(config['model'], {}).get('lr_scheduler', {}).get('train_epochs', 50)
    num_classes = config.get('vehicle_classification', {}).get('num_classes', 7)
    class_names = config.get('vehicle_classification', {}).get('class_names', None)
    
    # Training history
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0
    
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info("=" * 80)
    
    for epoch in range(num_epochs):
        # ====================================================================
        # Training Phase
        # ====================================================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, idx = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]
            
            # Apply augmentation if provided
            if augmenter is not None and apply_augmentation_fn is not None:
                data, labels = apply_augmentation_fn(augmenter, data, labels)

            shake = data["shake"]
            audio = shake["audio"]    # [B, 2, T, F]
            seis  = shake["seismic"]  # [B, 2, T, F]

            # Power = real^2 + imag^2
            audio_power = torch.sqrt(audio[0, 0, 1, :]**2 + audio[0, 1, 1, :]**2)   # [F]
            seis_power  = seis[0, 0, 5, :]**2  + seis[0, 1, 5, :]**2    # [F]

            import sys
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            # ===== SELECT ONE SAMPLE =====
            x0 = audio[0]            # [2, T, F]
            real = x0[0]             # [T, F]
            imag = x0[1]             # [T, F]
            # ===== RAW MAGNITUDE =====
            mag = torch.sqrt(real*real + imag*imag + 1e-12)   # [T, F]
            # ===== POSITIVE FREQUENCY HALF =====
            F = mag.shape[1]
            half = F // 2
            mag = mag[:, :half]      # [T, F_half]
            # ===== SCALE =====
            mag = mag / 1e5
            # ===== 1D FFT SLICE =====
            audio_power = mag[1].cpu().numpy()    # [F_half]
            freq_bins = np.linspace(0, 800, half)
            # ===== 2D MAP =====
            M = mag.cpu().numpy()   # [T, F_half]
            # ============================
            # COMBINED FIGURE
            # ============================
            fig = plt.figure(figsize=(10, 4))
            # Use nested GridSpec for different spacing
            from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
            # Main grid: colorbar+spectrogram on left, FFT on right
            gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.43)
            # Left side: nested grid for colorbar and spectrogram
            gs_left = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], 
                                            width_ratios=[0.05, 1], wspace=.15)
            # ===== COLORBAR AXIS (FAR LEFT) =====
            cbar_ax = fig.add_subplot(gs_left[0])
            # ===== LEFT: SPECTROGRAM =====
            ax1 = fig.add_subplot(gs_left[1])
            im = ax1.imshow(
                M.T,
                origin="lower",
                aspect="auto",
                extent=[0.0, 2.0, 0.0, 800.0],
                cmap="viridis"
            )
            ax1.set_xlabel("Time (sec)", fontsize=20, fontweight="bold")
            # Frequency ticks on right, no label
            ax1.yaxis.tick_right()
            ax1.yaxis.set_label_position("right")
            for spine in ax1.spines.values():
                spine.set_linewidth(2)
            ax1.tick_params(axis="both", labelsize=12, width=2, length=6)
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                label.set_fontweight("bold")
            # ---- Colorbar in separate axis ----
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label("Magnitude (×10⁵)", fontsize=20, fontweight="bold")
            # Move colorbar ticks and label to the LEFT side
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_label_position('left')
            cbar.ax.tick_params(labelsize=16, width=2, length=5)
            for label in cbar.ax.get_yticklabels():
                label.set_fontweight("bold")
            # ===== RIGHT: FFT =====
            ax2 = fig.add_subplot(gs_main[1])
            ax2.plot(audio_power, freq_bins, color="red", linewidth=1.5)
            ax2.set_xlabel("Audio FFT (×10⁵)", fontsize=20, fontweight="bold")
            ax2.set_ylabel("Frequency (Hz)", fontsize=20, fontweight="bold")
            ax2.set_ylim(0, 800)
            for spine in ax2.spines.values():
                spine.set_linewidth(2)
            ax2.tick_params(axis="both", labelsize=12, width=2, length=6)
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontweight("bold")
            ax2.grid(True, color="0.9")
            out = "audio_fft_and_spectrogram.png"
            plt.savefig(out, dpi=350, bbox_inches="tight")
            plt.close()
            print("SAVED:", out)

            # ===== KILL SCRIPT =====
            sys.exit(0)





            # x_bins = range(seis_power.shape[0])

            # fig = plt.figure() 
            # plt.plot(x_bins, seis_power.cpu(), color="blue") 
            # plt.xlabel("Frequency bin") 
            # plt.ylabel("Seismic FFT Power (batch=0, t=0)") 
            # plt.title("Seismic Power Spectrum (batch=0, t=0)") 
            # plt.grid(True)
            # out_path = "seis_power_b0_t0.png"
            # fig.savefig(out_path, dpi=200, bbox_inches="tight")
            # print(f"SAVED PLOT TO: {out_path}")
            # plt.close(fig)

            return model, train_history


if __name__ == "__main__":
    main()