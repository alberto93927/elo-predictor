"""
Utility functions for training and evaluation.
"""

import torch
import torch.nn as nn
import os
from typing import Dict, Optional
import json
from datetime import datetime


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        """Initialize meter."""
        self.reset()

    def reset(self):
        """Reset all values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update with a new value.

        Args:
            val: Value to add
            n: Weight of the value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "model",
) -> str:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        checkpoint_dir: Directory to save checkpoint
        model_name: Name of the model

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir, f"{model_name}_epoch_{epoch}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Dictionary with checkpoint info (epoch, best_val_loss)
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
    }


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    encoder=None,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        predictions: Model predictions (batch_size, 1) - normalized
        targets: Target values (batch_size,) - normalized
        encoder: FENEncoder for denormalization

    Returns:
        Dictionary with metrics (MSE, MAE, RMSE, etc.)
    """
    # Ensure same shape
    if predictions.dim() > 1:
        predictions = predictions.squeeze()

    # Calculate losses
    mse = torch.nn.functional.mse_loss(predictions, targets)
    mae = torch.nn.functional.l1_loss(predictions, targets)
    rmse = torch.sqrt(mse)

    metrics = {
        "mse": mse.item(),
        "mae": mae.item(),
        "rmse": rmse.item(),
    }

    # If encoder provided, denormalize and report in Elo points
    if encoder is not None:
        pred_elo = torch.tensor([
            encoder.denormalize_elo(p.item()) for p in predictions
        ])
        target_elo = torch.tensor([
            encoder.denormalize_elo(t.item()) for t in targets
        ])

        mae_elo = torch.nn.functional.l1_loss(pred_elo.float(), target_elo.float())
        rmse_elo = torch.sqrt(torch.nn.functional.mse_loss(
            pred_elo.float(), target_elo.float()
        ))

        metrics["mae_elo"] = mae_elo.item()
        metrics["rmse_elo"] = rmse_elo.item()

    return metrics


def log_metrics(
    epoch: int,
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Optional[Dict] = None,
    log_file: Optional[str] = None,
):
    """
    Log training metrics.

    Args:
        epoch: Current epoch
        train_metrics: Training metrics
        val_metrics: Validation metrics
        test_metrics: Test metrics (optional)
        log_file: File to log to (optional)
    """
    log_str = f"Epoch {epoch:3d} | "
    log_str += f"Train MAE: {train_metrics['mae']:.6f} | "
    log_str += f"Val MAE: {val_metrics['mae']:.6f}"

    if "mae_elo" in train_metrics:
        log_str += f" | Train MAE (Elo): {train_metrics['mae_elo']:.2f} | "
        log_str += f"Val MAE (Elo): {val_metrics['mae_elo']:.2f}"

    print(log_str)

    if log_file:
        with open(log_file, "a") as f:
            f.write(log_str + "\n")


def save_experiment_config(
    config: Dict,
    output_dir: str = "experiments",
) -> str:
    """
    Save experiment configuration.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save config

    Returns:
        Path to saved config
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(output_dir, f"config_{timestamp}.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path
