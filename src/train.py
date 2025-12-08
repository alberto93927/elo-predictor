"""
Training loop for Elo prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm

from src.utils import (
    AverageMeter, EarlyStopping, calculate_metrics, save_checkpoint,
    load_checkpoint, log_metrics, save_experiment_config
)


logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for model training and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        encoder=None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            encoder: FENEncoder for denormalization
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.encoder = encoder
        self.scheduler = scheduler

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """
        Train for one epoch.

        Args:
            train_loader: Training dataloader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        loss_meter = AverageMeter()
        mae_meter = AverageMeter()

        pbar = tqdm(train_loader, desc="Training")

        for batch in pbar:
            sequences = batch["sequence"].to(self.device)
            lengths = batch["length"].to(self.device)
            targets = batch["elo"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions, _ = self.model(sequences, lengths)
            loss = self.criterion(predictions.squeeze(), targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update meters
            loss_meter.update(loss.item(), len(sequences))
            mae = torch.nn.functional.l1_loss(predictions.squeeze(), targets)
            mae_meter.update(mae.item(), len(sequences))

            pbar.set_postfix({"loss": loss_meter.avg, "mae": mae_meter.avg})

        metrics = {
            "loss": loss_meter.avg,
            "mae": mae_meter.avg,
        }

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict:
        """
        Validate on validation set.

        Args:
            val_loader: Validation dataloader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        loss_meter = AverageMeter()
        metrics_dict = {"mse": 0, "mae": 0, "rmse": 0}

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")

            for batch in pbar:
                sequences = batch["sequence"].to(self.device)
                lengths = batch["length"].to(self.device)
                targets = batch["elo"].to(self.device)

                # Forward pass
                predictions, _ = self.model(sequences, lengths)
                loss = self.criterion(predictions.squeeze(), targets)

                loss_meter.update(loss.item(), len(sequences))

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

                pbar.set_postfix({"loss": loss_meter.avg})

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets, self.encoder)
        metrics["loss"] = loss_meter.avg

        return metrics

    def test(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate on test set.

        Args:
            test_loader: Test dataloader

        Returns:
            Dictionary with test metrics
        """
        return self.validate(test_loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping: bool = True,
        patience: int = 5,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
        log_file: Optional[str] = None,
    ) -> Dict:
        """
        Train model.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs to train
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            model_name: Name of the model
            log_file: File to log metrics to

        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
        }

        best_val_loss = float('inf')
        early_stop = EarlyStopping(patience=patience) if early_stopping else None

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_mae"].append(train_metrics["mae"])
            history["val_mae"].append(val_metrics["mae"])

            # Log metrics
            log_metrics(epoch, train_metrics, val_metrics, log_file=log_file)

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    self.model, self.optimizer, epoch, best_val_loss,
                    checkpoint_dir, model_name
                )

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])

            # Early stopping
            if early_stop is not None and early_stop(val_metrics["loss"]):
                print(f"Early stopping at epoch {epoch}")
                break

        return history
