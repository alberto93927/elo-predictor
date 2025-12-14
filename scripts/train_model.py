#!/usr/bin/env python3
"""
Training script for Elo prediction models.
"""

import os
import sys
import pickle
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import create_data_loaders
from src.models.transformer import TransformerEncoder
from src.models.lstm import LSTMEloPredictor
from src.train import Trainer
from src.utils import save_experiment_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(data_dir: str = "data/processed"):
    """Load preprocessed dataset splits."""
    splits_path = os.path.join(data_dir, "dataset_splits.pkl")

    if not os.path.exists(splits_path):
        logger.error(f"Dataset splits not found at {splits_path}")
        logger.error("Run scripts/preprocess_data.py first")
        return None

    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)

    return splits['train'], splits['val'], splits['test']


def train_model(
    train_games,
    val_games,
    test_games,
    model="transformer",
    batch_size=32,
    epochs=50,
    lr=1e-3,
    dropout=0.1,
    embedding_dim=128,
    seed=42,
    device_str=None,
    output_dir="experiments",
    early_stopping=True,
):
    """
    Train an Elo prediction model.
    
    Args:
        train_games: Training games list
        val_games: Validation games list
        test_games: Test games list
        model: Model architecture ("transformer" or "lstm")
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        dropout: Dropout rate
        embedding_dim: Embedding dimension
        seed: Random seed
        device_str: Device string ("cpu" or "cuda"), auto-detected if None
        output_dir: Output directory for results
        early_stopping: Whether to use early stopping
        
    Returns:
        Tuple of (trainer, history, test_metrics)
    """
    # Set random seed
    torch.manual_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Train: {len(train_games)}, Val: {len(val_games)}, Test: {len(test_games)}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader, encoder = create_data_loaders(
        train_games, val_games, test_games,
        batch_size=batch_size,
    )

    # Create model
    logger.info(f"Creating {model} model...")
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if model == "transformer":
        model_obj = TransformerEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_layers=4,
            num_heads=8,
        )
    else:  # lstm
        model_obj = LSTMEloPredictor(
            embedding_dim=embedding_dim,
            lstm_hidden_dim=256,
            dropout=dropout,
        )

    model_obj = model_obj.to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model_obj.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Create trainer
    trainer = Trainer(
        model_obj, optimizer, criterion, device,
        encoder=encoder, scheduler=scheduler
    )

    # Save config
    config = {
        "model": model,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "dropout": dropout,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "device": str(device),
    }
    config_path = save_experiment_config(config, output_dir)
    logger.info(f"Saved config to {config_path}")

    # Train
    logger.info("Starting training...")
    log_file = os.path.join(output_dir, "training_log.txt")

    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=epochs,
        early_stopping=early_stopping,
        patience=5,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        model_name=model,
        log_file=log_file,
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.test(test_loader)
    logger.info(f"Test MAE: {test_metrics['mae']:.6f}")
    if 'mae_elo' in test_metrics:
        logger.info(f"Test MAE (Elo): {test_metrics['mae_elo']:.2f}")

    logger.info("\nTraining complete!")
    
    return trainer, history, test_metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train Elo prediction model"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer", "lstm"],
        default="transformer",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with preprocessed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Use early stopping",
    )

    args = parser.parse_args()

    # Load dataset
    logger.info("Loading dataset...")
    train_games, val_games, test_games = load_dataset(args.data_dir)

    if train_games is None:
        return

    # Call train_model function
    train_model(
        train_games, val_games, test_games,
        model=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        device_str=args.device,
        output_dir=args.output_dir,
        early_stopping=args.early_stopping,
    )


if __name__ == "__main__":
    main()
