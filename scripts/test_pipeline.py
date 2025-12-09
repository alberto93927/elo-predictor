#!/usr/bin/env python3
"""
Quick pipeline test for transformer and LSTM models.
Uses small data subset for fast verification.
"""

import os
import sys
import pickle
import torch
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import create_data_loaders
from src.models.transformer import TransformerEncoder
from src.models.lstm import LSTMEloPredictor
from src.train import Trainer
from src.utils import EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_model(model_name: str, model: torch.nn.Module, train_loader, val_loader, device):
    """Test a model with quick training."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name.upper()} Model")
    logger.info(f"{'='*60}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Train for 2 epochs
    logger.info("Training for 2 epochs...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        early_stopping=False,  # No early stopping for test
        checkpoint_dir="test_checkpoints",
        model_name=model_name,
    )

    logger.info(f"\n✓ {model_name.upper()} training completed!")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"  Final train MAE: {history['train_mae'][-1]:.1f} Elo")
    logger.info(f"  Final val MAE: {history['val_mae'][-1]:.1f} Elo")

    return history


def main():
    """Run pipeline tests."""
    print("\n" + "="*60)
    print("QUICK PIPELINE TEST - Transformer & LSTM")
    print("="*60)

    # Load small subset of data
    logger.info("\nLoading small data subset...")
    with open('data/processed/dataset_splits.pkl', 'rb') as f:
        splits = pickle.load(f)

    # Use only 500 training, 100 val, 100 test
    train_small = splits['train'][:500]
    val_small = splits['val'][:100]
    test_small = splits['test'][:100]

    logger.info(f"  Train: {len(train_small)} games")
    logger.info(f"  Val: {len(val_small)} games")
    logger.info(f"  Test: {len(test_small)} games")

    # Create dataloaders with small settings
    logger.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, encoder = create_data_loaders(
        train_small, val_small, test_small,
        batch_size=32,
        max_sequence_length=100,  # Shorter for speed
    )

    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Test 1: Transformer
    logger.info("\n" + "="*60)
    logger.info("TEST 1: TRANSFORMER MODEL")
    logger.info("="*60)

    transformer = TransformerEncoder(
        input_channels=13,
        embedding_dim=64,  # Smaller for speed
        num_layers=2,      # Fewer layers
        num_heads=4,       # Fewer heads
        feedforward_dim=128,  # Smaller FFN
        dropout=0.1,
        max_seq_length=100,
    )

    transformer_history = test_model('transformer', transformer, train_loader, val_loader, device)

    # Test 2: LSTM
    logger.info("\n" + "="*60)
    logger.info("TEST 2: LSTM MODEL")
    logger.info("="*60)

    lstm = LSTMEloPredictor(
        input_channels=13,
        embedding_dim=64,
        lstm_hidden_dim=128,
        num_lstm_layers=2,
        dropout=0.1,
        bidirectional=True,
    )

    lstm_history = test_model('lstm', lstm, train_loader, val_loader, device)

    # Summary
    print("\n" + "="*60)
    print("PIPELINE TEST SUMMARY")
    print("="*60)
    print(f"\n✓ Transformer: PASSED")
    print(f"  - Final validation MAE: {transformer_history['val_mae'][-1]:.1f} Elo")
    print(f"  - Training completed without errors")

    print(f"\n✓ LSTM: PASSED")
    print(f"  - Final validation MAE: {lstm_history['val_mae'][-1]:.1f} Elo")
    print(f"  - Training completed without errors")

    print(f"\n{'='*60}")
    print("✓✓✓ ALL PIPELINE TESTS PASSED ✓✓✓")
    print(f"{'='*60}\n")

    logger.info("Both models are ready for full training!")


if __name__ == "__main__":
    main()
