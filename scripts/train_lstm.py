"""
TensorFlow LSTM training script for Elo prediction.
Matches the structure and features of train_overfit_transformer.py.
"""

import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from tensorflow import keras
import os
import sys
import argparse
from datetime import datetime

# Suppress TensorFlow warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GPU check
_skip_gpu_check = '--skip-gpu-init' in sys.argv or '--num-gpus' in sys.argv
_gpus_immediate = tf.config.list_physical_devices('GPU')
if not _skip_gpu_check:
    if len(_gpus_immediate) == 0:
        print("WARNING: No GPUs detected!")
    else:
        print(f"✓ Detected {len(_gpus_immediate)} GPU(s) immediately after TensorFlow import")

# Import src modules after path setup
from src.models.lstm_tf import build_lstm_model, print_model_summary, LSTMEloPredictor
from src.data.data_utils_tf import create_streaming_dataset


class ComprehensiveMetricsCallback(tf.keras.callbacks.Callback):
    """
    Comprehensive metrics tracking for rich visualizations.
    """
    
    ELO_RANGE = 2000  # 2800 - 800 = 2000 Elo range
    
    def __init__(self, log_dir, log_file=None):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        
        self.history = {
            'epoch': [],
            'lr': [],
            'time_seconds': [],
            'train_loss': [],
            'train_mae': [],
            'train_rmse_elo': [],
            'train_mae_elo': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse_elo': [],
            'val_mae_elo': [],
            'batch_loss_min': [],
            'batch_loss_max': [],
            'batch_loss_std': [],
            'batch_count': [],
        }
        
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_loss_epoch': 0,
            'best_val_mae_elo': float('inf'),
            'best_val_mae_elo_epoch': 0,
        }
        
        self._batch_losses = []
        self._epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self._batch_losses = []
        self._epoch_start_time = time.perf_counter()
    
    def on_train_batch_end(self, batch, logs=None):
        if logs and 'loss' in logs:
            self._batch_losses.append(logs['loss'])
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        epoch_time = time.perf_counter() - self._epoch_start_time
        
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except:
            lr = 0.0
        
        train_loss = logs.get('loss', 0)
        train_mae = logs.get('mae', 0)
        val_loss = logs.get('val_loss', 0)
        val_mae = logs.get('val_mae', 0)
        
        # Convert to Elo scale
        train_rmse_elo = np.sqrt(train_loss) * self.ELO_RANGE if train_loss else 0
        train_mae_elo = train_mae * self.ELO_RANGE if train_mae else 0
        val_rmse_elo = np.sqrt(val_loss) * self.ELO_RANGE if val_loss else 0
        val_mae_elo = val_mae * self.ELO_RANGE if val_mae else 0
        
        batch_losses = np.array(self._batch_losses) if self._batch_losses else np.array([0])
        
        self.history['epoch'].append(epoch + 1)
        self.history['lr'].append(lr)
        self.history['time_seconds'].append(epoch_time)
        self.history['train_loss'].append(float(train_loss))
        self.history['train_mae'].append(float(train_mae))
        self.history['train_rmse_elo'].append(float(train_rmse_elo))
        self.history['train_mae_elo'].append(float(train_mae_elo))
        self.history['val_loss'].append(float(val_loss))
        self.history['val_mae'].append(float(val_mae))
        self.history['val_rmse_elo'].append(float(val_rmse_elo))
        self.history['val_mae_elo'].append(float(val_mae_elo))
        self.history['batch_loss_min'].append(float(batch_losses.min()))
        self.history['batch_loss_max'].append(float(batch_losses.max()))
        self.history['batch_loss_std'].append(float(batch_losses.std()))
        self.history['batch_count'].append(len(self._batch_losses))
        
        if val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = float(val_loss)
            self.best_metrics['best_val_loss_epoch'] = epoch + 1
        
        if val_mae_elo < self.best_metrics['best_val_mae_elo']:
            self.best_metrics['best_val_mae_elo'] = float(val_mae_elo)
            self.best_metrics['best_val_mae_elo_epoch'] = epoch + 1
        
        # Log to file
        if self.log_file:
            self.log_file.write(
                f"Epoch {epoch+1:3d} | Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"Val RMSE (Elo): {val_rmse_elo:.1f}\n"
            )
            self.log_file.flush()
    
    def on_train_end(self, logs=None):
        import json
        import pandas as pd
        
        summary = {
            'total_epochs': len(self.history['epoch']),
            'total_training_time_seconds': sum(self.history['time_seconds']),
            'total_training_time_minutes': sum(self.history['time_seconds']) / 60,
            'avg_epoch_time_seconds': np.mean(self.history['time_seconds']) if self.history['time_seconds'] else 0,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'final_val_rmse_elo': self.history['val_rmse_elo'][-1] if self.history['val_rmse_elo'] else None,
            'final_val_mae_elo': self.history['val_mae_elo'][-1] if self.history['val_mae_elo'] else None,
            **self.best_metrics,
        }
        
        output = {
            'history': self.history,
            'summary': summary,
            'best_metrics': self.best_metrics,
        }
        
        json_path = self.log_dir / 'training_history.json'
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Comprehensive metrics saved to {json_path}")
        
        df = pd.DataFrame(self.history)
        csv_path = self.log_dir / 'detailed_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Detailed CSV metrics saved to {csv_path}")
    
    def get_history(self):
        return {
            'history': self.history.copy(),
            'summary': {
                'total_epochs': len(self.history['epoch']),
                'total_training_time_seconds': sum(self.history['time_seconds']),
                **self.best_metrics,
            },
            'best_metrics': self.best_metrics.copy(),
        }


def create_epoch_logger(log_every=1, log_file=None):
    """Create a callback to log epoch metrics."""
    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % log_every == 0:
                val_loss = logs.get('val_loss', 0)
                val_rmse_elo = np.sqrt(val_loss) * 2000 if val_loss else 0
                print(f"Epoch {epoch+1:3d} | Loss: {logs.get('loss', 0):.6f} | "
                      f"Val Loss: {val_loss:.6f} | Est. RMSE (Elo): {val_rmse_elo:.1f}")
    return EpochLogger()


def main(args=None):
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train TensorFlow LSTM for Elo prediction")
    
    # Model hyperparameters
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-hidden-dim", type=int, default=256)
    parser.add_argument("--num-lstm-layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--shuffle-buffer", type=int, default=5000)
    
    # Data and paths
    parser.add_argument("--parquet-dir", type=str, default="data/parquet")
    parser.add_argument("--model-prefix", type=str, default="lstm_elo")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    
    # System
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--skip-gpu-init", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    
    args = parser.parse_args(args)
    
    # Configuration
    EMBEDDING_DIM = args.embedding_dim
    LSTM_HIDDEN_DIM = args.lstm_hidden_dim
    NUM_LSTM_LAYERS = args.num_lstm_layers
    BIDIRECTIONAL = args.bidirectional and not args.no_bidirectional
    FEEDFORWARD_DIM = args.feedforward_dim
    DROPOUT = args.dropout
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    SHUFFLE_BUFFER = args.shuffle_buffer
    PARQUET_DIR = Path(args.parquet_dir)
    MODEL_PREFIX = args.model_prefix
    MAX_SEQUENCE_LENGTH = 200
    
    # GPU setup
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
        print(f"Using {num_gpus} GPU(s) (from --num-gpus override)")
    else:
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        print(f"Using {num_gpus} GPU(s) (auto-detected)")
    
    # Print configuration
    print("=" * 60)
    print("LSTM TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Embedding dim: {EMBEDDING_DIM}")
    print(f"LSTM hidden dim: {LSTM_HIDDEN_DIM}")
    print(f"LSTM layers: {NUM_LSTM_LAYERS}")
    print(f"Bidirectional: {BIDIRECTIONAL}")
    print(f"Feedforward dim: {FEEDFORWARD_DIM}")
    print(f"Dropout: {DROPOUT}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR:.2e}")
    print(f"Weight decay: {WEIGHT_DECAY:.2e}")
    print("=" * 60)
    
    # Create log directory
    LOGS_DIR = Path("logs")
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = LOGS_DIR / f"{MODEL_PREFIX}_{timestamp}"
    RUN_DIR.mkdir(exist_ok=True)
    
    log_file = open(RUN_DIR / "training.log", "w")
    
    def log_print(msg=""):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    
    log_print(f"\nTraining run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Log directory: {RUN_DIR}")
    log_print(f"Model prefix: {MODEL_PREFIX}")
    log_print("=" * 60)
    print()
    
    # Load data info
    train_pq = pq.ParquetFile(PARQUET_DIR / "train_games.parquet")
    val_pq = pq.ParquetFile(PARQUET_DIR / "val_games.parquet")
    total_train_rows = train_pq.metadata.num_rows
    total_val_rows = val_pq.metadata.num_rows
    
    log_print(f"Train: {total_train_rows:,} samples")
    log_print(f"Val: {total_val_rows:,} samples")
    
    # Create datasets
    train_dataset = create_streaming_dataset(
        PARQUET_DIR / "train_games.parquet",
        batch_size=BATCH_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        shuffle_buffer=SHUFFLE_BUFFER,
    ).repeat()
    
    val_dataset = create_streaming_dataset(
        PARQUET_DIR / "val_games.parquet",
        batch_size=BATCH_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        shuffle_buffer=1000,
    ).repeat()
    
    log_print("✓ Datasets created")
    print()
    
    # Multi-GPU strategy
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with MirroredStrategy")
        strategy = tf.distribute.MirroredStrategy()
        print("✓ MirroredStrategy initialized")
    else:
        strategy = tf.distribute.get_strategy()
        print("Using single GPU or CPU")
    print()
    
    # Build model within strategy scope
    with strategy.scope():
        model = build_lstm_model(
            embedding_dim=EMBEDDING_DIM,
            lstm_hidden_dim=LSTM_HIDDEN_DIM,
            num_lstm_layers=NUM_LSTM_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT,
            feedforward_dim=FEEDFORWARD_DIM,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            verbose=False,
        )
        
        # Print model architecture
        print("\nModel Architecture:")
        print_model_summary(model, MAX_SEQUENCE_LENGTH)
        print()
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=LR,
                weight_decay=WEIGHT_DECAY,
                clipnorm=1.0
            ),
            loss='mse',
            metrics=['mae']
        )
        log_print(f"✓ Model compiled with learning rate {LR:.2e}")
    
    # Comprehensive metrics callback
    metrics_callback = ComprehensiveMetricsCallback(log_dir=RUN_DIR, log_file=log_file)
    
    # Callbacks
    callbacks = [
        create_epoch_logger(log_every=1, log_file=log_file),
        metrics_callback,
        tf.keras.callbacks.TensorBoard(
            log_dir=str(RUN_DIR / "tensorboard"),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(RUN_DIR / "metrics.csv"),
            separator=',',
            append=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_PREFIX}_best.keras",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_PREFIX}_last.keras",
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    # Calculate steps
    effective_batch_size = BATCH_SIZE * max(1, num_gpus)
    estimated_train_samples = total_train_rows
    estimated_val_samples = total_val_rows
    
    if args.max_train_samples and args.max_train_samples < estimated_train_samples:
        log_print(f"Limiting training samples: {estimated_train_samples:,} -> {args.max_train_samples:,}")
        estimated_train_samples = args.max_train_samples
    
    if args.max_val_samples and args.max_val_samples < estimated_val_samples:
        log_print(f"Limiting validation samples: {estimated_val_samples:,} -> {args.max_val_samples:,}")
        estimated_val_samples = args.max_val_samples
    
    steps_per_epoch = max(1, estimated_train_samples // effective_batch_size)
    validation_steps = max(1, estimated_val_samples // effective_batch_size)
    
    log_print(f"Steps per epoch: {steps_per_epoch} (from {estimated_train_samples:,} samples)")
    log_print(f"Validation steps: {validation_steps} (from {estimated_val_samples:,} samples)")
    print()
    
    # Train
    log_print(f"Starting training ({EPOCHS} epochs, batch={BATCH_SIZE} per GPU, effective={effective_batch_size})")
    print()
    
    start_train = time.perf_counter()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callbacks,
    )
    
    train_secs = time.perf_counter() - start_train
    print(f"\n✓ Training completed in {train_secs/60:.2f} min ({train_secs/3600:.2f} hours)")
    
    # Final metrics
    final_loss = history.history["loss"][-1]
    final_val = history.history.get("val_loss", [None])[-1]
    print(f"Final metrics -> loss={final_loss:.6f}, val_loss={final_val:.6f}")
    
    if final_val is not None:
        rmse_normalized = np.sqrt(final_val)
        rmse_elo = rmse_normalized * 2000
        print(f"Final validation -> RMSE (normalized): {rmse_normalized:.6f}, Est. RMSE (Elo): {rmse_elo:.1f}")
    
    if "val_loss" in history.history:
        best_val_idx = int(np.argmin(history.history["val_loss"]))
        best_val = history.history["val_loss"][best_val_idx]
        best_loss = history.history["loss"][best_val_idx]
        best_rmse_elo = np.sqrt(best_val) * 2000
        print(f"Best epoch (by val_loss): {best_val_idx + 1}/{len(history.history['loss'])} "
              f"loss={best_loss:.6f}, val_loss={best_val:.6f}")
        print(f"Best validation -> RMSE (Elo): {best_rmse_elo:.1f}")
    
    # Save final model
    final_keras_path = f"{MODEL_PREFIX}.keras"
    model.save(final_keras_path)
    print(f"✓ Saved model to {final_keras_path}")
    
    # Close log file
    log_file.close()
    print(f"\n{'='*60}")
    print(f"Training completed. Logs saved to: {RUN_DIR}")
    print(f"  - TensorBoard logs: {RUN_DIR / 'tensorboard'}")
    print(f"  - CSV metrics: {RUN_DIR / 'metrics.csv'}")
    print(f"  - Training log: {RUN_DIR / 'training.log'}")
    print(f"\nTo view TensorBoard, run:")
    print(f"  tensorboard --logdir {RUN_DIR / 'tensorboard'}")
    print(f"{'='*60}")
    
    return {
        'history': history.history,
        'log_dir': str(RUN_DIR),
        'metrics_csv': str(RUN_DIR / 'metrics.csv'),
        'model_path': final_keras_path,
        'epochs_trained': len(history.history['loss']),
    }


def train_with_config(**config):
    """
    Train model with a configuration dictionary instead of command-line arguments.
    
    Example:
        train_with_config(
            parquet_dir="data/parquet",
            epochs=20,
            batch_size=32,
            embedding_dim=128,
            lstm_hidden_dim=256,
            num_lstm_layers=2,
            dropout=0.1,
            lr=1e-4,
            model_prefix="lstm_elo",
            max_train_samples=50000,
            max_val_samples=10000,
            num_gpus=4,
            skip_gpu_init=True,
        )
    """
    args_list = []
    for key, value in config.items():
        arg_name = key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                args_list.append(f"--{arg_name}")
        else:
            args_list.extend([f"--{arg_name}", str(value)])
    
    return main(args_list)


if __name__ == "__main__":
    main()
