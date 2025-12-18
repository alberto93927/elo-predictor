
"""
Optimized training script for Transformer Elo prediction model.
Matches the structure and optimizations from run_full_train_optimized.py.
"""


import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import os
import sys
import argparse
from datetime import datetime
import logging

# Suppress TensorFlow warnings about dataset end-of-sequence (harmless in multi-GPU training)
# Set to '2' to suppress INFO and WARNING (keeps ERROR visible for debugging)
# Note: Some INFO messages from C++ code may still appear, but this helps
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress INFO and WARNING
# Disable oneDNN optimizations that cause warnings
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Add parent directory to path for imports BEFORE importing src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IMMEDIATE GPU CHECK - right after TensorFlow import, before any other imports
# This helps diagnose if later imports are interfering with GPU detection
# Skip if --skip-gpu-init or --num-gpus is in command line args (notebook use case)
_skip_gpu_check = '--skip-gpu-init' in sys.argv or '--num-gpus' in sys.argv
_gpus_immediate = tf.config.list_physical_devices('GPU')
# Check for TensorRT availability (optional, for inference optimization)
TENSORRT_AVAILABLE = False
try:
    # Try to import TensorRT Python bindings
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # TensorRT not installed - that's okay, it's optional
    pass
if not _skip_gpu_check:
    if len(_gpus_immediate) == 0:
        print("WARNING: No GPUs detected immediately after TensorFlow import!")
        print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        print(
            f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    else:
        print(
            f"✓ Detected {len(_gpus_immediate)} GPU(s) immediately after TensorFlow import")
# Import src modules AFTER TensorFlow and sys.path setup
from src.models.transformer_tf import build_transformer_model, TransformerEncoder
from src.models.lstm_tf import build_lstm_model, LSTMEloPredictor
from src.data.data_utils_tf import (
    load_pickle_dataset,
    convert_games_to_parquet,
    create_streaming_dataset,
    create_dataset_from_pickle,
    preload_parquet_to_memory,
    create_preloaded_dataset,
)
from src.data.encoder import FENEncoder


@tf.keras.utils.register_keras_serializable(package="elo")
def safe_mse_loss(y_true, y_pred):
    """MSE loss with NaN protection."""
    # Clip predictions to [0, 1] range
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    # Replace any NaN in predictions with 0.5 (middle of range)
    y_pred = tf.where(tf.math.is_nan(y_pred),
                      0.5 * tf.ones_like(y_pred), y_pred)
    # Compute MSE manually: mean squared difference
    squared_diff = tf.square(y_true - y_pred)
    loss = tf.reduce_mean(squared_diff)
    # Replace any NaN in loss with a large value (will trigger gradient clipping)
    loss = tf.where(tf.math.is_nan(loss), 1e6 * tf.ones_like(loss), loss)
    return loss


@tf.keras.utils.register_keras_serializable(package="elo")
def safe_mae_metric(y_true, y_pred):
    """MAE metric with NaN protection."""
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    y_pred = tf.where(tf.math.is_nan(y_pred),
                      0.5 * tf.ones_like(y_pred), y_pred)
    # Compute MAE manually: mean absolute difference
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae = tf.where(tf.math.is_nan(mae), tf.zeros_like(mae), mae)
    return mae


class ComprehensiveMetricsCallback(tf.keras.callbacks.Callback):
    """
    Comprehensive metrics tracking for rich visualizations.
    
    Tracks per-epoch and per-batch statistics including:
    - Loss and MAE in both normalized and Elo scale
    - Learning rate schedule
    - Per-batch loss statistics (min, max, std)
    - Training time per epoch
    - Best metrics and when they occurred
    """
    
    ELO_RANGE = 2000  # 2800 - 800 = 2000 Elo range for denormalization
    
    def __init__(self, log_dir, log_file=None):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        
        # Per-epoch metrics
        self.history = {
            'epoch': [],
            'lr': [],
            'time_seconds': [],
            # Training metrics (normalized)
            'train_loss': [],
            'train_mae': [],
            # Training metrics (Elo scale)
            'train_rmse_elo': [],
            'train_mae_elo': [],
            # Validation metrics (normalized)
            'val_loss': [],
            'val_mae': [],
            # Validation metrics (Elo scale)
            'val_rmse_elo': [],
            'val_mae_elo': [],
            # Per-batch statistics for training loss
            'batch_loss_min': [],
            'batch_loss_max': [],
            'batch_loss_std': [],
            'batch_count': [],
        }
        
        # Best metrics tracking
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_loss_epoch': 0,
            'best_val_mae_elo': float('inf'),
            'best_val_mae_elo_epoch': 0,
        }
        
        # Per-batch tracking (reset each epoch)
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
        
        # Get current learning rate
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except:
            lr = 0.0
        
        # Extract metrics
        train_loss = logs.get('loss', 0)
        train_mae = logs.get('safe_mae_metric', logs.get('mae', 0))
        val_loss = logs.get('val_loss', 0)
        val_mae = logs.get('val_safe_mae_metric', logs.get('val_mae', 0))
        
        # Convert to Elo scale
        train_rmse_elo = np.sqrt(train_loss) * self.ELO_RANGE if train_loss else 0
        train_mae_elo = train_mae * self.ELO_RANGE if train_mae else 0
        val_rmse_elo = np.sqrt(val_loss) * self.ELO_RANGE if val_loss else 0
        val_mae_elo = val_mae * self.ELO_RANGE if val_mae else 0
        
        # Batch statistics
        batch_losses = np.array(self._batch_losses) if self._batch_losses else np.array([0])
        
        # Store metrics
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
        
        # Update best metrics
        if val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = float(val_loss)
            self.best_metrics['best_val_loss_epoch'] = epoch + 1
        
        if val_mae_elo < self.best_metrics['best_val_mae_elo']:
            self.best_metrics['best_val_mae_elo'] = float(val_mae_elo)
            self.best_metrics['best_val_mae_elo_epoch'] = epoch + 1
    
    def on_train_end(self, logs=None):
        """Save comprehensive metrics to JSON file."""
        import json
        
        # Compute summary statistics
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
        
        # Save to JSON
        json_path = self.log_dir / 'training_history.json'
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Comprehensive metrics saved to {json_path}")
        
        # Also save as CSV for easy loading in pandas
        import pandas as pd
        df = pd.DataFrame(self.history)
        csv_path = self.log_dir / 'detailed_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Detailed CSV metrics saved to {csv_path}")
    
    def get_history(self):
        """Return the history dict for programmatic access."""
        return {
            'history': self.history.copy(),
            'summary': {
                'total_epochs': len(self.history['epoch']),
                'total_training_time_seconds': sum(self.history['time_seconds']),
                **self.best_metrics,
            },
            'best_metrics': self.best_metrics.copy(),
        }


# Ensure custom classes and functions are registered in TensorFlow Keras global registry
# This is critical for TensorFlow Keras to find them during model loading
# The decorators above register them, but we also explicitly add them to the registry
# to ensure they're available when loading checkpoints
try:
    # Register with package prefix (format: "package>ClassName")
    custom_objects_registry = tf.keras.utils.get_custom_objects()
    custom_objects_registry['elo>TransformerEncoder'] = TransformerEncoder
    custom_objects_registry['elo>LSTMEloPredictor'] = LSTMEloPredictor
    custom_objects_registry['elo>safe_mse_loss'] = safe_mse_loss
    custom_objects_registry['elo>safe_mae_metric'] = safe_mae_metric
except (AttributeError, KeyError):
    pass  # Will rely on custom_objects parameter instead

def extract_hyperparameters_from_checkpoint(checkpoint_path: Path):
    """
    Extract hyperparameters from a checkpoint file by reading its config.
    Returns a dict with hyperparameters or None if extraction fails.
    """
    try:
        import zipfile
        import json
        
        # .keras files are zip archives
        with zipfile.ZipFile(checkpoint_path, 'r') as z:
            # Read the config.json file
            config_data = z.read('config.json')
            config = json.loads(config_data)
            
            hyperparams = {}
            
            # Check if this is a TransformerEncoder at the top level
            # Config structure: {'module': ..., 'class_name': 'TransformerEncoder', 'config': {...}, ...}
            class_name = config.get('class_name', '')
            registered_name = config.get('registered_name', '')
            
            if 'TransformerEncoder' in class_name or 'TransformerEncoder' in registered_name:
                # Found it at top level! Extract hyperparameters from 'config' key
                if 'config' in config:
                    cfg = config['config']
                    print(f"Debug: Found TransformerEncoder config keys: {list(cfg.keys())}")
                    
                    # Extract all hyperparameters we care about
                    if 'embedding_dim' in cfg:
                        hyperparams['embedding_dim'] = cfg['embedding_dim']
                    if 'num_layers' in cfg:
                        hyperparams['num_layers'] = cfg['num_layers']
                    if 'num_heads' in cfg:
                        hyperparams['num_heads'] = cfg['num_heads']
                    if 'feedforward_dim' in cfg:
                        hyperparams['feedforward_dim'] = cfg['feedforward_dim']
                    if 'dropout' in cfg:
                        hyperparams['dropout'] = cfg['dropout']
                    if 'max_sequence_length' in cfg:
                        hyperparams['max_sequence_length'] = cfg['max_sequence_length']
            
            # Also check for LSTMEloPredictor
            elif 'LSTMEloPredictor' in class_name or 'LSTMEloPredictor' in registered_name:
                if 'config' in config:
                    cfg = config['config']
                    print(f"Debug: Found LSTMEloPredictor config keys: {list(cfg.keys())}")
                    
                    if 'embedding_dim' in cfg:
                        hyperparams['embedding_dim'] = cfg['embedding_dim']
                    if 'num_lstm_layers' in cfg:
                        hyperparams['num_layers'] = cfg['num_lstm_layers']
                    if 'lstm_hidden_dim' in cfg:
                        hyperparams['lstm_hidden_dim'] = cfg['lstm_hidden_dim']
                    if 'bidirectional' in cfg:
                        hyperparams['bidirectional'] = cfg['bidirectional']
                    if 'feedforward_dim' in cfg:
                        hyperparams['feedforward_dim'] = cfg['feedforward_dim']
                    if 'dropout' in cfg:
                        hyperparams['dropout'] = cfg['dropout']
                    if 'max_sequence_length' in cfg:
                        hyperparams['max_sequence_length'] = cfg['max_sequence_length']
            else:
                # Debug: print config structure if extraction failed
                print(f"Debug: class_name='{class_name}', registered_name='{registered_name}'")
                print(f"Debug: Top-level keys: {list(config.keys())}")
                if 'config' in config:
                    print(f"Debug: config keys: {list(config['config'].keys())}")
            
            if hyperparams:
                return hyperparams
                
    except Exception as e:
        print(f"Could not extract hyperparameters from checkpoint: {e}")
        import traceback
        traceback.print_exc()
    
    return None




def create_epoch_logger(log_every: int = 1, log_file=None):
    """Create a callback that logs metrics every N epochs."""
    class EpochLogger(tf.keras.callbacks.Callback):
        def __init__(self, log_every=1, log_file=None):
            super().__init__()
            self.log_every = log_every
            self.log_file = log_file

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}

            if (epoch + 1) % self.log_every == 0:
                loss = logs.get('loss', 'N/A')
                val_loss = logs.get('val_loss', 'N/A')

                # Convert normalized loss to Elo estimate if possible
                if isinstance(val_loss, (int, float)) and val_loss != 'N/A':
                    # Rough estimate: RMSE in normalized space -> Elo
                    # Assuming normalized [0,1] maps to [800, 2800] Elo
                    rmse_normalized = np.sqrt(val_loss)
                    rmse_elo = rmse_normalized * 2000  # 2800 - 800 = 2000
                    message = f"Epoch {epoch + 1:3d} | Loss: {loss:.6f} | Val Loss: {val_loss:.6f} | Est. RMSE (Elo): {rmse_elo:.1f}"
                    print(message)
                    if self.log_file:
                        print(message, file=self.log_file)
                        self.log_file.flush()
                else:
                    message = f"Epoch {epoch + 1:3d} | Loss: {loss:.6f} | Val Loss: {val_loss:.6f}"
                    print(message)
                    if self.log_file:
                        print(message, file=self.log_file)
                        self.log_file.flush()

    return EpochLogger(log_every=log_every, log_file=log_file)


def main(args=None):
    # ============================================================================
    # OPTIMIZED TRAINING PARAMETERS
    # ============================================================================
    # Parse command-line arguments if provided
    parser = argparse.ArgumentParser(
        description="Train optimized Transformer or LSTM model for Elo prediction"
    )
    parser.add_argument("--model-type", type=str, default="transformer",
                        choices=["transformer", "lstm"],
                        help="Model architecture: 'transformer' or 'lstm' (default: transformer)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per GPU (default: 32)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Early stopping patience - stop if val_loss doesn't improve for N epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5, lowered for better stability)")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dimension (default: 256, increased from 128)")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Number of transformer layers or LSTM layers (default: 6)")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads (transformer only, default: 8)")
    parser.add_argument("--lstm-hidden-dim", type=int, default=256,
                        help="LSTM hidden dimension (LSTM only, default: 256)")
    parser.add_argument("--no-bidirectional", action="store_false", dest="bidirectional",
                        help="Disable bidirectional LSTM (LSTM only, default: True)")
    # Set default for bidirectional (True by default)
    parser.set_defaults(bidirectional=True)
    parser.add_argument("--feedforward-dim", type=int, default=1024,
                        help="Feedforward network dimension (default: 1024, increased from 512)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (default: 0.2, increased from 0.1 for better regularization)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay for AdamW optimizer (default: 5e-4)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of warmup epochs for learning rate schedule (default: 5)")
    parser.add_argument("--use-cosine-schedule", action="store_true",
                        help="Use cosine annealing schedule instead of ReduceLROnPlateau")
    parser.add_argument("--max-seq-len", type=int, default=200,
                        help="Maximum sequence length (default: 200)")
    parser.add_argument("--shuffle-buffer", type=int, default=50000,
                        help="Shuffle buffer size (default: 50000, larger = better randomness)")
    parser.add_argument("--preload", action="store_true", default=False,
                        help="Pre-load data into RAM (use only with small datasets)")
    parser.add_argument("--no-preload", dest="preload", action="store_false",
                        help="Use streaming mode (default, memory-safe)")
    parser.add_argument("--max-samples", type=int, default=250000,
                        help="Max samples to preload if --preload is used (default: 250K)")
    parser.add_argument("--model-prefix", type=str, default=None,
                        help="Model prefix for saved models (default: {model_type}_elo_optimized)")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Data directory (default: data/processed)")
    parser.add_argument("--parquet-dir", type=str, default="data/parquet",
                        help="Parquet data directory (default: data/parquet)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Maximum training samples per epoch (for quick testing with large datasets)")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Maximum validation samples per epoch (for quick testing with large datasets)")
    parser.add_argument("--use-mixed-precision", action="store_true",
                        help="Enable mixed precision training (FP16) for ~2x speedup. Default: disabled for stability.")
    parser.add_argument("--no-parquet", action="store_true",
                        help="Force use of pickle files instead of Parquet")
    parser.add_argument("--use-black-elo", action="store_true",
                        help="Use black player's Elo instead of white's")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Override GPU count (use when GPUs already detected, e.g., in notebooks)")
    parser.add_argument("--skip-gpu-init", action="store_true",
                        help="Skip GPU initialization (use when TensorFlow already configured)")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    DATA_DIR = Path(args.data_dir)
    PARQUET_DIR = Path(args.parquet_dir)

    # Batch size per GPU - increased for better GPU utilization
    # With 4 GPUs, effective batch size = BATCH_SIZE * 4
    BATCH_SIZE = args.batch_size

    # Model type - define early so it can be used for MODEL_PREFIX
    MODEL_TYPE = args.model_type.lower()
    
    EPOCHS = args.epochs
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    WARMUP_EPOCHS = args.warmup_epochs
    USE_COSINE_SCHEDULE = args.use_cosine_schedule
    WEIGHT_DECAY = args.weight_decay
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    # Set default model prefix based on model type if not provided
    if args.model_prefix is None:
        MODEL_PREFIX = f"{MODEL_TYPE}_elo_optimized"
    else:
        MODEL_PREFIX = args.model_prefix

    # Optimized shuffle buffer
    SHUFFLE_BUFFER = args.shuffle_buffer

    # Mixed precision (FP16) - disabled by default for stability
    # Can cause numerical instability and overflow warnings
    # Enable with --use-mixed-precision for ~2x speedup on modern GPUs
    USE_MIXED_PRECISION = args.use_mixed_precision

    # Learning rate - may need adjustment with larger batch size
    BASE_LR = args.lr
    # Scale LR with batch size if needed
    SCALED_LR = BASE_LR

    # Data loading parallelism
    NUM_PARALLEL_CALLS = tf.data.AUTOTUNE

    # Model hyperparameters
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    LSTM_HIDDEN_DIM = args.lstm_hidden_dim
    BIDIRECTIONAL = args.bidirectional
    FEEDFORWARD_DIM = args.feedforward_dim
    DROPOUT = args.dropout
    MAX_SEQUENCE_LENGTH = args.max_seq_len

    # Use Parquet if available, otherwise fall back to pickle
    USE_PARQUET = not args.no_parquet
    USE_WHITE_ELO = not args.use_black_elo

    # GPU detection - use override if provided (e.g., from notebook)
    if args.num_gpus is not None:
        # Use the provided GPU count (already detected in notebook)
        num_gpus = args.num_gpus
        gpus = tf.config.list_physical_devices('GPU')  # Still get list for display
        print(f"Using {num_gpus} GPU(s) (from --num-gpus override)")
    else:
        # Detect GPUs
        gpus = tf.config.list_physical_devices('GPU')
        num_gpus = len(gpus)

    print("=" * 60)
    print("OPTIMIZED TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model type: {MODEL_TYPE.upper()}")
    print(f"Batch size per GPU: {BATCH_SIZE}")
    print(f"Number of GPUs: {num_gpus}")
    if num_gpus > 0:
        print(f"Effective batch size: {BATCH_SIZE * num_gpus}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print(f"Effective batch size: {BATCH_SIZE} (CPU only)")
        print("  WARNING: No GPUs detected!")
        print(
            f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        print(
            f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Shuffle buffer: {SHUFFLE_BUFFER:,}")
    print(f"Mixed precision: {USE_MIXED_PRECISION}")
    print(f"Learning rate: {SCALED_LR:.2e}")
    print(f"Weight decay: {WEIGHT_DECAY:.2e}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
    if USE_COSINE_SCHEDULE:
        print(
            f"LR schedule: Cosine annealing with {WARMUP_EPOCHS} epoch warmup")
    else:
        print(f"LR schedule: ReduceLROnPlateau")
    if MODEL_TYPE == "transformer":
        print(
            f"Model: Transformer (embed_dim={EMBEDDING_DIM}, layers={NUM_LAYERS}, heads={NUM_HEADS})")
    else:
        print(
            f"Model: LSTM (embed_dim={EMBEDDING_DIM}, layers={NUM_LAYERS}, hidden_dim={LSTM_HIDDEN_DIM}, bidirectional={BIDIRECTIONAL})")
    if TENSORRT_AVAILABLE:
        print(f"TensorRT: Available (will optimize inference)")
    else:
        print(f"TensorRT: Not available (optional, for inference optimization)")
    print("=" * 60)
    print()

    # Enable mixed precision if requested
    # WARNING: Mixed precision can cause:
    #   - Numerical instability (NaN/Inf)
    #   - Overflow warnings during casting
    #   - Slightly different results vs FP32
    # Benefits: ~2x faster training, ~50% less memory
    # Default: disabled for stability. Enable with --use-mixed-precision
    if USE_MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✓ Mixed precision (FP16) enabled")
        print("  Note: Output layer will use float32 for numerical stability")
        print("  Warning: May cause overflow warnings - these are usually harmless")
        print()
    else:
        print("✓ Mixed precision disabled (using FP32)")
        print("  This is more stable but slower and uses more memory")
        print()

    # Load or convert data
    encoder = FENEncoder(MAX_SEQUENCE_LENGTH)

    # Check for Parquet files
    train_parquet = PARQUET_DIR / "train_games.parquet"
    val_parquet = PARQUET_DIR / "val_games.parquet"
    test_parquet = PARQUET_DIR / "test_games.parquet"

    if USE_PARQUET and train_parquet.exists() and val_parquet.exists():
        print("Using Parquet data files...")

        # Get Parquet metadata
        pf_train = pq.ParquetFile(train_parquet)
        pf_val = pq.ParquetFile(val_parquet)

        num_train_groups = pf_train.num_row_groups
        num_val_groups = pf_val.num_row_groups

        total_train_rows = sum(
            pf_train.metadata.row_group(i).num_rows
            for i in range(num_train_groups)
        )
        total_val_rows = sum(
            pf_val.metadata.row_group(i).num_rows
            for i in range(num_val_groups)
        )

        print(
            f"Train: {num_train_groups} row groups, ~{total_train_rows:,} games")
        print(f"Val: {num_val_groups} row groups, ~{total_val_rows:,} games")

        # Get row group indices
        train_row_groups = list(range(num_train_groups))
        val_row_groups = list(range(num_val_groups))

        # Choose between preloading (faster) and streaming (lower memory)
        PRELOAD_DATA = args.preload
        
        if PRELOAD_DATA:
            MAX_SAMPLES = args.max_samples
            if MAX_SAMPLES > 0:
                est_gb = MAX_SAMPLES * 0.00063  # ~0.63 MB per sample
                print(f"Pre-loading up to {MAX_SAMPLES:,} samples (~{est_gb:.0f} GB)...")
            else:
                print("Pre-loading ALL data into RAM...")
            
            # Pre-load training data
            print("  Train: ", end="", flush=True)
            train_sequences, train_lengths, train_targets, train_opp_features = \
                preload_parquet_to_memory(train_parquet, train_row_groups, encoder, max_samples=MAX_SAMPLES)
            
            # Pre-load validation data (use proportional limit: ~12.5% of train)
            val_max = MAX_SAMPLES // 8 if MAX_SAMPLES > 0 else 0
            print("  Val:   ", end="", flush=True)
            val_sequences, val_lengths, val_targets, val_opp_features = \
                preload_parquet_to_memory(val_parquet, val_row_groups, encoder, max_samples=val_max)
            
            train_dataset = create_preloaded_dataset(
                train_sequences, train_lengths, train_targets, train_opp_features,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_remainder=True,
            )
            val_dataset = create_preloaded_dataset(
                val_sequences, val_lengths, val_targets, val_opp_features,
                batch_size=BATCH_SIZE,
                shuffle=False,
                drop_remainder=True,
            )
            
            # Update row counts for steps calculation
            total_train_rows = len(train_sequences)
            total_val_rows = len(val_sequences)
            print()
        else:
            # Streaming mode (original approach)
            print("Creating streaming datasets...")
            train_dataset = create_streaming_dataset(
                train_parquet,
                train_row_groups,
                BATCH_SIZE,
                shuffle_buffer=SHUFFLE_BUFFER,
                num_parallel_calls=NUM_PARALLEL_CALLS,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
            val_dataset = create_streaming_dataset(
                val_parquet,
                val_row_groups,
                BATCH_SIZE,
                shuffle_buffer=None,  # No shuffle for validation
                num_parallel_calls=NUM_PARALLEL_CALLS,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
            print("✓ Streaming datasets created")
            print()

    else:
        # Fall back to pickle files
        print("Parquet files not found, using pickle files...")
        print("(Consider converting to Parquet for better performance)")
        print()
        
        # Preloading not supported for pickle (already in-memory)
        PRELOAD_DATA = False

        # Load pickle data
        train_games, val_games, test_games = load_pickle_dataset(str(DATA_DIR))

        print(f"Train: {len(train_games):,} games")
        print(f"Val: {len(val_games):,} games")
        print(f"Test: {len(test_games):,} games")
        print()

        # Create datasets from pickle
        print("Creating datasets from pickle...")
        train_dataset = create_dataset_from_pickle(
            train_games,
            encoder,
            BATCH_SIZE,
            shuffle=True,
            use_white_elo=USE_WHITE_ELO,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        val_dataset = create_dataset_from_pickle(
            val_games,
            encoder,
            BATCH_SIZE,
            shuffle=False,
            use_white_elo=USE_WHITE_ELO,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        print("✓ Datasets created")
        print()

    # Create logs directory for TensorBoard, CSV, and text logs
    LOGS_DIR = Path("logs")
    LOGS_DIR.mkdir(exist_ok=True)

    # Create a timestamped run directory for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = LOGS_DIR / f"{MODEL_PREFIX}_{timestamp}"
    RUN_DIR.mkdir(exist_ok=True)

    # Set up text log file to capture all print statements
    log_file_path = RUN_DIR / "training.log"
    log_file = open(log_file_path, 'w')

    # Create a custom print function that writes to both console and file
    def log_print(*args, **kwargs):
        """Print to both console and log file."""
        message = ' '.join(str(arg) for arg in args)
        print(*args, **kwargs)
        print(message, file=log_file, **
              {k: v for k, v in kwargs.items() if k != 'end'})
        log_file.flush()  # Ensure immediate write

    # Log training start info
    log_print(
        f"\nTraining run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Log directory: {RUN_DIR}")
    log_print(f"Model prefix: {MODEL_PREFIX}")
    log_print("=" * 60)
    print()

    # Setup multi-GPU strategy (exactly like run_full_train_optimized.py)
    # Use num_gpus from earlier detection (may be overridden via --num-gpus)
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with MirroredStrategy")
        strategy = tf.distribute.MirroredStrategy()
        print(f"✓ MirroredStrategy initialized")
    else:
        strategy = tf.distribute.get_strategy()
        print(f"Using single GPU or CPU")
    print()

    # Build model within strategy scope
    with strategy.scope():
        if MODEL_TYPE == "transformer":
            print("Building Transformer model...")
            model = build_transformer_model(
                embedding_dim=EMBEDDING_DIM,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                feedforward_dim=FEEDFORWARD_DIM,
                dropout=DROPOUT,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                fast=False,
                verbose=False,  # We'll build and show summary here instead
            )
        else:  # LSTM
            print("Building LSTM model...")
            model = build_lstm_model(
                embedding_dim=EMBEDDING_DIM,
                lstm_hidden_dim=LSTM_HIDDEN_DIM,
                num_lstm_layers=NUM_LAYERS,
                bidirectional=BIDIRECTIONAL,
                dropout=DROPOUT,
                feedforward_dim=FEEDFORWARD_DIM,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                fast=False,
                verbose=False,  # We'll build and show summary here instead
            )
        
        # Build the model by calling it with sample data (proper way for subclassed models)
        # This ensures all layers are properly initialized and shapes are computed
        print("Building model with sample input...")
        import numpy as np
        sample_sequence = np.zeros((1, MAX_SEQUENCE_LENGTH, 13, 8, 8), dtype=np.float32)
        sample_length = np.array([MAX_SEQUENCE_LENGTH], dtype=np.int32)
        _ = model([sample_sequence, sample_length], training=False)
        # Print model architecture using custom summary (handles subclassed models properly)
        from src.models.transformer_tf import print_model_summary
        print("\nModel Architecture:")
        print_model_summary(model, MAX_SEQUENCE_LENGTH)
        print()
        
        # Compile model with gradient clipping to prevent NaN
        # Use clipnorm=1.0 to clip gradients by norm, preventing explosion
        # Weight decay is configurable via --weight-decay argument
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=SCALED_LR,
                weight_decay=WEIGHT_DECAY,  # Configurable, default 5e-4
                clipnorm=1.0  # Clip gradients to prevent explosion
            ),
            loss=safe_mse_loss,
            metrics=[safe_mae_metric]
        )
        print(f"✓ Model compiled with learning rate {SCALED_LR:.2e}")

    # For mixed precision, ensure output layer uses float32
    if USE_MIXED_PRECISION:
        # Output layer should already be float32, but verify
        output_layer = model.get_layer('elo_prediction')
        if output_layer.dtype_policy.name != 'float32':
            print("Warning: Output layer should use float32 for numerical stability")

    # Comprehensive metrics callback for rich visualizations
    metrics_callback = ComprehensiveMetricsCallback(log_dir=RUN_DIR, log_file=log_file)
    
    # Callbacks
    callbacks = [
        create_epoch_logger(log_every=1, log_file=log_file),
        metrics_callback,  # Track detailed metrics for visualization
        # TensorBoard logging for visualization
        tf.keras.callbacks.TensorBoard(
            log_dir=str(RUN_DIR / "tensorboard"),
            histogram_freq=1,  # Log weight histograms every epoch
            write_graph=True,  # Write the model graph
            write_images=False,  # Don't write images (saves space)
            update_freq='epoch',  # Update after each epoch
            profile_batch=0,  # Disable profiling (can slow training)
        ),
        # CSV logger for easy plotting
        tf.keras.callbacks.CSVLogger(
            filename=str(RUN_DIR / "metrics.csv"),
            separator=',',
            append=False,  # Overwrite if exists
        ),
        # Best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_PREFIX}_best.keras",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        # Last epoch checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_PREFIX}_last.keras",
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False,
        ),
        # Early stopping to prevent overfitting
        # Stops training if validation loss doesn't improve for N epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,  # Restore weights from best epoch
            verbose=1,
            mode='min',  # Stop when val_loss stops decreasing
        ),
    ]

    # Learning rate schedule: warmup + cosine decay or ReduceLROnPlateau
    if USE_COSINE_SCHEDULE:
        # Cosine annealing with warmup
        def lr_schedule(epoch, lr):
            if epoch < WARMUP_EPOCHS:
                # Linear warmup
                return SCALED_LR * (epoch + 1) / WARMUP_EPOCHS
            else:
                # Cosine annealing after warmup
                progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
                return SCALED_LR * 0.5 * (1 + np.cos(np.pi * progress))

        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lr_schedule,
                verbose=1
            )
        )
    else:
        # Learning rate reduction on plateau (default)
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            )
        )

    final_loss = None
    final_val = None
    best_val = None
    effective_batch = BATCH_SIZE * max(1, num_gpus)
    print(
        f"Starting training ({EPOCHS} epochs, "
        f"batch={BATCH_SIZE} per GPU, effective={effective_batch}, "
        f"val_split={VAL_SPLIT})"
    )
    mode = "preloaded (full shuffle)" if PRELOAD_DATA else f"streaming (shuffle={SHUFFLE_BUFFER:,})"
    print(f"Data mode: {mode}")
    print()

    start_train = time.perf_counter()

    # Suppress warnings about end-of-sequence and rendezvous cancellations (normal in multi-GPU training)
    # These are harmless INFO messages from TensorFlow's multi-GPU coordination
    import logging
    import sys

    # Suppress TensorFlow INFO and WARNING messages via logging
    tf_logger = logging.getLogger('tensorflow')
    old_level = tf_logger.level
    tf_logger.setLevel(logging.ERROR)  # Only show ERROR and above

    # Filter out specific rendezvous messages by redirecting stderr temporarily
    # Note: These messages come from C++ code and bypass Python logging
    class RendezvousFilter:
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr
            self.buffer = []

        def write(self, text):
            # Filter out "Local rendezvous" messages
            if 'Local rendezvous' not in text and 'rendezvous' not in text.lower():
                self.original_stderr.write(text)

        def flush(self):
            self.original_stderr.flush()

    # Only filter if we're in multi-GPU mode (to avoid affecting single-GPU)
    original_stderr = None
    if num_gpus > 1:
        original_stderr = sys.stderr
        filtered_stderr = RendezvousFilter(original_stderr)
        sys.stderr = filtered_stderr

    # Calculate steps per epoch for repeating datasets
    # When using .repeat(), we need to tell Keras how many steps = one epoch
    # This prevents the "input ran out of data" warning
    effective_batch_size = BATCH_SIZE * max(1, num_gpus)
    
    if USE_PARQUET:
        # Use actual row counts from Parquet metadata
        estimated_train_samples = total_train_rows
        estimated_val_samples = total_val_rows
    else:
        # For pickle, we have exact counts
        estimated_train_samples = len(train_games)
        estimated_val_samples = len(val_games)
    
    # Apply sample limits if specified (for quick testing with large datasets)
    MAX_TRAIN_SAMPLES = args.max_train_samples
    MAX_VAL_SAMPLES = args.max_val_samples
    
    if MAX_TRAIN_SAMPLES and MAX_TRAIN_SAMPLES < estimated_train_samples:
        log_print(f"Limiting training samples: {estimated_train_samples:,} -> {MAX_TRAIN_SAMPLES:,}")
        estimated_train_samples = MAX_TRAIN_SAMPLES
    
    if MAX_VAL_SAMPLES and MAX_VAL_SAMPLES < estimated_val_samples:
        log_print(f"Limiting validation samples: {estimated_val_samples:,} -> {MAX_VAL_SAMPLES:,}")
        estimated_val_samples = MAX_VAL_SAMPLES
    
    steps_per_epoch = max(1, estimated_train_samples // effective_batch_size)
    validation_steps = max(1, estimated_val_samples // effective_batch_size)
    
    log_print(f"Steps per epoch: {steps_per_epoch} (from {estimated_train_samples:,} samples)")
    log_print(f"Validation steps: {validation_steps} (from {estimated_val_samples:,} samples)")
    
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1,
            callbacks=callbacks,
        )
    finally:
        # Restore logging level and stderr
        tf_logger.setLevel(old_level)
        if original_stderr is not None:
            sys.stderr = original_stderr
    train_secs = time.perf_counter() - start_train
    print(
        f"\n✓ Training completed in {train_secs/60:.2f} min ({train_secs/3600:.2f} hours)")

    final_loss = history.history["loss"][-1]
    final_val = history.history.get("val_loss", [None])[-1]
    print(
        f"Final metrics -> loss={final_loss:.6f}, val_loss={final_val:.6f}")

    # Convert loss to Elo estimates
    if final_val is not None:
        rmse_normalized = np.sqrt(final_val)
        rmse_elo = rmse_normalized * 2000  # Rough estimate
        print(
            f"Final validation -> RMSE (normalized): {rmse_normalized:.6f}, Est. RMSE (Elo): {rmse_elo:.1f}")

    if "val_loss" in history.history:
        best_val_idx = int(np.argmin(history.history["val_loss"]))
        best_val = history.history["val_loss"][best_val_idx]
        best_loss = history.history["loss"][best_val_idx]
        best_rmse_normalized = np.sqrt(best_val)
        best_rmse_elo = best_rmse_normalized * 2000
        print(
            f"Best epoch (by val_loss): {best_val_idx + 1}/{len(history.history['loss'])} "
            f"loss={best_loss:.6f}, val_loss={best_val:.6f}"
        )
        print(
            f"Best validation -> RMSE (normalized): {best_rmse_normalized:.6f}, Est. RMSE (Elo): {best_rmse_elo:.1f}")

    # Evaluate on validation set to provide an objective metric (useful for tuning)
    try:
        eval_results = model.evaluate(
            val_dataset,
            steps=validation_steps,
            verbose=0,
        )
        eval_val_loss = float(eval_results[0]) if isinstance(eval_results, (list, tuple)) else float(eval_results)
        if best_val is None or eval_val_loss < best_val:
            best_val = eval_val_loss
        print(f"✓ Validation eval after training: val_loss={eval_val_loss:.6f}")
    except Exception as e:
        print(f"Warning: validation evaluation failed: {e}")
        eval_val_loss = None

    # Save final model
    final_keras_path = f"{MODEL_PREFIX}.keras"
    model.save(final_keras_path)
    print(f"✓ Saved model to {final_keras_path}")

    # Export to TensorRT (optional, for GPU inference optimization)
    if TENSORRT_AVAILABLE:
        try:
            print("\nExporting to TensorRT for optimized GPU inference...")
            # TensorRT conversion requires a saved model format
            saved_model_path = f"{MODEL_PREFIX}_saved_model"
            # Keras 3: use tf.saved_model.save instead of deprecated save_format arg
            tf.saved_model.save(model, saved_model_path)

            # Convert to TensorRT using TensorFlow's TensorRT integration
            # Note: This requires TensorRT to be properly installed with CUDA
            converter = tf.experimental.tensorrt.Converter(
                input_saved_model_dir=saved_model_path,
                precision_mode='FP16'  # Use FP16 for better performance
            )
            converter.convert()

            trt_model_path = f"{MODEL_PREFIX}_tensorrt"
            converter.save(trt_model_path)
            print(f"✓ Saved TensorRT optimized model to {trt_model_path}")
        except Exception as e:
            print(f"Note: TensorRT conversion failed (this is optional): {e}")
            print(
                "  TensorRT is available but conversion failed. Model will still work without it.")
    else:
        log_print(
            "\nNote: TensorRT not available. For faster GPU inference, install TensorRT:")
        log_print(
            "  1. Download TensorRT from NVIDIA: https://developer.nvidia.com/tensorrt")
        log_print("  2. Install Python bindings: pip install nvidia-tensorrt")
        log_print("  3. Ensure CUDA/cuDNN versions are compatible with TensorRT")
        log_print(
            "  (The warning about TensorRT is harmless - training works fine without it)")

    # Close log file and print summary
    log_file.close()
    print(f"\n{'='*60}")
    print(f"Training completed. Logs saved to: {RUN_DIR}")
    print(f"  - TensorBoard logs: {RUN_DIR / 'tensorboard'}")
    print(f"  - CSV metrics: {RUN_DIR / 'metrics.csv'}")
    print(f"  - Training log: {RUN_DIR / 'training.log'}")
    print(f"\nTo view TensorBoard, run:")
    print(f"  tensorboard --logdir {RUN_DIR / 'tensorboard'}")
    print(f"{'='*60}")

    # Return metrics for programmatic use (e.g., hyperparameter tuning)
    return {
        "final_loss": final_loss,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "eval_val_loss": eval_val_loss,
        "model_prefix": MODEL_PREFIX,
        "run_dir": str(RUN_DIR),
    }


if __name__ == "__main__":
    main()
