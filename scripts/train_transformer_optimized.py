
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
_gpus_immediate = tf.config.list_physical_devices('GPU')
if len(_gpus_immediate) == 0:
    print("WARNING: No GPUs detected immediately after TensorFlow import!")
    print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    print(
        f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
else:
    print(
        f"✓ Detected {len(_gpus_immediate)} GPU(s) immediately after TensorFlow import")


# Check for TensorRT availability (optional, for inference optimization)
TENSORRT_AVAILABLE = False
try:
    # Try to import TensorRT Python bindings
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # TensorRT not installed - that's okay, it's optional
    pass

# Import src modules AFTER TensorFlow and sys.path setup
from src.models.transformer_tf import build_transformer_model
from src.data.data_utils_tf import (
    load_pickle_dataset,
    convert_games_to_parquet,
    create_streaming_dataset,
    create_dataset_from_pickle,
)
from src.data.encoder import FENEncoder

def find_latest_checkpoint(checkpoint_dir: Path, prefix: str):
    """
    Find the latest checkpoint file in the checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Prefix for checkpoint filenames (e.g., "transformer_elo")

    Returns:
        Path to latest checkpoint or None if not found
    """
    if not checkpoint_dir.exists():
        return None

    # Look for checkpoint files matching pattern: {prefix}_checkpoint_epoch-{epoch}.keras
    pattern = f"{prefix}_checkpoint_epoch-*.keras"
    checkpoints = sorted(checkpoint_dir.glob(pattern))

    if not checkpoints:
        return None

    # Return the latest (highest epoch number)
    return checkpoints[-1]


def get_epoch_from_checkpoint(checkpoint_path: Path) -> int:
    """
    Extract epoch number from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Epoch number (0-indexed)
    """
    # Expected format: {prefix}_checkpoint_epoch-{epoch}.keras
    filename = checkpoint_path.stem  # Get name without extension
    parts = filename.split("_epoch-")
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


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
        description="Train optimized Transformer model for Elo prediction"
    )
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
                        help="Number of transformer layers (default: 6, increased from 4)")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads (default: 8)")
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
    parser.add_argument("--shuffle-buffer", type=int, default=10000,
                        help="Shuffle buffer size (default: 10000)")
    parser.add_argument("--model-prefix", type=str, default="transformer_elo_optimized",
                        help="Model prefix for checkpoints (default: transformer_elo_optimized)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Data directory (default: data/processed)")
    parser.add_argument("--parquet-dir", type=str, default="data/parquet",
                        help="Parquet data directory (default: data/parquet)")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--no-parquet", action="store_true",
                        help="Force use of pickle files instead of Parquet")
    parser.add_argument("--use-black-elo", action="store_true",
                        help="Use black player's Elo instead of white's")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    DATA_DIR = Path(args.data_dir)
    PARQUET_DIR = Path(args.parquet_dir)

    # Batch size per GPU - increased for better GPU utilization
    # With 4 GPUs, effective batch size = BATCH_SIZE * 4
    BATCH_SIZE = args.batch_size

    EPOCHS = args.epochs
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    WARMUP_EPOCHS = args.warmup_epochs
    USE_COSINE_SCHEDULE = args.use_cosine_schedule
    WEIGHT_DECAY = args.weight_decay
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    MODEL_PREFIX = args.model_prefix
    CHECKPOINT_DIR = Path(args.checkpoint_dir)

    # Optimized shuffle buffer
    SHUFFLE_BUFFER = args.shuffle_buffer

    # Enable mixed precision for ~2x speedup on modern GPUs
    USE_MIXED_PRECISION = not args.no_mixed_precision

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
    FEEDFORWARD_DIM = args.feedforward_dim
    DROPOUT = args.dropout
    MAX_SEQUENCE_LENGTH = args.max_seq_len

    # Use Parquet if available, otherwise fall back to pickle
    USE_PARQUET = not args.no_parquet
    USE_WHITE_ELO = not args.use_black_elo

    # Check GPU detection immediately after TensorFlow import
    gpus = tf.config.list_physical_devices('GPU')

    print("=" * 60)
    print("OPTIMIZED TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Batch size per GPU: {BATCH_SIZE}")
    print(f"Number of GPUs: {len(gpus)}")
    if len(gpus) > 0:
        print(f"Effective batch size: {BATCH_SIZE * len(gpus)}")
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
    print(
        f"Model: Transformer (embed_dim={EMBEDDING_DIM}, layers={NUM_LAYERS})")
    if TENSORRT_AVAILABLE:
        print(f"TensorRT: Available (will optimize inference)")
    else:
        print(f"TensorRT: Not available (optional, for inference optimization)")
    print("=" * 60)
    print()

    # Enable mixed precision if requested
    if USE_MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✓ Mixed precision (FP16) enabled")
        print("  Note: Output layer will use float32 for numerical stability")
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

        # Create streaming datasets
        train_row_groups = list(range(num_train_groups))
        val_row_groups = list(range(num_val_groups))

        print("Creating optimized datasets...")
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
        print("✓ Datasets created")
        print()

    else:
        # Fall back to pickle files
        print("Parquet files not found, using pickle files...")
        print("(Consider converting to Parquet for better performance)")
        print()

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

    # Check for existing checkpoint
    CHECKPOINT_DIR.mkdir(exist_ok=True)

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

    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR, MODEL_PREFIX)
    initial_epoch = 0

    if latest_checkpoint is not None:
        epoch_num = get_epoch_from_checkpoint(latest_checkpoint)
        print(f"Found checkpoint: {latest_checkpoint}")
        print(f"Resuming from epoch {epoch_num + 1}")
        initial_epoch = epoch_num + 1
    else:
        print("No checkpoint found, starting from scratch")
    print()

    # Setup multi-GPU strategy (exactly like run_full_train_optimized.py)
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        print(f"Using {len(gpus)} GPUs with MirroredStrategy")
        strategy = tf.distribute.MirroredStrategy()
        print(f"✓ MirroredStrategy initialized")
    else:
        strategy = tf.distribute.get_strategy()
        print(f"Using single GPU or CPU")
    print()

    # Build or load model within strategy scope
    with strategy.scope():
        if latest_checkpoint is not None:
            print(f"Loading model from {latest_checkpoint}...")
            model = tf.keras.models.load_model(latest_checkpoint)
            print("✓ Checkpoint loaded successfully")
        else:
            model = build_transformer_model(
                embedding_dim=EMBEDDING_DIM,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                feedforward_dim=FEEDFORWARD_DIM,
                dropout=DROPOUT,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                fast=False,
                verbose=True,
            )

            # Custom loss function that handles NaN gracefully
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
                loss = tf.where(tf.math.is_nan(loss), 1e6 *
                                tf.ones_like(loss), loss)
                return loss

            def safe_mae_metric(y_true, y_pred):
                """MAE metric with NaN protection."""
                y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
                y_pred = tf.where(tf.math.is_nan(y_pred),
                                  0.5 * tf.ones_like(y_pred), y_pred)
                # Compute MAE manually: mean absolute difference
                mae = tf.reduce_mean(tf.abs(y_true - y_pred))
                mae = tf.where(tf.math.is_nan(mae), tf.zeros_like(mae), mae)
                return mae

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

    # Callbacks
    callbacks = [
        create_epoch_logger(log_every=1, log_file=log_file),
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
        # Periodic checkpoint saving (every epoch)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                CHECKPOINT_DIR / f"{MODEL_PREFIX}_checkpoint_epoch-{{epoch:02d}}.keras"),
            save_freq="epoch",
            save_weights_only=False,
            verbose=0,
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

    remaining_epochs = EPOCHS - initial_epoch
    if remaining_epochs <= 0:
        print(f"Training already completed ({initial_epoch}/{EPOCHS} epochs)")
        print("Using checkpointed model for export...")
    else:
        print(
            f"Starting/resuming training ({remaining_epochs} epochs remaining, "
            f"batch={BATCH_SIZE} per GPU, effective={BATCH_SIZE * len(tf.config.list_physical_devices('GPU'))}, "
            f"val_split={VAL_SPLIT})"
        )
        print("Using optimized streaming datasets with:")
        print(f"  - Shuffle buffer: {SHUFFLE_BUFFER:,}")
        print(f"  - Parallel data loading: {NUM_PARALLEL_CALLS}")
        print(f"  - Mixed precision: {USE_MIXED_PRECISION}")
        print()

        start_train = time.perf_counter()

        # Suppress warnings about end-of-sequence and rendezvous cancellations (normal in multi-GPU training)
        # These are harmless INFO messages from TensorFlow's multi-GPU coordination
        import warnings
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
        if len(tf.config.list_physical_devices('GPU')) > 1:
            original_stderr = sys.stderr
            filtered_stderr = RendezvousFilter(original_stderr)
            sys.stderr = filtered_stderr

        try:
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=EPOCHS,
                initial_epoch=initial_epoch,
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

    # Save final model
    final_keras_path = f"{MODEL_PREFIX}.keras"
    model.save(final_keras_path)
    print(f"✓ Saved model to {final_keras_path}")

    # Export to TFLite (optional - may fail with LayerNormalization)
    # Note: TFLite has limited support for some Keras layers, particularly LayerNormalization
    # This is a known limitation and doesn't affect training or inference with the .keras model
    print("\nAttempting TFLite export (this may fail with transformer models due to LayerNormalization)...")
    try:
        # Try converting to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Set converter options to be more permissive
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops
            tf.lite.OpsSet.SELECT_TF_OPS,    # Enable TensorFlow ops (fallback)
        ]
        converter._experimental_lower_tensor_list_ops = False

        tflite_float = converter.convert()
        float_path = Path(f"{MODEL_PREFIX}.tflite")
        float_path.write_bytes(tflite_float)
        log_print(f"✓ Saved TFLite (float32) to {float_path}")

        # Try int8 quantization (may also fail)
        try:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_int8 = converter.convert()
            int8_path = Path(f"{MODEL_PREFIX}_int8.tflite")
            int8_path.write_bytes(tflite_int8)
            log_print(f"✓ Saved TFLite (dynamic range int8) to {int8_path}")
        except Exception as e2:
            log_print(
                f"Note: Int8 quantization failed (this is optional): {e2}")

    except Exception as e:
        log_print(f"⚠ TFLite export failed: {e}")
        log_print(
            "  This is a known limitation with transformer models using LayerNormalization.")
        log_print(
            "  The .keras model works fine for inference - TFLite is only needed for mobile/edge deployment.")
        log_print("  If you need TFLite, consider:")
        log_print(
            "    1. Using SavedModel format instead: model.save('model_savedmodel', save_format='tf')")
        log_print("    2. Using TensorFlow Serving for deployment")
        log_print(
            "    3. Replacing LayerNormalization with BatchNormalization (may affect performance)")

    # Export to TensorRT (optional, for GPU inference optimization)
    if TENSORRT_AVAILABLE:
        try:
            print("\nExporting to TensorRT for optimized GPU inference...")
            # TensorRT conversion requires a saved model format
            saved_model_path = f"{MODEL_PREFIX}_saved_model"
            model.save(saved_model_path, save_format='tf')

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


if __name__ == "__main__":
    main()
