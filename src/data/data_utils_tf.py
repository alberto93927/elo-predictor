"""
TensorFlow data utilities for Elo prediction training.
Supports both pickle and Parquet data formats.
"""
import pickle
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Tuple, Optional
import tensorflow as tf

from src.data.encoder import FENEncoder


def load_pickle_dataset(data_dir: str = "data/processed"):
    """Load dataset splits from pickle files."""
    splits_path = Path(data_dir) / "dataset_splits.pkl"
    
    if not splits_path.exists():
        raise FileNotFoundError(f"Dataset splits not found at {splits_path}")
    
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    return splits['train'], splits['val'], splits['test']


def convert_games_to_parquet(
    games: List[Tuple],
    output_path: Path,
    encoder: FENEncoder,
    use_white_elo: bool = True,
    batch_size: int = 10000,
    write_batch_size: int = 5000,
):
    """
    Convert games list to Parquet format using incremental writing.
    This avoids loading all games into memory at once.
    
    Args:
        games: List of (fen_sequence, white_elo, black_elo, result) tuples
        output_path: Path to save Parquet file
        encoder: FENEncoder instance
        use_white_elo: If True use white's Elo, else use black's
        batch_size: Number of games to process at once
        write_batch_size: Number of games to accumulate before writing to Parquet
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {len(games):,} games to Parquet (incremental writing)...")
    
    # Use ParquetWriter for incremental writing
    writer = None
    total_written = 0
    
    # Process games in batches
    for i in range(0, len(games), batch_size):
        batch_games = games[i:i + batch_size]
        
        # Accumulate data for this batch
        sequences_list = []
        lengths_list = []
        elo_list = []
        
        for fen_sequence, white_elo, black_elo, result in batch_games:
            # Encode sequence
            sequence, length = encoder.encode_sequence(fen_sequence)
            
            # Convert to numpy (remove torch dependency)
            if hasattr(sequence, 'numpy'):
                sequence_np = sequence.numpy()
            else:
                sequence_np = np.array(sequence)
            
            # Select target Elo
            target_elo = white_elo if use_white_elo else black_elo
            normalized_elo = encoder.normalize_elo(target_elo)
            
            # Flatten sequence for storage: (seq_len, 13, 8, 8) -> (seq_len, 832)
            seq_flat = sequence_np.reshape(sequence_np.shape[0], -1).astype(np.float32)
            
            # Ensure length is Python int
            length_int = int(length)
            
            sequences_list.append(seq_flat)
            lengths_list.append(length_int)
            elo_list.append(float(normalized_elo))
        
        # Create Arrow table for this batch
        # Convert sequences to nested lists for PyArrow (it needs list of lists, not numpy arrays)
        sequences_nested = [seq.tolist() for seq in sequences_list]
        
        batch_table = pa.table({
            'sequence': sequences_nested,  # List of lists (nested structure)
            'length': pa.array(lengths_list, type=pa.int32()),
            'elo': pa.array(elo_list, type=pa.float32()),
        })
        
        # Initialize writer on first batch
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                batch_table.schema,
                compression='snappy',
            )
        
        # Write batch to Parquet
        writer.write_table(batch_table)
        total_written += len(batch_games)
        
        # Progress update
        if (i // batch_size + 1) % 10 == 0 or total_written == len(games):
            print(f"  Written {total_written:,} / {len(games):,} games")
    
    # Close writer
    if writer is not None:
        writer.close()
    
    print(f"✓ Saved {total_written:,} games to {output_path}")


def parquet_row_group_generator(parquet_path, row_group_indices, encoder: FENEncoder):
    """
    Generator that yields batches from Parquet row groups.
    
    Args:
        parquet_path: Path to Parquet file
        row_group_indices: List of row group indices to read
        encoder: FENEncoder for sequence reshaping
    
    Yields:
        Tuple of (sequences, lengths, targets) as numpy arrays
    """
    pf = pq.ParquetFile(parquet_path)
    
    for rg_idx in row_group_indices:
        table = pf.read_row_group(rg_idx)
        df = table.to_pandas()
        
        # Check for different data formats
        if 'sequence_flat' in df.columns:
            # New format: Pre-encoded sequences stored as flattened arrays
            sequences = []
            valid_lengths = []
            valid_targets = []
            
            # Get the expected shape from encoder
            max_seq_len = encoder.max_sequence_length
            seq_shape = (max_seq_len, encoder.board_channels, 8, 8)
            flat_size = max_seq_len * encoder.board_channels * 8 * 8
            
            for idx, (seq_flat, length, elo) in enumerate(zip(df['sequence_flat'], df['length'], df['elo'])):
                try:
                    # Convert to numpy array
                    if isinstance(seq_flat, list):
                        seq_flat = np.array(seq_flat, dtype=np.float32)
                    else:
                        seq_flat = np.array(seq_flat, dtype=np.float32)
                    
                    # Reshape from flattened to (max_seq_len, 13, 8, 8)
                    if seq_flat.size != flat_size:
                        # Handle mismatched sizes (shouldn't happen, but be safe)
                        print(f"Warning: Sequence size mismatch: expected {flat_size}, got {seq_flat.size}")
                        continue
                    
                    sequence = seq_flat.reshape(seq_shape).astype(np.float32)
                    
                    # Skip sequences with zero or very short length (can cause NaN)
                    length_int = int(length)
                    if length_int < 1:
                        continue
                    
                    sequences.append(sequence)
                    valid_lengths.append(length_int)
                    
                    # Ensure ELO is normalized (should be in [0, 1] range)
                    elo_float = float(elo)
                    if elo_float > 1.0 or elo_float < 0.0:
                        # Raw ELO detected, normalize it
                        elo_float = encoder.normalize_elo(int(elo_float))
                    valid_targets.append(elo_float)
                except Exception as e:
                    print(f"Warning: Skipping row {idx}: {e}")
                    continue
        elif 'fen_sequence' in df.columns:
            # Old format: FEN strings stored directly - encode on-the-fly (backward compatibility)
            sequences = []
            valid_lengths = []
            valid_targets = []
            
            for idx, (fen_seq, length, elo) in enumerate(zip(df['fen_sequence'], df['length'], df['elo'])):
                try:
                    # Encode FEN sequence on-the-fly
                    sequence, seq_length = encoder.encode_sequence(fen_seq)
                    
                    # Convert to numpy if needed
                    if hasattr(sequence, 'numpy'):
                        sequence_np = sequence.numpy()
                    elif hasattr(sequence, 'detach'):
                        sequence_np = sequence.detach().cpu().numpy()
                    else:
                        sequence_np = np.array(sequence, dtype=np.float32)
                    
                    # Skip sequences with zero or very short length (can cause NaN)
                    if seq_length < 1:
                        continue
                    
                    sequences.append(sequence_np)
                    valid_lengths.append(int(seq_length))
                    # Ensure ELO is normalized (should be in [0, 1] range)
                    elo_float = float(elo)
                    if elo_float > 1.0 or elo_float < 0.0:
                        # Raw ELO detected, normalize it
                        elo_float = encoder.normalize_elo(int(elo_float))
                    valid_targets.append(elo_float)
                except Exception as e:
                    print(f"Warning: Skipping row {idx}: {e}")
                    continue
        else:
            # Unknown format
            print(f"Warning: Row group {rg_idx} uses unknown format, skipping")
            continue
        
        if len(sequences) == 0:
            # Skip this row group if no valid sequences
            continue
        
        # Stack sequences
        sequences = np.stack(sequences).astype(np.float32)
        # CRITICAL: Ensure length is int32, not float
        lengths = np.array(valid_lengths, dtype=np.int32)
        targets = np.array(valid_targets, dtype=np.float32).reshape(-1, 1)
        
        # Clip targets to [0, 1] range to prevent NaN from out-of-range values
        # This is a safety measure in case normalization wasn't applied correctly
        targets = np.clip(targets, 0.0, 1.0)
        
        yield (sequences, lengths, targets)


def create_streaming_dataset(
    parquet_path: Path,
    row_group_indices: List[int],
    batch_size: int,
    shuffle_buffer: Optional[int] = None,
    num_parallel_calls: Optional[int] = None,
    max_sequence_length: int = 200,
):
    """
    Create an optimized tf.data.Dataset that streams from Parquet row groups.
    
    Args:
        parquet_path: Path to Parquet file
        row_group_indices: List of row group indices to include
        batch_size: Batch size for training
        shuffle_buffer: Optional buffer size for shuffling (None = no shuffle)
        num_parallel_calls: Number of parallel calls (None = AUTOTUNE)
        max_sequence_length: Maximum sequence length
    
    Returns:
        tf.data.Dataset configured for streaming
    """
    parquet_path_str = str(parquet_path)
    
    # Create encoder for reshaping (we just need max_sequence_length)
    encoder = FENEncoder(max_sequence_length)
    
    # Simplified approach: single generator that loads all row groups
    # This avoids TensorFlow's generator state management issues with parallel interleave
    def load_all_samples():
        """Generator that loads all samples from all row groups sequentially."""
        for rg_idx in row_group_indices:
            try:
                for sequences, lengths, targets in parquet_row_group_generator(
                    parquet_path_str, [rg_idx], encoder
                ):
                    # Yield individual samples with correct dtypes at source
                    for i in range(len(sequences)):
                        # lengths is already int32 from numpy array above
                        # Convert to Python int for TensorFlow (will become int32 tensor)
                        length_int = int(lengths[i])
                        
                        # Ensure sequence is float32
                        seq = sequences[i].astype(np.float32)
                        
                        yield {
                            "sequence": seq,
                            "length": length_int,  # Python int -> TensorFlow int32
                        }, targets[i].astype(np.float32)
            except Exception as e:
                # Skip problematic row groups
                print(f"Warning: Skipping row group {rg_idx}: {e}")
                continue
    
    # Create dataset from generator
    # Use a single generator to avoid TensorFlow's parallel generator state issues
    dataset = tf.data.Dataset.from_generator(
        load_all_samples,
        output_signature=(
            {
                "sequence": tf.TensorSpec(
                    shape=(max_sequence_length, 13, 8, 8), 
                    dtype=tf.float32
                ),
                "length": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        ),
    )
    
    # Shuffle if requested
    if shuffle_buffer is not None and shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Configure dataset options to handle end-of-sequence gracefully
    # This reduces warnings in multi-GPU training
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    
    # CRITICAL: Ensure length is int32 after batching
    # Mixed precision can convert everything to float16, so we must explicitly cast
    def ensure_int32_length(x, y):
        # Force cast to int32 - handle float16 -> int32 conversion
        length = x['length']
        # Convert via float32 to handle float16 properly
        x['length'] = tf.cast(tf.cast(length, tf.float32), tf.int32)
        return x, y
    
    dataset = dataset.map(ensure_int32_length, num_parallel_calls=tf.data.AUTOTUNE)
    # Aggressive prefetching: prefetch multiple batches to keep GPU fed
    # Prefetch 4-8 batches ahead to prevent GPU starvation
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_dataset_from_pickle(
    games: List[Tuple],
    encoder: FENEncoder,
    batch_size: int,
    shuffle: bool = True,
    use_white_elo: bool = True,
    max_sequence_length: int = 200,
):
    """
    Create tf.data.Dataset from pickle games list.
    For smaller datasets or when Parquet is not available.
    
    Args:
        games: List of (fen_sequence, white_elo, black_elo, result) tuples
        encoder: FENEncoder instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        use_white_elo: Whether to use white or black Elo
        max_sequence_length: Maximum sequence length
    
    Returns:
        tf.data.Dataset
    """
    def generator():
        for fen_sequence, white_elo, black_elo, result in games:
            # Encode sequence
            sequence, length = encoder.encode_sequence(fen_sequence)
            
            # Convert to numpy
            if hasattr(sequence, 'numpy'):
                sequence_np = sequence.numpy()
            else:
                sequence_np = np.array(sequence)
            
            # Select target Elo
            target_elo = white_elo if use_white_elo else black_elo
            normalized_elo = encoder.normalize_elo(target_elo)
            
            # Ensure length is a Python int, not numpy scalar
            length_int = int(length)
            
            yield {
                "sequence": sequence_np.astype(np.float32),
                "length": length_int,  # Python int, will be converted to int32 by TensorFlow
            }, np.array([normalized_elo], dtype=np.float32)
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                "sequence": tf.TensorSpec(
                    shape=(max_sequence_length, 13, 8, 8),
                    dtype=tf.float32
                ),
                "length": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        ),
    )
    
    if shuffle:
        dataset = dataset.shuffle(min(len(games), 10000), reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size)
    
    # CRITICAL: After batching, mixed precision might convert length to float16
    # Force it back to int32 - this is the source fix
    def fix_length_dtype(x, y):
        # Convert length from any dtype (including float16) back to int32
        x['length'] = tf.cast(x['length'], tf.int32)
        return x, y
    
    # Apply the fix immediately after batching
    dataset = dataset.map(fix_length_dtype, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

