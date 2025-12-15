#!/usr/bin/env python3
"""
Optimized data preprocessing script - extract and process Lichess PGN dataset.
Performance improvements:
- Parallel game parsing with multiprocessing
- Streaming decompression (no full file in memory)
- Direct batch writing to disk
- Optimized game boundary detection
"""
import os
import sys
import time
import zstandard as zstd
import pickle
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import io
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from src (after path is set)
from src.data.parser import parse_pgn_game
from src.data.encoder import FENEncoder


def parse_game_batch(game_texts):
    """
    Parse a batch of games in parallel.

    Args:
        game_texts: List of game text strings

    Returns:
        List of parsed games
    """
    results = []
    for game_text in game_texts:
        result = parse_pgn_game(game_text)
        if result:
            results.append(result)
    return results


def write_games_to_parquet(
    games: list,
    output_path: Path,
    encoder: FENEncoder,
    use_white_elo: bool = True,
    batch_size: int = 500,  # Reduced to avoid OOM when converting to lists
    write_batch_size: int = 500,  # Write to Parquet in smaller chunks
):
    """
    Write games directly to Parquet format using incremental writing.
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

    print(f"Writing {len(games):,} games to Parquet (incremental writing)...")

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
            # Encode FEN sequence during preprocessing (faster training)
            try:
                # Encode the sequence
                sequence, seq_length = encoder.encode_sequence(fen_sequence)
                
                # Convert torch tensor to numpy if needed
                if hasattr(sequence, 'numpy'):
                    sequence_np = sequence.numpy()
                elif hasattr(sequence, 'detach'):
                    # PyTorch tensor - convert to numpy
                    sequence_np = sequence.detach().cpu().numpy()
                else:
                    sequence_np = np.array(sequence, dtype=np.float32)
                
                # Flatten for storage: (max_seq_len, 13, 8, 8) -> (max_seq_len * 13 * 8 * 8,)
                # This makes it easier to store in Parquet
                seq_flat = sequence_np.flatten().astype(np.float32)
                
                # Select target Elo
                target_elo = white_elo if use_white_elo else black_elo
                normalized_elo = encoder.normalize_elo(target_elo)
                
                # Skip sequences with zero or very short length
                if seq_length < 1:
                    continue
                
                sequences_list.append(seq_flat.tolist())  # Convert to list for PyArrow
                lengths_list.append(int(seq_length))
                elo_list.append(float(normalized_elo))
            except Exception as e:
                # Skip games that fail to encode
                print(f"Warning: Failed to encode game: {e}")
                continue

        # Skip empty batches
        if len(sequences_list) == 0:
            continue
            
        # Create Arrow table for this batch
        # Store encoded sequences as flattened arrays
        # We need to store the shape info: (max_seq_len, 13, 8, 8) = (200, 13, 8, 8)
        batch_table = pa.table({
            'sequence_flat': sequences_list,  # List of flattened encoded sequences
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

        # Free the table immediately after writing to reduce memory
        del batch_table

        # Progress update
        if (i // batch_size + 1) % 10 == 0 or total_written == len(games):
            print(f"  Written {total_written:,} / {len(games):,} games")

    # Close writer
    if writer is not None:
        writer.close()

    print(f"✓ Saved {total_written:,} games to {output_path}")


def stream_decompress_and_parse(
    input_path: str,
    output_dir: str = "data/processed",
    max_games: int = None,
    batch_size: int = 1000,
    num_workers: int = None,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
) -> list:
    """
    Stream decompress and parse PGN games with parallel processing.

    Args:
        input_path: Path to .pgn.zst file
        output_dir: Directory to save processed data
        max_games: Maximum games to process (None for all)
        batch_size: Games per batch for parallel processing
        num_workers: Number of parallel workers (None = CPU count)
        chunk_size: Size of chunks to read from decompressed stream

    Returns:
        List of all parsed games
    """
    os.makedirs(output_dir, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free

    print(f"Decompressing and parsing PGN file: {input_path}")
    print(f"Using {num_workers} parallel workers")
    print()

    dctx = zstd.ZstdDecompressor()
    all_games = []
    game_count = 0
    batch_count = 0
    start_time = time.time()

    # Buffer for accumulating game text
    current_game = []
    game_buffer = []

    # Create progress bar
    pbar = tqdm(unit="games", desc="Processing")

    def process_game_batch(games_text_batch):
        """Process a batch of games in parallel."""
        with Pool(num_workers) as pool:
            # Split games into chunks for each worker
            chunk_size = max(1, len(games_text_batch) // num_workers)
            chunks = [
                games_text_batch[i:i + chunk_size]
                for i in range(0, len(games_text_batch), chunk_size)
            ]
            results = pool.map(parse_game_batch, chunks)
            # Flatten results
            return [game for chunk_results in results for game in chunk_results]

    with open(input_path, 'rb') as f_in:
        with dctx.stream_reader(f_in) as reader:
            # Read and decompress in chunks for better memory efficiency
            text_buffer = ""

            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break

                # Decode chunk and add to buffer
                try:
                    text_buffer += chunk.decode('utf-8', errors='ignore')
                except:
                    continue

                # Process complete lines from buffer
                while '\n' in text_buffer:
                    line, text_buffer = text_buffer.split('\n', 1)

                    current_game.append(line)

                    # Check for game boundary (blank line after moves)
                    if line.strip() == "" and current_game:
                        # Check if this looks like a complete game
                        has_moves = any(
                            l.strip().startswith("1. ") or " 1. " in l
                            for l in current_game
                        )

                        if has_moves:
                            game_text = "\n".join(current_game)
                            game_buffer.append(game_text)

                            # Process batch when we have enough games
                            if len(game_buffer) >= batch_size:
                                parsed_batch = process_game_batch(game_buffer)
                                all_games.extend(parsed_batch)
                                game_count += len(parsed_batch)
                                batch_count += 1

                                # Update progress
                                elapsed = time.time() - start_time
                                games_per_sec = game_count / elapsed if elapsed > 0 else 0
                                pbar.update(len(parsed_batch))
                                pbar.set_description(
                                    f"Processing | {game_count:,} games | {games_per_sec:.0f} games/sec"
                                )

                                # Optional: Save checkpoint periodically (can be disabled for Parquet-only)
                                # if batch_count % 50 == 0:
                                #     checkpoint_path = os.path.join(
                                #         output_dir, f"games_batch_{batch_count}.pkl"
                                #     )
                                #     with open(checkpoint_path, 'wb') as f:
                                #         pickle.dump(all_games, f)
                                #     pbar.write(f"Checkpoint: Saved {game_count:,} games")

                                game_buffer = []

                                if max_games and game_count >= max_games:
                                    break

                            current_game = []

            # Process remaining games in buffer
            if game_buffer:
                parsed_batch = process_game_batch(game_buffer)
                all_games.extend(parsed_batch)
                game_count += len(parsed_batch)
                pbar.update(len(parsed_batch))

    pbar.close()

    elapsed = time.time() - start_time
    games_per_sec = game_count / elapsed if elapsed > 0 else 0

    print(f"\n✓ Total games parsed: {game_count:,}")
    print(f"✓ Processing time: {elapsed:.1f} seconds")
    print(f"✓ Average speed: {games_per_sec:.0f} games/second")
    if num_workers > 1:
        print(f"✓ Used {num_workers} parallel workers")
        print(f"  (Typical speedup: 3-5x, limited by chess library and I/O)")
    print()

    # Games will be saved to Parquet in create_datasets()
    # Skip pickle save to save time and disk space

    return all_games


def create_datasets(
    games: list,
    output_dir: str = "data/processed",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    parquet_dir: str = "data/parquet",
    use_white_elo: bool = True,
    max_sequence_length: int = 200,
    save_pickle: bool = False,
) -> tuple:
    """
    Split games into train/val/test sets and save to Parquet.

    Args:
        games: List of parsed games
        output_dir: Directory for optional pickle output
        train_ratio: Train split ratio
        val_ratio: Val split ratio
        test_ratio: Test split ratio
        parquet_dir: Directory for Parquet files
        use_white_elo: Whether to use white or black Elo
        max_sequence_length: Maximum sequence length
        save_pickle: If True, also save pickle files (for backward compatibility)

    Returns:
        Tuple of (train_games, val_games, test_games)
    """
    from src.data.parser import split_dataset

    print("Splitting dataset into train/val/test...")
    train_games, val_games, test_games = split_dataset(
        games, train_ratio, val_ratio, test_ratio
    )

    print(f"\n✓ Train: {len(train_games):,} games ({100*train_ratio:.1f}%)")
    print(f"✓ Val:   {len(val_games):,} games ({100*val_ratio:.1f}%)")
    print(f"✓ Test:  {len(test_games):,} games ({100*test_ratio:.1f}%)")

    # Save to Parquet (primary output format)
    print("\nSaving to Parquet format...")
    encoder = FENEncoder(max_sequence_length)
    parquet_path = Path(parquet_dir)
    parquet_path.mkdir(parents=True, exist_ok=True)

    print("\nWriting train split to Parquet...")
    write_games_to_parquet(
        train_games,
        parquet_path / "train_games.parquet",
        encoder,
        use_white_elo=use_white_elo,
    )

    print("\nWriting validation split to Parquet...")
    write_games_to_parquet(
        val_games,
        parquet_path / "val_games.parquet",
        encoder,
        use_white_elo=use_white_elo,
    )

    print("\nWriting test split to Parquet...")
    write_games_to_parquet(
        test_games,
        parquet_path / "test_games.parquet",
        encoder,
        use_white_elo=use_white_elo,
    )
    print()

    # Optionally save pickle splits for backward compatibility
    if save_pickle:
        os.makedirs(output_dir, exist_ok=True)
        print("Saving dataset splits (pickle) for backward compatibility...")
        splits = {
            'train': train_games,
            'val': val_games,
            'test': test_games,
        }

        splits_path = os.path.join(output_dir, "dataset_splits.pkl")
        with open(splits_path, 'wb') as f:
            pickle.dump(splits, f)
        print(f"✓ Saved to {splits_path}\n")

    return train_games, val_games, test_games


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Optimized preprocessing for Lichess PGN dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="lichess_db_standard_rated_2013-01.pgn.zst",
        help="Path to input .pgn.zst file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,  # Larger batch for better parallelization
        help="Batch size for parallel processing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,  # 1MB
        help="Chunk size for streaming decompression",
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default="data/parquet",
        help="Directory for Parquet output files",
    )
    parser.add_argument(
        "--save-pickle",
        action="store_true",
        help="Also save pickle files (for backward compatibility)",
    )
    parser.add_argument(
        "--use-black-elo",
        action="store_true",
        help="Use black player's Elo instead of white's",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=200,
        help="Maximum sequence length for encoding",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print(f"\nExpected location: {os.path.abspath(args.input)}")
        return

    # Decompress and parse with optimizations
    games = stream_decompress_and_parse(
        args.input,
        args.output,
        args.max_games,
        args.batch_size,
        args.workers,
        args.chunk_size,
    )

    # Create dataset splits and save to Parquet
    create_datasets(
        games,
        args.output,
        parquet_dir=args.parquet_dir,
        use_white_elo=not args.use_black_elo,
        max_sequence_length=args.max_seq_len,
        save_pickle=args.save_pickle,
    )

    print("=" * 60)
    print("✓ Optimized data preprocessing complete!")
    print(f"✓ Parquet files saved to: {args.parquet_dir}")
    if args.save_pickle:
        print(f"✓ Pickle files saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
