#!/usr/bin/env python3
"""
Data preprocessing script - extract and process Lichess PGN dataset.
"""

import os
import sys
import time
import zstandard as zstd
import pickle
from tqdm import tqdm
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.parser import parse_pgn_stream, split_dataset


def decompress_and_parse_pgn(
    input_path: str,
    output_dir: str = "data/processed",
    max_games: int = None,
    batch_size: int = 1000,
) -> tuple:
    """
    Decompress zstd-compressed PGN and parse games.

    Args:
        input_path: Path to .pgn.zst file
        output_dir: Directory to save processed data
        max_games: Maximum games to process (None for all)
        batch_size: Games per batch

    Returns:
        Tuple of (all_games, stats)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Decompressing and parsing PGN file: {input_path}\n")

    dctx = zstd.ZstdDecompressor()
    all_games = []
    batch_count = 0
    start_time = time.time()

    with open(input_path, 'rb') as f_in:
        with dctx.stream_reader(f_in) as reader:
            # Convert to text stream
            text_stream = reader.read().decode('utf-8').splitlines()

            # Create progress bar for batches
            pbar = tqdm(
                parse_pgn_stream(text_stream, max_games, batch_size),
                unit="batch",
                desc="Processing",
            )

            for batch in pbar:
                all_games.extend(batch)
                batch_count += 1

                # Calculate stats
                elapsed = time.time() - start_time
                games_per_sec = len(all_games) / elapsed if elapsed > 0 else 0

                # Update progress bar description
                pbar.set_description(
                    f"Processing | {len(all_games):,} games | {games_per_sec:.0f} games/sec"
                )

                # Optional: save intermediate checkpoints
                if batch_count % 100 == 0 and batch_count > 0:
                    checkpoint_path = os.path.join(
                        output_dir, f"games_batch_{batch_count}.pkl"
                    )
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(all_games, f)
                    pbar.write(f"Checkpoint: Saved {len(all_games):,} games to {checkpoint_path}")

    elapsed = time.time() - start_time
    games_per_sec = len(all_games) / elapsed if elapsed > 0 else 0

    print(f"\n✓ Total games parsed: {len(all_games):,}")
    print(f"✓ Processing time: {elapsed:.1f} seconds")
    print(f"✓ Average speed: {games_per_sec:.0f} games/second\n")

    # Save all games
    print("Saving all games...")
    games_path = os.path.join(output_dir, "all_games.pkl")
    with open(games_path, 'wb') as f:
        pickle.dump(all_games, f)
    print(f"✓ Saved to {games_path}\n")

    return all_games


def create_datasets(
    games: list,
    output_dir: str = "data/processed",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple:
    """
    Split games into train/val/test sets.

    Args:
        games: List of parsed games
        output_dir: Directory to save datasets
        train_ratio: Train split ratio
        val_ratio: Val split ratio
        test_ratio: Test split ratio

    Returns:
        Tuple of (train_games, val_games, test_games)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Splitting dataset into train/val/test...")
    train_games, val_games, test_games = split_dataset(
        games, train_ratio, val_ratio, test_ratio
    )

    print(f"\n✓ Train: {len(train_games):,} games ({100*train_ratio:.1f}%)")
    print(f"✓ Val:   {len(val_games):,} games ({100*val_ratio:.1f}%)")
    print(f"✓ Test:  {len(test_games):,} games ({100*test_ratio:.1f}%)")

    # Save splits
    print("\nSaving dataset splits...")
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
        description="Preprocess Lichess PGN dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/lichess_db_standard_rated_2013-01.pgn.zst",
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
        default=1000,
        help="Batch size for processing",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print(f"\nExpected location: {os.path.abspath(args.input)}")
        print("\nAvailable files in data/raw/:")
        raw_dir = "data/raw"
        if os.path.exists(raw_dir):
            for f in os.listdir(raw_dir):
                print(f"  - {f}")
        else:
            print("  (data/raw/ directory does not exist)")
        return

    # Decompress and parse
    games = decompress_and_parse_pgn(
        args.input,
        args.output,
        args.max_games,
        args.batch_size,
    )

    # Create dataset splits
    create_datasets(games, args.output)

    print("=" * 60)
    print("✓ Data preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
