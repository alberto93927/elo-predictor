#!/usr/bin/env python3
"""
Convert pickle dataset to Parquet format for optimized training.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_utils_tf import load_pickle_dataset, convert_games_to_parquet
from src.data.encoder import FENEncoder


def main():
    """Convert pickle dataset splits to Parquet format."""
    DATA_DIR = Path("data/processed")
    PARQUET_DIR = Path("data/parquet")
    MAX_SEQUENCE_LENGTH = 200
    USE_WHITE_ELO = True
    
    print("=" * 60)
    print("CONVERTING DATASET TO PARQUET FORMAT")
    print("=" * 60)
    print()
    
    # Load pickle data
    print("Loading pickle dataset...")
    train_games, val_games, test_games = load_pickle_dataset(str(DATA_DIR))
    
    print(f"Train: {len(train_games):,} games")
    print(f"Val: {len(val_games):,} games")
    print(f"Test: {len(test_games):,} games")
    print()
    
    # Create encoder
    encoder = FENEncoder(MAX_SEQUENCE_LENGTH)
    
    # Create output directory
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert each split
    print("Converting train split...")
    convert_games_to_parquet(
        train_games,
        PARQUET_DIR / "train_games.parquet",
        encoder,
        use_white_elo=USE_WHITE_ELO,
    )
    print()
    
    print("Converting validation split...")
    convert_games_to_parquet(
        val_games,
        PARQUET_DIR / "val_games.parquet",
        encoder,
        use_white_elo=USE_WHITE_ELO,
    )
    print()
    
    print("Converting test split...")
    convert_games_to_parquet(
        test_games,
        PARQUET_DIR / "test_games.parquet",
        encoder,
        use_white_elo=USE_WHITE_ELO,
    )
    print()
    
    print("=" * 60)
    print("✓ Conversion complete!")
    print(f"Parquet files saved to: {PARQUET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()



