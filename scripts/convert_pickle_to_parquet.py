#!/usr/bin/env python3
"""
Convert pickle dataset to Parquet format for optimized training.
Memory-efficient version that processes splits one at a time.
"""
import os
import sys
import gc
import pickle
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_utils_tf import convert_games_to_parquet
from src.data.encoder import FENEncoder


def main(pickle_data_dir="data/small", parquet_output_dir="data/small/parquet", 
         max_sequence_length=200, use_white_elo=True):
    """Convert pickle dataset splits to Parquet format."""
    
    pickle_data_dir = Path(pickle_data_dir)
    parquet_output_dir = Path(parquet_output_dir)
    
    print("=" * 60)
    print("CONVERTING PICKLE DATASET TO PARQUET FORMAT")
    print("=" * 60)
    print()
    
    # Check if pickle files exist
    splits_path = pickle_data_dir / "dataset_splits.pkl"
    if not splits_path.exists():
        print(f"⚠ Pickle dataset not found at {splits_path}")
        return False
    
    try:
        # Create encoder once
        encoder = FENEncoder(max_sequence_length)
        
        # Create output directory
        parquet_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load pickle file
        print(f"Loading pickle dataset from {pickle_data_dir}...")
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        
        print(f"✓ Found dataset splits:")
        for split_name in ['train', 'val', 'test']:
            if split_name in splits:
                print(f"  {split_name.capitalize()}: {len(splits[split_name]):,} games")
        print()
        
        # Convert each split separately to avoid memory issues
        splits_to_convert = [
            ("train", "train_games.parquet"),
            ("val", "val_games.parquet"),
            ("test", "test_games.parquet"),
        ]
        
        success_count = 0
        for split_name, output_filename in splits_to_convert:
            if split_name not in splits:
                print(f"⚠ Skipping {split_name} split (not found in dataset)")
                continue
                
            games = splits[split_name]
            print(f"Converting {split_name} split to Parquet ({len(games):,} games)...")
            
            try:
                convert_games_to_parquet(
                    games,
                    parquet_output_dir / output_filename,
                    encoder,
                    use_white_elo=use_white_elo,
                    batch_size=5000,  # Smaller batches to reduce memory
                    write_batch_size=2500,  # Smaller write batches
                )
                print(f"✓ {split_name.capitalize()} conversion complete\n")
                success_count += 1
                
                # Free memory after each conversion
                del games
                gc.collect()
                
            except Exception as e:
                print(f"✗ Error converting {split_name} split: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next split even if one fails
                continue
        
        # Clean up
        del splits
        gc.collect()
        
        if success_count > 0:
            print("=" * 60)
            print("✓ Conversion complete!")
            print(f"Parquet files saved to: {parquet_output_dir}")
            print("=" * 60)
            print()
            print("Benefits of Parquet format:")
            print("  - ~60% smaller file size (Snappy compression)")
            print("  - Faster I/O with columnar storage")
            print("  - Streaming support for large datasets")
            print("  - Better performance in training scripts")
            return True
        else:
            print("✗ No splits were successfully converted")
            return False
        
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert pickle dataset to Parquet format")
    parser.add_argument("--pickle-dir", type=str, default="data/small",
                        help="Directory containing pickle files (default: data/small)")
    parser.add_argument("--output-dir", type=str, default="data/small/parquet",
                        help="Output directory for Parquet files (default: data/small/parquet)")
    parser.add_argument("--max-sequence-length", type=int, default=200,
                        help="Maximum sequence length (default: 200)")
    parser.add_argument("--use-white-elo", action="store_true", default=True,
                        help="Use white player's Elo (default: True)")
    
    args = parser.parse_args()
    
    main(
        pickle_data_dir=args.pickle_dir,
        parquet_output_dir=args.output_dir,
        max_sequence_length=args.max_sequence_length,
        use_white_elo=args.use_white_elo
    )

