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
from multiprocessing import Pool, cpu_count

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.parser import parse_pgn_stream, split_dataset, parse_pgn_game


def parse_game_batch(game_texts):
    """Parse a batch of games (for parallel processing)."""
    results = []
    for game_text in game_texts:
        result = parse_pgn_game(game_text)
        if result:
            results.append(result)
    return results


def decompress_and_parse_pgn(
    input_path: str,
    output_dir: str = "data/processed",
    max_games: int = None,
    batch_size: int = 1000,
    num_workers: int = None,
) -> tuple:
    """
    Decompress zstd-compressed PGN and parse games with parallel processing.

    Args:
        input_path: Path to .pgn.zst file
        output_dir: Directory to save processed data
        max_games: Maximum games to process (None for all)
        batch_size: Games per batch for parallel processing
        num_workers: Number of parallel workers (None = CPU count - 1)

    Returns:
        Tuple of (all_games, stats)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free

    print(f"Decompressing and parsing PGN file: {input_path}")
    if num_workers > 1:
        print(f"Using {num_workers} parallel workers")
        print(f"Note: Actual speedup depends on bottlenecks (I/O, chess library operations)")
        print(f"      Typical speedup: 3-5x (not linear due to I/O and library overhead)")
    print()

    dctx = zstd.ZstdDecompressor()
    all_games = []
    batch_count = 0
    start_time = time.time()
    
    # Collect game texts first (faster than parsing immediately)
    game_texts = []
    game_count = 0

    with open(input_path, 'rb') as f_in:
        with dctx.stream_reader(f_in) as reader:
            # Use the existing parse_pgn_stream function for game extraction
            # But read in chunks to avoid loading entire file
            text_buffer = ""
            current_game = []
            chunk_size = 1024 * 1024  # 1MB chunks
            
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                
                text_buffer += chunk.decode('utf-8', errors='ignore')
                
                # Process complete lines
                while '\n' in text_buffer:
                    line, text_buffer = text_buffer.split('\n', 1)
                    current_game.append(line)
                    
                    # Check for game boundary
                    if line.strip() == "" and current_game:
                        # Check if it's a complete game
                        has_moves = any(
                            l.strip().startswith("1. ") or " 1. " in l
                            for l in current_game
                        )
                        if has_moves:
                            game_text = "\n".join(current_game)
                            game_texts.append(game_text)
                            game_count += 1
                            
                            if max_games and game_count >= max_games:
                                break
                        current_game = []
            
            # Process any remaining game
            if current_game:
                game_text = "\n".join(current_game)
                game_texts.append(game_text)

    # Now parse games in parallel batches
    print(f"Parsing {len(game_texts):,} games with {num_workers} workers...")
    pbar = tqdm(total=len(game_texts), unit="games", desc="Parsing")
    
    if num_workers > 1:
        # Parallel processing
        with Pool(num_workers) as pool:
            # Split into batches
            batches = [
                game_texts[i:i + batch_size]
                for i in range(0, len(game_texts), batch_size)
            ]
            
            # Process batches in parallel
            for batch_texts in batches:
                # Split batch across workers
                chunk_size = max(1, len(batch_texts) // num_workers)
                chunks = [
                    batch_texts[i:i + chunk_size]
                    for i in range(0, len(batch_texts), chunk_size)
                ]
                
                # Parse chunks in parallel
                results = pool.map(parse_game_batch, chunks)
                parsed_batch = [game for chunk_results in results for game in chunk_results]
                
                all_games.extend(parsed_batch)
                batch_count += 1
                pbar.update(len(batch_texts))
                
                # Calculate stats
                elapsed = time.time() - start_time
                games_per_sec = len(all_games) / elapsed if elapsed > 0 else 0
                pbar.set_description(
                    f"Parsing | {len(all_games):,} games | {games_per_sec:.0f} games/sec"
                )
                
                # Save checkpoint periodically
                if batch_count % 50 == 0:
                    checkpoint_path = os.path.join(
                        output_dir, f"games_batch_{batch_count}.pkl"
                    )
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(all_games, f)
                    pbar.write(f"Checkpoint: Saved {len(all_games):,} games")
    else:
        # Sequential processing (fallback)
        for i in range(0, len(game_texts), batch_size):
            batch_texts = game_texts[i:i + batch_size]
            parsed_batch = parse_game_batch(batch_texts)
            all_games.extend(parsed_batch)
            batch_count += 1
            pbar.update(len(batch_texts))
            
            elapsed = time.time() - start_time
            games_per_sec = len(all_games) / elapsed if elapsed > 0 else 0
            pbar.set_description(
                f"Parsing | {len(all_games):,} games | {games_per_sec:.0f} games/sec"
            )
    
    pbar.close()

    elapsed = time.time() - start_time
    games_per_sec = len(all_games) / elapsed if elapsed > 0 else 0

    print(f"\n✓ Total games parsed: {len(all_games):,}")
    print(f"✓ Processing time: {elapsed:.1f} seconds")
    print(f"✓ Average speed: {games_per_sec:.0f} games/second")
    if num_workers > 1:
        # Realistic speedup: typically 3-5x due to I/O and chess library bottlenecks
        # Don't claim linear speedup - it's misleading
        print(f"✓ Used {num_workers} parallel workers")
        print(f"  (Note: Speedup is limited by I/O and chess library operations)")
    print()

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
        default=2000,
        help="Batch size for parallel processing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
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
        args.workers,
    )

    # Create dataset splits
    create_datasets(games, args.output)

    print("=" * 60)
    print("✓ Data preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
