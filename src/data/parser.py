"""
PGN Parser for Lichess dataset.

Extracts chess games from PGN format with metadata (Elo ratings, moves).
Converts move sequences to FEN strings for model input.
"""

import io
from typing import List, Tuple, Optional
import chess
import chess.pgn


def parse_pgn_game(game_text: str) -> Optional[Tuple[List[str], int, int, str]]:
    """
    Parse a single PGN game string.

    Args:
        game_text: PGN game text

    Returns:
        Tuple of (fen_sequence, white_elo, black_elo, result) or None if invalid
        - fen_sequence: List of FEN strings (board position after each move)
        - white_elo: White player's Elo rating
        - black_elo: Black player's Elo rating
        - result: Game result ('1-0', '0-1', '1/2-1/2')
    """
    try:
        # Parse the PGN
        pgn = chess.pgn.read_game(io.StringIO(game_text))

        if pgn is None:
            return None

        # Extract Elo ratings
        headers = pgn.headers
        white_elo = headers.get("WhiteElo")
        black_elo = headers.get("BlackElo")
        result = headers.get("Result")

        # Skip if missing required information
        if not white_elo or not black_elo or not result:
            return None

        try:
            white_elo = int(white_elo)
            black_elo = int(black_elo)
        except (ValueError, TypeError):
            return None

        # Validate Elo ratings are reasonable
        if white_elo < 100 or white_elo > 5000 or black_elo < 100 or black_elo > 5000:
            return None

        # Extract move sequence as FEN strings
        fen_sequence = []
        board = chess.Board()

        # Add initial position
        fen_sequence.append(board.fen())

        # Process each move
        for move in pgn.mainline_moves():
            board.push(move)
            fen_sequence.append(board.fen())

        # Skip games with very few moves (< 2 moves)
        if len(fen_sequence) < 3:  # Initial + at least 2 moves
            return None

        return fen_sequence, white_elo, black_elo, result

    except Exception:
        # Skip games that fail to parse
        return None


def parse_pgn_stream(pgn_file, max_games: Optional[int] = None,
                     batch_size: int = 1000) -> Tuple[List, int]:
    """
    Stream parse PGN games from a file object.

    Yields batches of parsed games.

    Args:
        pgn_file: Open file object or iterable of lines
        max_games: Maximum number of games to parse (None for all)
        batch_size: Number of games to accumulate before yielding

    Yields:
        List of parsed games (fen_sequence, white_elo, black_elo, result)
    """
    batch = []
    game_count = 0
    current_game = []

    for line in pgn_file:
        current_game.append(line)

        # Games are separated by blank lines
        # Only process if we have content and it looks like a complete game
        if line.strip() == "" and current_game:
            # Check if this looks like a complete game (has both headers and moves)
            # Look for move notation: lines starting with "1. " or containing " 1. "
            has_moves = any(
                l.strip().startswith("1. ") or " 1. " in l
                for l in current_game
            )

            if has_moves:
                game_text = "\n".join(current_game)
                result = parse_pgn_game(game_text)

                if result:
                    batch.append(result)
                    game_count += 1

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                current_game = []

                if max_games and game_count >= max_games:
                    break
            # else: skip this blank line, continue accumulating

    # Yield remaining batch
    if batch:
        yield batch


def extract_white_moves(fen_sequence: List[str]) -> List[str]:
    """
    Extract only the board positions after White's moves.

    Args:
        fen_sequence: List of FEN strings for entire game

    Returns:
        List of FEN strings after each White move
    """
    # White moves are at even indices (0, 2, 4, ...)
    return fen_sequence[::2]


def extract_black_moves(fen_sequence: List[str]) -> List[str]:
    """
    Extract only the board positions after Black's moves.

    Args:
        fen_sequence: List of FEN strings for entire game

    Returns:
        List of FEN strings after each Black move
    """
    # Black moves are at odd indices (1, 3, 5, ...)
    return fen_sequence[1::2]


def split_dataset(
    games: List[Tuple[List[str], int, int, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test sets.

    Args:
        games: List of parsed games
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_games, val_games, test_games)
    """
    import random
    random.seed(seed)

    # Shuffle games
    shuffled = games.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test
