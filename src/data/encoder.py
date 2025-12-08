"""
FEN String Encoder for feature engineering.

Converts FEN strings (board positions) into numerical embeddings for model input.
"""

import numpy as np
from typing import List, Tuple
import torch


class FENEncoder:
    """Encode FEN strings into numerical representations."""

    def __init__(self, max_sequence_length: int = 200):
        """
        Initialize FEN encoder.

        Args:
            max_sequence_length: Maximum number of positions in a sequence
        """
        self.max_sequence_length = max_sequence_length

        # Piece to index mapping (for 8x8 board = 64 squares)
        self.piece_to_idx = {
            'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,  # Black pieces
            'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12,  # White pieces
            '.': 0  # Empty square
        }

        self.board_channels = 13  # 12 piece types + 1 empty

    def fen_to_board_tensor(self, fen: str) -> torch.Tensor:
        """
        Convert a single FEN string to a board tensor.

        Args:
            fen: FEN string (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        Returns:
            Tensor of shape (13, 8, 8) - 13 channels for piece types, 8x8 board
        """
        # Extract board position from FEN (before the space)
        board_part = fen.split(' ')[0]

        # Initialize board tensor
        board = torch.zeros(self.board_channels, 8, 8, dtype=torch.float32)

        # Parse board from FEN
        ranks = board_part.split('/')
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    file_idx += int(char)
                else:
                    # Piece
                    piece_idx = self.piece_to_idx.get(char, 0)
                    board[piece_idx, rank_idx, file_idx] = 1.0
                    file_idx += 1

        return board

    def fen_to_board_matrix(self, fen: str) -> np.ndarray:
        """
        Convert FEN string to 8x8 piece matrix.

        Args:
            fen: FEN string

        Returns:
            Array of shape (8, 8) with piece indices
        """
        board_part = fen.split(' ')[0]
        board = np.zeros((8, 8), dtype=np.int32)

        ranks = board_part.split('/')
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    piece_idx = self.piece_to_idx.get(char, 0)
                    board[rank_idx, file_idx] = piece_idx
                    file_idx += 1

        return board

    def encode_sequence(self, fen_sequence: List[str]) -> Tuple[torch.Tensor, int]:
        """
        Encode a sequence of FEN strings.

        Args:
            fen_sequence: List of FEN strings

        Returns:
            Tuple of (sequence_tensor, actual_length)
            - sequence_tensor: Shape (max_seq_len, 13, 8, 8)
            - actual_length: Actual number of positions in sequence
        """
        actual_length = min(len(fen_sequence), self.max_sequence_length)

        # Create tensor with padding
        sequence = torch.zeros(
            self.max_sequence_length, self.board_channels, 8, 8,
            dtype=torch.float32
        )

        # Encode positions
        for i, fen in enumerate(fen_sequence[:self.max_sequence_length]):
            try:
                sequence[i] = self.fen_to_board_tensor(fen)
            except Exception:
                # Skip invalid FEN strings
                pass

        return sequence, actual_length

    def extract_fen_features(self, fen: str) -> dict:
        """
        Extract additional features from FEN string.

        Args:
            fen: FEN string

        Returns:
            Dictionary with features:
            - piece_count: Total number of pieces on board
            - white_pieces: Number of white pieces
            - black_pieces: Number of black pieces
            - to_move: 1 if white to move, 0 if black
        """
        parts = fen.split(' ')
        board_part = parts[0]
        to_move = 1.0 if parts[1] == 'w' else 0.0

        # Count pieces
        piece_count = 0
        white_count = 0
        black_count = 0

        for char in board_part:
            if char in self.piece_to_idx and char != '.':
                piece_count += 1
                if char.isupper():
                    white_count += 1
                else:
                    black_count += 1

        return {
            'piece_count': piece_count,
            'white_pieces': white_count,
            'black_pieces': black_count,
            'to_move': to_move,
        }

    def normalize_elo(self, elo: int, min_elo: int = 800, max_elo: int = 2800) -> float:
        """
        Normalize Elo rating to [0, 1] range.

        Args:
            elo: Elo rating
            min_elo: Minimum Elo (default: 800)
            max_elo: Maximum Elo (default: 2800)

        Returns:
            Normalized Elo rating
        """
        elo = max(min_elo, min(max_elo, elo))  # Clamp to range
        return (elo - min_elo) / (max_elo - min_elo)

    def denormalize_elo(self, normalized: float, min_elo: int = 800, max_elo: int = 2800) -> int:
        """
        Denormalize Elo rating from [0, 1] back to original scale.

        Args:
            normalized: Normalized Elo rating
            min_elo: Minimum Elo
            max_elo: Maximum Elo

        Returns:
            Denormalized Elo rating
        """
        return int(normalized * (max_elo - min_elo) + min_elo)
