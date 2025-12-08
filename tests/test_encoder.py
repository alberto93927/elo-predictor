"""
Unit tests for FEN encoder.
"""

import pytest
import torch
import numpy as np
from src.data.encoder import FENEncoder


class TestFENEncoder:
    """Test FEN encoding functionality."""

    def setup_method(self):
        """Setup encoder for each test."""
        self.encoder = FENEncoder(max_sequence_length=200)

    def test_fen_to_board_tensor(self):
        """Test converting FEN string to tensor."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        board_tensor = self.encoder.fen_to_board_tensor(fen)

        assert board_tensor.shape == (13, 8, 8)
        assert board_tensor.dtype == torch.float32
        assert board_tensor.sum() > 0  # Should have pieces

    def test_fen_to_board_matrix(self):
        """Test converting FEN string to piece matrix."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        board_matrix = self.encoder.fen_to_board_matrix(fen)

        assert board_matrix.shape == (8, 8)
        assert board_matrix.dtype == np.int32
        # Check some known pieces
        assert board_matrix[0, 0] == self.encoder.piece_to_idx['r']  # Black rook
        assert board_matrix[7, 0] == self.encoder.piece_to_idx['R']  # White rook

    def test_encode_sequence(self):
        """Test encoding a sequence of FEN strings."""
        fen_sequence = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ]

        sequence, length = self.encoder.encode_sequence(fen_sequence)

        assert sequence.shape == (200, 13, 8, 8)
        assert length == 3
        assert sequence[0].sum() > 0  # First position encoded
        assert sequence[1].sum() > 0  # Second position encoded
        assert sequence[4].sum() == 0  # Padded positions are zero

    def test_encode_long_sequence(self):
        """Test encoding a sequence longer than max_length."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen_sequence = [fen] * 250  # 250 positions, max is 200

        sequence, length = self.encoder.encode_sequence(fen_sequence)

        assert sequence.shape == (200, 13, 8, 8)
        assert length == 200  # Should be capped at max_length

    def test_extract_fen_features(self):
        """Test extracting features from FEN string."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        features = self.encoder.extract_fen_features(fen)

        assert "piece_count" in features
        assert "white_pieces" in features
        assert "black_pieces" in features
        assert "to_move" in features

        assert features["piece_count"] == 32
        assert features["white_pieces"] == 16
        assert features["black_pieces"] == 16
        assert features["to_move"] == 1.0  # White to move

    def test_normalize_denormalize_elo(self):
        """Test Elo normalization and denormalization."""
        original_elo = 1600

        normalized = self.encoder.normalize_elo(original_elo)
        denormalized = self.encoder.denormalize_elo(normalized)

        assert 0 <= normalized <= 1
        assert abs(denormalized - original_elo) < 2  # Allow small rounding error

    def test_elo_clamping(self):
        """Test that Elo ratings are clamped to valid range."""
        low_elo = 500  # Below minimum
        high_elo = 3000  # Above maximum

        normalized_low = self.encoder.normalize_elo(low_elo)
        normalized_high = self.encoder.normalize_elo(high_elo)

        assert normalized_low >= 0
        assert normalized_high <= 1

    def test_empty_sequence(self):
        """Test encoding empty sequence."""
        empty_sequence = []

        sequence, length = self.encoder.encode_sequence(empty_sequence)

        assert sequence.shape == (200, 13, 8, 8)
        assert length == 0
        assert sequence.sum() == 0  # All zeros


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
