"""
Unit tests for PGN parser.
"""

import pytest
from src.data.parser import (
    parse_pgn_game, extract_white_moves, extract_black_moves
)


class TestPGNParser:
    """Test PGN parsing functionality."""

    def test_parse_valid_pgn(self):
        """Test parsing a valid PGN game."""
        pgn_text = """[Event "Test"]
[Site "Online"]
[Date "2023.01.01"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1400"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 Na5 9. Bc2 c5 10. d4 Qc7 1-0"""

        result = parse_pgn_game(pgn_text)
        assert result is not None

        fen_sequence, white_elo, black_elo, game_result = result

        assert white_elo == 1600
        assert black_elo == 1400
        assert game_result == "1-0"
        assert len(fen_sequence) > 0
        assert isinstance(fen_sequence[0], str)

    def test_parse_missing_elo(self):
        """Test that games without Elo ratings are skipped."""
        pgn_text = """[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3"""

        result = parse_pgn_game(pgn_text)
        assert result is None

    def test_parse_invalid_elo(self):
        """Test that games with invalid Elo ratings are skipped."""
        pgn_text = """[Event "Test"]
[WhiteElo "invalid"]
[BlackElo "1400"]
[Result "1-0"]

1. e4 e5"""

        result = parse_pgn_game(pgn_text)
        assert result is None

    def test_parse_out_of_range_elo(self):
        """Test that games with out-of-range Elo are skipped."""
        pgn_text = """[Event "Test"]
[WhiteElo "50"]
[BlackElo "1400"]
[Result "1-0"]

1. e4 e5"""

        result = parse_pgn_game(pgn_text)
        assert result is None

    def test_extract_white_moves(self):
        """Test extracting white's moves from FEN sequence."""
        fen_sequence = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Initial
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # After white move
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # After black move
        ]

        white_moves = extract_white_moves(fen_sequence)
        assert len(white_moves) == 2
        assert white_moves[0] == fen_sequence[0]
        assert white_moves[1] == fen_sequence[2]

    def test_extract_black_moves(self):
        """Test extracting black's moves from FEN sequence."""
        fen_sequence = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ]

        black_moves = extract_black_moves(fen_sequence)
        assert len(black_moves) == 1
        assert black_moves[0] == fen_sequence[1]

    def test_short_game_skipped(self):
        """Test that games with very few moves are skipped."""
        pgn_text = """[Event "Test"]
[WhiteElo "1600"]
[BlackElo "1400"]
[Result "1-0"]

1. e4"""

        result = parse_pgn_game(pgn_text)
        assert result is None or len(result[0]) < 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
