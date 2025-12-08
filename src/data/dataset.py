"""
PyTorch Dataset for chess Elo prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from src.data.encoder import FENEncoder


class ChessEloDataset(Dataset):
    """Dataset for chess games with Elo ratings."""

    def __init__(
        self,
        games: List[Tuple[List[str], int, int, str]],
        encoder: Optional[FENEncoder] = None,
        max_sequence_length: int = 200,
        use_white_elo: bool = True,
        normalize_elo: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            games: List of (fen_sequence, white_elo, black_elo, result) tuples
            encoder: FENEncoder instance (creates new if None)
            max_sequence_length: Maximum sequence length
            use_white_elo: If True use white's Elo, else use black's
            normalize_elo: If True normalize Elo ratings to [0, 1]
        """
        self.games = games
        self.encoder = encoder or FENEncoder(max_sequence_length)
        self.use_white_elo = use_white_elo
        self.normalize_elo = normalize_elo

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.games)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single game sample.

        Args:
            idx: Game index

        Returns:
            Dictionary with:
            - 'sequence': Encoded board sequence (max_len, 13, 8, 8)
            - 'length': Actual sequence length
            - 'elo': Target Elo rating (normalized if applicable)
            - 'white_elo': White's Elo (original)
            - 'black_elo': Black's Elo (original)
            - 'result': Game result
        """
        fen_sequence, white_elo, black_elo, result = self.games[idx]

        # Encode sequence
        sequence, length = self.encoder.encode_sequence(fen_sequence)

        # Select target Elo
        target_elo = white_elo if self.use_white_elo else black_elo

        # Normalize if requested
        if self.normalize_elo:
            normalized_elo = self.encoder.normalize_elo(target_elo)
        else:
            normalized_elo = float(target_elo)

        return {
            'sequence': sequence,
            'length': length,
            'elo': normalized_elo,
            'white_elo': float(white_elo),
            'black_elo': float(black_elo),
            'result': result,
        }


def create_data_loaders(
    train_games: List,
    val_games: List,
    test_games: List,
    batch_size: int = 32,
    num_workers: int = 0,
    max_sequence_length: int = 200,
    use_white_elo: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, FENEncoder]:
    """
    Create train/val/test dataloaders.

    Args:
        train_games: Training games
        val_games: Validation games
        test_games: Test games
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_sequence_length: Maximum sequence length
        use_white_elo: Whether to use white or black Elo

    Returns:
        Tuple of (train_loader, val_loader, test_loader, encoder)
    """
    encoder = FENEncoder(max_sequence_length)

    train_dataset = ChessEloDataset(
        train_games,
        encoder,
        max_sequence_length,
        use_white_elo,
        normalize_elo=True,
    )
    val_dataset = ChessEloDataset(
        val_games,
        encoder,
        max_sequence_length,
        use_white_elo,
        normalize_elo=True,
    )
    test_dataset = ChessEloDataset(
        test_games,
        encoder,
        max_sequence_length,
        use_white_elo,
        normalize_elo=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, encoder
