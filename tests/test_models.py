"""
Unit tests for model architectures.
"""

import pytest
import torch
from src.models.transformer import TransformerEncoder
from src.models.lstm import LSTMEloPredictor


class TestTransformerEncoder:
    """Test Transformer model."""

    def setup_method(self):
        """Setup model for each test."""
        self.model = TransformerEncoder(
            input_channels=13,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
            feedforward_dim=128,
        )
        self.model.eval()

    def test_forward_pass(self):
        """Test forward pass through model."""
        batch_size, seq_len = 4, 50

        sequences = torch.randn(batch_size, seq_len, 13, 8, 8)
        lengths = torch.tensor([50, 45, 30, 50])

        predictions, _ = self.model(sequences, lengths)

        assert predictions.shape == (batch_size, 1)
        assert torch.all((predictions >= 0) & (predictions <= 1))  # Sigmoid output

    def test_forward_without_lengths(self):
        """Test forward pass without length information."""
        batch_size, seq_len = 2, 50

        sequences = torch.randn(batch_size, seq_len, 13, 8, 8)

        predictions, _ = self.model(sequences, lengths=None)

        assert predictions.shape == (batch_size, 1)

    def test_attention_weights(self):
        """Test getting attention weights."""
        batch_size, seq_len = 2, 50

        sequences = torch.randn(batch_size, seq_len, 13, 8, 8)
        lengths = torch.tensor([50, 30])

        attention = self.model.get_attention_weights(sequences, lengths)

        assert attention.shape == (batch_size, seq_len)
        assert torch.all(attention >= 0)

    def test_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        param_count = sum(p.numel() for p in self.model.parameters())
        assert param_count > 1000  # Should have meaningful number of params


class TestLSTMEloPredictor:
    """Test LSTM baseline model."""

    def setup_method(self):
        """Setup model for each test."""
        self.model = LSTMEloPredictor(
            input_channels=13,
            embedding_dim=64,
            lstm_hidden_dim=128,
            num_lstm_layers=2,
        )
        self.model.eval()

    def test_forward_pass(self):
        """Test forward pass through model."""
        batch_size, seq_len = 4, 50

        sequences = torch.randn(batch_size, seq_len, 13, 8, 8)
        lengths = torch.tensor([50, 45, 30, 50])

        predictions, outputs = self.model(sequences, lengths)

        assert predictions.shape == (batch_size, 1)
        assert torch.all((predictions >= 0) & (predictions <= 1))  # Sigmoid output
        assert outputs.shape[0] == batch_size
        assert outputs.shape[1] == seq_len

    def test_forward_without_lengths(self):
        """Test forward pass without length information."""
        batch_size, seq_len = 2, 50

        sequences = torch.randn(batch_size, seq_len, 13, 8, 8)

        predictions, _ = self.model(sequences, lengths=None)

        assert predictions.shape == (batch_size, 1)

    def test_attention_weights(self):
        """Test getting attention-like weights."""
        batch_size, seq_len = 2, 50

        sequences = torch.randn(batch_size, seq_len, 13, 8, 8)
        lengths = torch.tensor([50, 30])

        attention = self.model.get_attention_weights(sequences, lengths)

        assert attention.shape == (batch_size, seq_len)
        assert torch.all(attention >= 0)

    def test_bidirectional(self):
        """Test bidirectional LSTM."""
        model = LSTMEloPredictor(
            input_channels=13,
            embedding_dim=64,
            lstm_hidden_dim=128,
            bidirectional=True,
        )
        model.eval()

        batch_size, seq_len = 2, 50
        sequences = torch.randn(batch_size, seq_len, 13, 8, 8)

        predictions, _ = model(sequences)

        assert predictions.shape == (batch_size, 1)

    def test_unidirectional(self):
        """Test unidirectional LSTM."""
        model = LSTMEloPredictor(
            input_channels=13,
            embedding_dim=64,
            lstm_hidden_dim=128,
            bidirectional=False,
        )
        model.eval()

        batch_size, seq_len = 2, 50
        sequences = torch.randn(batch_size, seq_len, 13, 8, 8)

        predictions, _ = model(sequences)

        assert predictions.shape == (batch_size, 1)

    def test_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        param_count = sum(p.numel() for p in self.model.parameters())
        assert param_count > 1000  # Should have meaningful number of params


class TestModelOutputRange:
    """Test that both models output predictions in [0, 1] range."""

    def test_transformer_output_range(self):
        """Test transformer outputs are in valid range."""
        model = TransformerEncoder(embedding_dim=64, num_layers=2)
        model.eval()

        with torch.no_grad():
            sequences = torch.randn(8, 100, 13, 8, 8)
            predictions, _ = model(sequences)

            assert torch.all(predictions >= 0)
            assert torch.all(predictions <= 1)

    def test_lstm_output_range(self):
        """Test LSTM outputs are in valid range."""
        model = LSTMEloPredictor(embedding_dim=64)
        model.eval()

        with torch.no_grad():
            sequences = torch.randn(8, 100, 13, 8, 8)
            predictions, _ = model(sequences)

            assert torch.all(predictions >= 0)
            assert torch.all(predictions <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
