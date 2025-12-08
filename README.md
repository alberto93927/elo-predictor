# Chess Elo Rating Prediction via Machine Learning

Predict chess player Elo ratings from game move sequences using deep learning models (Transformer Encoder and LSTM baseline).

## Project Overview

This project develops machine learning models that can predict a chess player's Elo rating by analyzing the sequence of board positions (FEN strings) from their games. The model enables:

- **Instant skill assessment** from a single game without requiring multiple rated games
- **Player matchmaking** based on predicted skill level
- **Dynamic AI difficulty adjustment** by segmenting players into difficulty tiers
- **Analysis of skill indicators** through model attention weights

**Target Performance:** 100-200 Elo point prediction error (represents a moderate but manageable skill gap)

## Data

**Dataset:** Lichess database of rated chess games in PGN format
- Path: `data/raw/lichess_db_standard_rated_2013-01.pgn.zst`
- Contains millions of games with move sequences and rating information
- Games are parsed and converted to FEN strings for model input

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── parser.py          # PGN parsing and game extraction
│   │   ├── encoder.py         # FEN to tensor encoding
│   │   └── dataset.py         # PyTorch Dataset classes
│   ├── models/
│   │   ├── transformer.py     # Transformer Encoder architecture
│   │   └── lstm.py            # LSTM baseline model
│   ├── train.py               # Training loop and Trainer class
│   ├── evaluate.py            # Evaluation and visualization utilities
│   └── utils.py               # Helper functions (checkpointing, metrics, etc.)
├── scripts/
│   ├── preprocess_data.py     # Extract and parse PGN dataset
│   └── train_model.py         # Main training script
├── tests/
│   ├── test_parser.py         # PGN parser tests
│   ├── test_encoder.py        # FEN encoder tests
│   └── test_models.py         # Model architecture tests
├── notebooks/
│   └── exploration.ipynb      # Exploratory data analysis
├── data/
│   ├── raw/                   # Original compressed PGN file
│   └── processed/             # Parsed game splits
├── models/                    # Saved model checkpoints
├── experiments/               # Training results and configs
├── requirements.txt           # Python dependencies
├── setup.py                   # Package configuration
└── CLAUDE.md                  # Development guide
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

The `pip install -e .` step is important - it installs the package so Python scripts can import from `src/` without path issues.

### 2. Preprocess Data

The dataset is already in `data/raw/`. Extract and parse it:

```bash
python scripts/preprocess_data.py --input data/raw/lichess_db_standard_rated_2013-01.pgn.zst
```

This will:
- Decompress the zstd-compressed PGN file
- Parse games and extract move sequences
- Convert moves to FEN strings
- Split into train/val/test sets
- Save processed data to `data/processed/`

The script shows a progress bar with:
- Games processed
- Processing speed (games/second)
- Estimated time remaining

**Timing Estimate (M2 Pro, 16GB):**
- 100,000 games: ~30-60 seconds
- 1,000,000 games: ~5-10 minutes
- 5,000,000+ games: ~30-60+ minutes

**Start small for testing:**
```bash
# Process only first 10,000 games for quick testing
python scripts/preprocess_data.py --max-games 10000

# Or first 100,000 to see realistic timing
python scripts/preprocess_data.py --max-games 100000

# Custom output directory
python scripts/preprocess_data.py --output custom_data/
```

### 3. Train Model

#### Train Transformer (recommended)

```bash
python scripts/train_model.py --model transformer --epochs 50 --batch-size 32
```

#### Train LSTM baseline

```bash
python scripts/train_model.py --model lstm --epochs 50 --batch-size 32
```

#### Training options

```bash
python scripts/train_model.py --help
```

Key arguments:
- `--model`: "transformer" or "lstm"
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--dropout`: Dropout rate (default: 0.1)
- `--embedding-dim`: Embedding dimension (default: 128)
- `--device`: "cuda" or "cpu"
- `--output-dir`: Directory for results (default: experiments/)

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_encoder.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Model Architectures

### Transformer Encoder (Primary Model)

- **Input:** Variable-length sequence of board positions (max 200)
- **Architecture:**
  - CNN board embedding (13 channels → 128 dimensions)
  - Positional encoding for sequence positions
  - Multi-head self-attention (8 heads, 4 layers)
  - Feedforward layers
  - Global average pooling
  - Output head for Elo prediction

- **Advantages:**
  - Parallel processing of sequences (efficient)
  - Self-attention reveals which positions are skill indicators
  - Can capture long-range dependencies

### LSTM Baseline

- **Input:** Same as Transformer
- **Architecture:**
  - CNN board embedding (13 channels → 128 dimensions)
  - Bidirectional LSTM (256 hidden, 2 layers)
  - Final hidden state from both directions
  - Output head for Elo prediction

- **Purpose:**
  - Baseline comparison with traditional RNN approach
  - Validate that Transformer provides real improvements
  - Simpler sequential processing model

## Key Features

### FEN Encoding

Each board position is encoded as:
- **13-channel representation:** 12 piece types + 1 empty square channel
- **Shape:** (13, 8, 8) per position
- **Processing:** CNN reduces to fixed-size embedding

### Elo Normalization

- **Range:** 800-2800 Elo
- **Normalized:** [0, 1] for training
- **Denormalization:** Convert predictions back to Elo points for interpretability

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Primary metric in Elo points
- **RMSE:** Penalizes larger errors
- **R² Score:** Variance explained
- **Segmentation:** Performance breakdown by Elo difficulty level

### Early Stopping & Checkpointing

- Save best model based on validation loss
- Early stopping after 5 epochs of no improvement
- Learning rate scheduling (reduce on plateau)

## Attention Analysis

The Transformer's attention weights can be extracted to identify:
- Which board positions were most important for prediction
- Which moves indicate skill level
- How the model uses move sequences to assess skill

```python
from src.models.transformer import TransformerEncoder

model = TransformerEncoder(...)
attention_weights = model.get_attention_weights(sequences, lengths)
# Shape: (batch_size, sequence_length)
```

## Configuration & Hyperparameters

Default hyperparameters are in `scripts/train_model.py`. Common choices:

- **Embedding Dimension:** 128-256
- **Number of Layers:** 2-4
- **Number of Heads:** 4-8
- **Feedforward Dimension:** 2x embedding dimension
- **Dropout:** 0.1-0.2
- **Learning Rate:** 1e-3 to 1e-4
- **Batch Size:** 32-64

## Common Tasks

### Compare Model Performance

```bash
# Train both models and compare results
python scripts/train_model.py --model transformer --output-dir exp/transformer
python scripts/train_model.py --model lstm --output-dir exp/lstm

# Check logs
cat exp/transformer/training_log.txt
cat exp/lstm/training_log.txt
```

### Debug Data Issues

Use the Jupyter notebook in `notebooks/` for interactive exploration:

```bash
jupyter notebook notebooks/exploration.ipynb
```

### Extract Model Predictions

```python
import torch
from src.models.transformer import TransformerEncoder
from src.data.encoder import FENEncoder

model = TransformerEncoder()
model.load_state_dict(torch.load("models/best_checkpoint.pt"))

# Use on new games
predictions, _ = model(sequences, lengths)
```

## Computational Requirements

- **Minimum:** CPU only (slow)
- **Recommended:** GPU with 4GB+ VRAM
- **Available:** ORCA cluster with 192GB VRAM, 64 CPU cores, 576GB RAM

The project was tested on similar scales and scales down gracefully if needed.

## References

- Lichess Database: https://lichess.org/database
- Transformer: "Attention is All You Need" (Vaswani et al., 2017)
- python-chess: https://python-chess.readthedocs.io/

## Team

- Logan Druley
- Alberto Garcia
- Roberto Palacios

Course: CST 463 - Final Project, CSUMB Fall 2025
