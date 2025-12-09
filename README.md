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

#### Platform-Specific Training Commands

**macOS (Apple Silicon - M1/M2/M3)**

```bash
# Transformer (recommended)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_model.py \
  --model transformer \
  --epochs 30 \
  --batch-size 32

# LSTM baseline
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_model.py \
  --model lstm \
  --epochs 30 \
  --batch-size 32
```

**Windows/Linux (NVIDIA GPU)**

```bash
# Transformer (recommended)
python scripts/train_model.py \
  --model transformer \
  --epochs 30 \
  --batch-size 32 \
  --device cuda

# LSTM baseline
python scripts/train_model.py \
  --model lstm \
  --epochs 30 \
  --batch-size 32 \
  --device cuda
```

**Any Platform (CPU-only)**

```bash
# Transformer (slower but works everywhere)
python scripts/train_model.py \
  --model transformer \
  --epochs 20 \
  --batch-size 16 \
  --device cpu

# LSTM baseline (more CPU-friendly)
python scripts/train_model.py \
  --model lstm \
  --epochs 20 \
  --batch-size 32 \
  --device cpu
```

#### Expected Training Times (Full Dataset: ~121K games)

| Setup | Transformer | LSTM |
|-------|-------------|------|
| **M2 Pro (MPS)** | ~85 min/epoch<br>30 epochs: ~42 hours | ~4-5 hours/epoch<br>30 epochs: ~5 days |
| **NVIDIA GPU (CUDA)** | ~60-90 min/epoch<br>30 epochs: ~30-45 hours | ~3-4 hours/epoch<br>30 epochs: ~4 days |
| **8-core CPU** | ~8-10 hours/epoch<br>20 epochs: ~7 days | ~3-5 hours/epoch<br>20 epochs: ~4 days |

**Recommendation:** Start with 10-20 epochs to see convergence, then decide if more epochs are needed based on validation loss curve.

#### Training Options

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
- `--device`: "cuda", "cpu", or auto-detect
- `--output-dir`: Directory for results (default: experiments/)
- `--early-stopping`: Enable early stopping (default: True)

### 4. Verify Training Pipeline (Quick Test)

Before running full training, verify that both models work on your system:

#### macOS / Linux

```bash
# Quick pipeline test (~2 minutes total for both models)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/test_pipeline.py
```

#### Windows (Command Prompt)

```cmd
# Quick pipeline test (~2 minutes total for both models)
python scripts/test_pipeline.py
```

#### Windows (PowerShell)

```powershell
# Quick pipeline test (~2 minutes total for both models)
$env:PYTORCH_ENABLE_MPS_FALLBACK="1"
python scripts/test_pipeline.py
```

**What this tests:**
- Data loading from preprocessed files
- Transformer model training (2 epochs on 500 games)
- LSTM model training (2 epochs on 500 games)
- Forward/backward passes
- Validation loop
- MPS (Apple Silicon) / CUDA (NVIDIA) / CPU compatibility

**Expected output:**
```
============================================================
PIPELINE TEST SUMMARY
============================================================

✓ Transformer: PASSED
  - Final validation MAE: ~100 Elo
  - Training completed without errors

✓ LSTM: PASSED
  - Final validation MAE: ~100 Elo
  - Training completed without errors

============================================================
✓✓✓ ALL PIPELINE TESTS PASSED ✓✓✓
============================================================
```

**Timing:**
- **macOS M2 Pro (MPS GPU):** ~1-2 minutes total
- **Windows/Linux (CUDA GPU):** ~1-2 minutes total
- **CPU-only:** ~5-10 minutes total

**Platform Notes:**

| Platform | Accelerator | Notes |
|----------|-------------|-------|
| macOS (Apple Silicon) | MPS | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for transformer attention |
| Windows/Linux (NVIDIA GPU) | CUDA | Auto-detected if available |
| Windows/Linux (AMD GPU) | ROCm | May require specific PyTorch build |
| Any (CPU-only) | CPU | Works everywhere, ~5x slower |

### 5. Run Unit Tests

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

- **Minimum:** CPU only (slow but functional)
- **Recommended:** GPU with 4GB+ VRAM (NVIDIA CUDA or Apple MPS)
- **Available:** ORCA cluster with 192GB VRAM, 64 CPU cores, 576GB RAM

The project was tested on similar scales and scales down gracefully if needed.

## Cross-Platform Compatibility

### Tested Platforms

| Platform | Hardware | Status | Notes |
|----------|----------|--------|-------|
| macOS (Apple Silicon) | M2 Pro, 16GB RAM | ✅ Fully tested | Requires `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Windows (NVIDIA GPU) | RTX series | ✅ Compatible | Auto-detects CUDA |
| Linux (NVIDIA GPU) | Various | ✅ Compatible | Auto-detects CUDA |
| Windows/Linux (CPU) | 8+ cores | ✅ Compatible | Slower but works |
| macOS (Intel) | Intel CPU | ✅ Compatible | CPU-only mode |

### Troubleshooting

**Issue: "MPS backend not available" on macOS**
```bash
# Your Mac doesn't have Apple Silicon, use CPU instead
python scripts/train_model.py --model transformer --device cpu
```

**Issue: "CUDA not available" on Windows/Linux with NVIDIA GPU**
```bash
# Check if PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Issue: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"**
- This was fixed in the current version. Make sure you have the latest code.
- If you cloned earlier, do `git pull` to get the fixes.

**Issue: Training is very slow**
```bash
# Check which device is being used
# Should show "mps", "cuda", or "cpu"

# For Apple Silicon, make sure MPS is enabled:
python -c "import torch; print(torch.backends.mps.is_available())"

# For NVIDIA, check CUDA:
python -c "import torch; print(torch.cuda.is_available())"
```

**Issue: Out of memory errors**
```bash
# Reduce batch size
python scripts/train_model.py --model transformer --batch-size 16

# Or use smaller model on CPU
python scripts/train_model.py --model lstm --batch-size 8 --device cpu
```

### Windows-Specific Notes

- **Environment Variables:** Use PowerShell syntax for setting env vars:
  ```powershell
  $env:PYTORCH_ENABLE_MPS_FALLBACK="1"
  ```

- **Path Separators:** Python handles this automatically, no changes needed

- **Virtual Environment Activation:**
  ```cmd
  # Command Prompt
  venv\Scripts\activate

  # PowerShell
  venv\Scripts\Activate.ps1
  ```

### Performance Expectations

The models work identically across platforms. Only the speed differs based on available hardware acceleration:

- **GPU (MPS/CUDA):** Fast training (~1-2 hours/epoch for Transformer)
- **CPU:** Slower but reliable (~8-10 hours/epoch for Transformer)
- **Results:** Same model quality regardless of platform

## References

- Lichess Database: https://lichess.org/database
- Transformer: "Attention is All You Need" (Vaswani et al., 2017)
- python-chess: https://python-chess.readthedocs.io/

## Team

- Logan Druley
- Alberto Garcia
- Roberto Palacios

Course: CST 463 - Final Project, CSUMB Fall 2025
