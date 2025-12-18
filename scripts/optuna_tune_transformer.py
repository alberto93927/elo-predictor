"""
Optuna-based hyperparameter tuning for the Transformer model.

This script calls `train_transformer_optimized.main` programmatically with
sampled hyperparameters and uses validation loss as the objective.
"""

import argparse
import optuna

# Ensure local imports work when running as a script
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_transformer_optimized import main as train_main  # noqa: E402


def build_objective(
    max_epochs: int,
    patience: int,
    base_batch_size: int,
    model_prefix_root: str,
):
    """Create an Optuna objective function that returns validation loss."""

    def objective(trial: optuna.Trial):
        # Hyperparameter search space
        dropout = trial.suggest_float("dropout", 0.15, 0.35)
        weight_decay = trial.suggest_float("weight_decay", 5e-4, 1e-3, log=True)
        lr = trial.suggest_float("lr", 3e-5, 8e-5, log=True)
        num_heads = trial.suggest_categorical("num_heads", [8, 12])
        feedforward_dim = trial.suggest_categorical(
            "feedforward_dim", [896, 1024, 1152, 1280]
        )
        warmup_epochs = trial.suggest_int("warmup_epochs", 4, 8)

        # Unique prefix per trial to avoid collisions
        model_prefix = f"{model_prefix_root}_trial{trial.number}"

        # Build argument list for the training script
        args = [
            "--model-type",
            "transformer",
            "--epochs",
            str(max_epochs),
            "--early-stopping-patience",
            str(patience),
            "--lr",
            str(lr),
            "--embedding-dim",
            "256",
            "--num-layers",
            "6",
            "--num-heads",
            str(num_heads),
            "--feedforward-dim",
            str(feedforward_dim),
            "--dropout",
            str(dropout),
            "--weight-decay",
            str(weight_decay),
            "--warmup-epochs",
            str(warmup_epochs),
            "--use-cosine-schedule",
            "--batch-size",
            str(base_batch_size),
            "--model-prefix",
            model_prefix,
        ]

        result = train_main(args)
        if not isinstance(result, dict):
            raise optuna.TrialPruned("Training did not return metrics.")

        # Prefer best_val_loss; fallback to eval_val_loss
        val_loss = result.get("best_val_loss")
        if val_loss is None:
            val_loss = result.get("eval_val_loss")
        if val_loss is None:
            raise optuna.TrialPruned("No validation loss available.")

        return float(val_loss)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for Transformer model")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=15,
        help="Max epochs per trial (keep small for speed)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience per trial",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per GPU for tuning trials",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="transformer_optuna",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db). If None, uses in-memory.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="minimize",
        choices=["minimize", "maximize"],
        help="Optimization direction (default: minimize val loss)",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "none"],
        help="Pruner type (default: median).",
    )

    args = parser.parse_args()

    study_kwargs = {
        "study_name": args.study_name,
        "direction": args.direction,
    }
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True

    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    else:
        pruner = optuna.pruners.NopPruner()

    study_kwargs["pruner"] = pruner

    study = optuna.create_study(**study_kwargs)

    objective = build_objective(
        max_epochs=args.max_epochs,
        patience=args.patience,
        base_batch_size=args.batch_size,
        model_prefix_root=args.study_name,
    )

    study.optimize(objective, n_trials=args.trials)

    print("Best trial:")
    print(f"  Value (val_loss): {study.best_trial.value}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
"""
Lightweight Optuna tuner for the Transformer training script.

Runs train_transformer_optimized.py in subprocesses with sampled hyperparameters
and reads val_loss from the generated metrics.csv. Designed to be simple and
require no new dependencies beyond optuna.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional

import optuna


def read_best_val_loss(run_dir: Path) -> Optional[float]:
    """Return the minimum val_loss from metrics.csv in run_dir, if present."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    best = None
    with metrics_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "val_loss" not in row or row["val_loss"] in ("", "nan", "NaN"):
                continue
            try:
                v = float(row["val_loss"])
            except Exception:
                continue
            best = v if best is None else min(best, v)
    return best


def find_latest_run_dir(model_prefix: str) -> Optional[Path]:
    """Find the newest logs/* directory matching prefix."""
    logs_root = Path("logs")
    if not logs_root.exists():
        return None
    candidates = sorted(
        [p for p in logs_root.glob(f"{model_prefix}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def make_cmd(args, trial, trial_params):
    train_script = Path(__file__).parent / "train_transformer_optimized.py"
    model_prefix = f"{args.model_prefix_base}_trial{trial.number}"

    cmd = [
        sys.executable,
        str(train_script),
        "--model-type",
        "transformer",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--model-prefix",
        model_prefix,
        "--embedding-dim",
        str(trial_params["embedding_dim"]),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(trial_params["num_heads"]),
        "--feedforward-dim",
        str(trial_params["feedforward_dim"]),
        "--dropout",
        str(trial_params["dropout"]),
        "--weight-decay",
        str(trial_params["weight_decay"]),
        "--lr",
        str(trial_params["lr"]),
        "--warmup-epochs",
        str(trial_params["warmup_epochs"]),
        "--max-seq-len",
        str(args.max_seq_len),
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--parquet-dir",
        args.parquet_dir,
        "--data-dir",
        args.data_dir,
        "--shuffle-buffer",
        str(args.shuffle_buffer),
        "--use-cosine-schedule",
    ]
    if args.use_mixed_precision:
        cmd.append("--use-mixed-precision")
    if args.use_black_elo:
        cmd.append("--use-black-elo")
    if args.no_parquet:
        cmd.append("--no-parquet")
    return cmd, model_prefix


def objective(trial, args):
    # Search space (kept tight to limit run count/time)
    params = {
        "dropout": trial.suggest_float("dropout", 0.2, 0.35, step=0.05),
        "weight_decay": trial.suggest_float("weight_decay", 5e-4, 1e-3, step=5e-4),
        "lr": trial.suggest_categorical("lr", [5e-5, 3e-5]),
        "num_heads": trial.suggest_categorical("num_heads", [8, 12]),
        "feedforward_dim": trial.suggest_categorical("feedforward_dim", [1024, 1280]),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [256, 320]),
        "warmup_epochs": trial.suggest_categorical("warmup_epochs", [5, 8]),
    }

    cmd, model_prefix = make_cmd(args, trial, params)
    print(f"[trial {trial.number}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(
            f"[trial {trial.number}] Training failed (exit {result.returncode}); "
            f"stderr:\n{result.stderr.decode(errors='ignore')[-2000:]}"
        )
        return float("inf")

    run_dir = find_latest_run_dir(model_prefix)
    if run_dir is None:
        print(f"[trial {trial.number}] No run dir found for prefix {model_prefix}")
        return float("inf")

    best_val = read_best_val_loss(run_dir)
    if best_val is None:
        print(f"[trial {trial.number}] No val_loss found in {run_dir}")
        return float("inf")

    print(f"[trial {trial.number}] best val_loss={best_val:.6f} @ {run_dir}")
    return best_val


def main():
    parser = argparse.ArgumentParser(description="Optuna sweep for transformer.")
    parser.add_argument("--trials", type=int, default=8, help="Number of trials.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per trial.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-layers", type=int, default=6, help="Transformer layers.")
    parser.add_argument("--max-seq-len", type=int, default=200, help="Max seq len.")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--parquet-dir", type=str, default="data/parquet")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--model-prefix-base", type=str, default="optuna_transformer")
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--use-mixed-precision", action="store_true")
    parser.add_argument("--use-black-elo", action="store_true")
    parser.add_argument("--no-parquet", action="store_true")

    args = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args), n_trials=args.trials)

    print("\n=== Optuna summary ===")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()


