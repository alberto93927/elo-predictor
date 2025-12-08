"""
Evaluation and analysis utilities for Elo prediction models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json


class ModelEvaluator:
    """Evaluate model performance and create visualizations."""

    def __init__(self, encoder=None):
        """
        Initialize evaluator.

        Args:
            encoder: FENEncoder for denormalization
        """
        self.encoder = encoder

    def evaluate_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict:
        """
        Evaluate model predictions.

        Args:
            predictions: Model predictions (normalized)
            targets: Target values (normalized)

        Returns:
            Dictionary with evaluation metrics
        """
        # Normalized metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)

        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        # Denormalize to Elo points if encoder available
        if self.encoder is not None:
            pred_elo = np.array([
                self.encoder.denormalize_elo(p) for p in predictions
            ])
            target_elo = np.array([
                self.encoder.denormalize_elo(t) for t in targets
            ])

            mae_elo = mean_absolute_error(target_elo, pred_elo)
            rmse_elo = np.sqrt(mean_squared_error(target_elo, pred_elo))

            metrics["mae_elo"] = mae_elo
            metrics["rmse_elo"] = rmse_elo

        return metrics

    def segment_by_elo_range(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict:
        """
        Segment evaluation by Elo rating ranges.

        Args:
            predictions: Model predictions (normalized)
            targets: Target values (normalized)
            ranges: List of (min_elo, max_elo) tuples. If None, uses defaults.

        Returns:
            Dictionary with metrics for each Elo range
        """
        if self.encoder is None:
            raise ValueError("Encoder required for Elo range segmentation")

        # Default ranges: beginner, intermediate, advanced, master
        if ranges is None:
            ranges = [
                (800, 1200, "Beginner"),
                (1200, 1600, "Intermediate"),
                (1600, 2000, "Advanced"),
                (2000, 2800, "Master"),
            ]

        pred_elo = np.array([
            self.encoder.denormalize_elo(p) for p in predictions
        ])
        target_elo = np.array([
            self.encoder.denormalize_elo(t) for t in targets
        ])

        segmentation = {}

        for min_elo, max_elo, label in ranges:
            mask = (target_elo >= min_elo) & (target_elo < max_elo)

            if mask.sum() == 0:
                continue

            seg_preds = pred_elo[mask]
            seg_targets = target_elo[mask]

            mae = mean_absolute_error(seg_targets, seg_preds)
            rmse = np.sqrt(mean_squared_error(seg_targets, seg_preds))

            segmentation[label] = {
                "range": f"{min_elo}-{max_elo}",
                "count": int(mask.sum()),
                "mae": mae,
                "rmse": rmse,
            }

        return segmentation

    def plot_predictions_vs_targets(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_path: Optional[str] = None,
    ):
        """
        Plot predictions vs targets.

        Args:
            predictions: Model predictions
            targets: Target values
            output_path: Path to save figure (optional)
        """
        if self.encoder is not None:
            pred_elo = np.array([
                self.encoder.denormalize_elo(p) for p in predictions
            ])
            target_elo = np.array([
                self.encoder.denormalize_elo(t) for t in targets
            ])
        else:
            pred_elo = predictions
            target_elo = targets

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(target_elo, pred_elo, alpha=0.5)

        # Perfect prediction line
        min_val = min(target_elo.min(), pred_elo.min())
        max_val = max(target_elo.max(), pred_elo.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        ax.set_xlabel("Target Elo", fontsize=12)
        ax.set_ylabel("Predicted Elo", fontsize=12)
        ax.set_title("Model Predictions vs Target Elo", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        return fig, ax

    def plot_residuals(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_path: Optional[str] = None,
    ):
        """
        Plot prediction residuals.

        Args:
            predictions: Model predictions
            targets: Target values
            output_path: Path to save figure (optional)
        """
        if self.encoder is not None:
            pred_elo = np.array([
                self.encoder.denormalize_elo(p) for p in predictions
            ])
            target_elo = np.array([
                self.encoder.denormalize_elo(t) for t in targets
            ])
        else:
            pred_elo = predictions
            target_elo = targets

        residuals = target_elo - pred_elo

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals vs target
        axes[0].scatter(target_elo, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel("Target Elo", fontsize=12)
        axes[0].set_ylabel("Residuals", fontsize=12)
        axes[0].set_title("Residuals vs Target Elo", fontsize=14)
        axes[0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel("Residuals", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title("Distribution of Residuals", fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        return fig, axes

    def extract_important_positions(
        self,
        model: nn.Module,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Extract important positions based on model attention.

        Args:
            model: Model with get_attention_weights method
            sequences: Input sequences
            lengths: Sequence lengths
            top_k: Number of top positions to return

        Returns:
            List of (position_index, importance_score) tuples
        """
        with torch.no_grad():
            attention_weights = model.get_attention_weights(sequences, lengths)

        # Average attention across batch
        avg_attention = attention_weights.mean(dim=0).cpu().numpy()

        # Get top k positions
        top_indices = np.argsort(-avg_attention)[:top_k]
        top_scores = avg_attention[top_indices]

        return list(zip(top_indices, top_scores))

    def save_evaluation_report(
        self,
        metrics: Dict,
        segmentation: Dict,
        output_path: str,
    ):
        """
        Save evaluation report to JSON.

        Args:
            metrics: Overall metrics
            segmentation: Segmentation by Elo range
            output_path: Path to save report
        """
        report = {
            "overall_metrics": {k: float(v) for k, v in metrics.items()},
            "segmentation": segmentation,
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Saved evaluation report to {output_path}")


def create_difficulty_segments(
    encoder,
    num_segments: int = 4,
    min_elo: int = 800,
    max_elo: int = 2800,
) -> List[Dict]:
    """
    Create difficulty segments for chess AI tuning.

    Args:
        encoder: FENEncoder for Elo conversion
        num_segments: Number of segments (e.g., easy, normal, hard, impossible)
        min_elo: Minimum Elo
        max_elo: Maximum Elo

    Returns:
        List of segment dictionaries with Elo ranges and labels
    """
    segment_labels = ["Easy", "Normal", "Hard", "Impossible"]
    segment_colors = ["green", "yellow", "orange", "red"]

    elo_range = max_elo - min_elo
    segment_size = elo_range / num_segments

    segments = []
    for i in range(num_segments):
        start_elo = int(min_elo + i * segment_size)
        end_elo = int(min_elo + (i + 1) * segment_size)

        segments.append({
            "difficulty": segment_labels[i] if i < len(segment_labels) else f"Level_{i+1}",
            "elo_range": f"{start_elo}-{end_elo}",
            "min_elo": start_elo,
            "max_elo": end_elo,
            "color": segment_colors[i] if i < len(segment_colors) else "blue",
        })

    return segments
