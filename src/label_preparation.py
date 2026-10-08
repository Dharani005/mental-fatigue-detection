"""
Stage 3: Label Preparation Module.

Formats target variable for both continuous regression (1–7 Karolinska score)
and binary classification fallback (0 = Not Fatigued [1–4], 1 = Fatigued [5–7]).
Analyzes target distribution and determines modeling strategy.
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from src.feature_engineering import build_feature_table



def prepare_labels(df_features: pd.DataFrame, binary_threshold: int = 5) -> dict:
    """
    Prepare regression labels and binary classification labels from feature table.

    Args:
        df_features: DataFrame extracted from feature engineering.
        binary_threshold: Score threshold for binary fatigue classification (>= threshold is Fatigued).

    Returns:
        Dict containing:
        - X: Feature matrix (excluding identifiers, fatigue score, and difficulty tier)
        - y_reg: Continuous regression target (1-7 scale)
        - y_clf: Binary classification target (0 or 1)
        - feature_names: List of predictor feature names
        - label_stats: Dict of distribution metrics
    """
    if df_features.empty:
        raise ValueError("Feature table is empty.")

    # Ground truth continuous rating (1-7)
    y_reg = df_features["fatigue_score"].values.astype(float)

    # Binary classification target: 0 for alert (1-4), 1 for fatigued (5-7)
    y_clf = (y_reg >= binary_threshold).astype(int)

    # Define predictor column list — EXCLUDE identifiers, target, and controlled variable difficulty_tier
    exclude_cols = [
        "session_id",
        "window_checkpoint",
        "window_id",
        "fatigue_score",
        "difficulty_tier"  # Controlled variable, excluded from prediction as per prompt
    ]

    predictor_cols = [c for c in df_features.columns if c not in exclude_cols]
    X = df_features[predictor_cols].copy()

    # Target stats & distribution summary
    reg_counts = pd.Series(y_reg).value_counts().sort_index().to_dict()
    clf_counts = pd.Series(y_clf).value_counts().to_dict()

    label_stats = {
        "total_samples": len(y_reg),
        "reg_min": float(np.min(y_reg)),
        "reg_max": float(np.max(y_reg)),
        "reg_mean": float(np.mean(y_reg)),
        "reg_std": float(np.std(y_reg)),
        "reg_distribution": reg_counts,
        "clf_0_not_fatigued": clf_counts.get(0, 0),
        "clf_1_fatigued": clf_counts.get(1, 0),
        "imbalance_ratio": float(clf_counts.get(1, 0) / len(y_clf)) if len(y_clf) > 0 else 0.0
    }

    return {
        "X": X,
        "y_reg": y_reg,
        "y_clf": y_clf,
        "feature_names": predictor_cols,
        "label_stats": label_stats
    }


def print_label_report(label_data: dict):
    """Print label distribution and balance metrics."""
    stats = label_data["label_stats"]
    print("=" * 60)
    print("STAGE 3 — LABEL PREPARATION REPORT")
    print("=" * 60)
    print(f"Total Window Samples: {stats['total_samples']}")
    print(f"Continuous Fatigue Rating Range: [{stats['reg_min']}, {stats['reg_max']}]")
    print(f"Mean Rating: {stats['reg_mean']:.2f} ± {stats['reg_std']:.2f}")
    print("\n[Continuous Scale Distribution (1–7)]:")
    for k, v in sorted(stats["reg_distribution"].items()):
        print(f"  Score {int(k)}: {v} samples ({v / stats['total_samples'] * 100:.1f}%)")

    print("\n[Binary Classification Fallback Distribution]:")
    print(f"  Class 0 (Not Fatigued [1-4]): {stats['clf_0_not_fatigued']} samples ({stats['clf_0_not_fatigued'] / stats['total_samples'] * 100:.1f}%)")
    print(f"  Class 1 (Fatigued [5-7]):     {stats['clf_1_fatigued']} samples ({stats['clf_1_fatigued'] / stats['total_samples'] * 100:.1f}%)")
    print(f"  Fatigued Proportion:         {stats['imbalance_ratio'] * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    df_feat = build_feature_table(DEFAULT_DATA_DIR)
    label_info = prepare_labels(df_feat)
    print_label_report(label_info)
