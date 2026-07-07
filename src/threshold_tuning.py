from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
    grid: tuple[float, float, float] = (0.05, 0.95, 0.01),
) -> dict[str, float]:
    """
    For each label independently, sweep a threshold grid and keep the value
    that maximises that label's F1 on the validation set. Returns a dict
    {label_name: optimal_threshold}.
    """
    start, stop, step = grid
    thresholds = np.arange(start, stop + 1e-9, step)
    optimal: dict[str, float] = {}

    for i, label in enumerate(label_names):
        best_f1, best_t = -1.0, 0.5
        col_true = y_true[:, i]
        col_proba = y_proba[:, i]
        if col_true.sum() == 0:
            # no positive examples for this label in val set — keep default
            optimal[label] = 0.5
            continue
        for t in thresholds:
            col_pred = (col_proba >= t).astype(int)
            f1 = f1_score(col_true, col_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        optimal[label] = best_t

    return optimal


def apply_thresholds(y_proba: np.ndarray, label_names: list[str], thresholds: dict[str, float]) -> np.ndarray:
    """Apply per-label thresholds to a probability matrix to get binary predictions."""
    preds = np.zeros_like(y_proba, dtype=int)
    for i, label in enumerate(label_names):
        t = thresholds.get(label, 0.5)
        preds[:, i] = (y_proba[:, i] >= t).astype(int)
    return preds
