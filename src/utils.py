"""
utils.py
Shared helpers used across every component: config loading, global seeding,
and the data-leakage guard mandated by the project guidelines.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load the single source of truth for all hyperparameters."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for python, numpy and (if available) torch, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return PROJECT_ROOT / relative_path


class LeakageGuardError(RuntimeError):
    """Raised when an evaluation-only column is about to be used as a model input."""


def guard_against_leakage(df: pd.DataFrame, feature_columns: list[str], config: dict) -> None:
    """
    Project guideline #1: summary_ref, entities_ref and mis_risk_label must
    NEVER be used as input features. Call this before building any feature
    matrix / tokenizer input.
    """
    leakage_columns = set(config.get("leakage_columns", []))
    used = leakage_columns.intersection(set(feature_columns))
    if used:
        raise LeakageGuardError(
            f"Refusing to build features: {sorted(used)} are evaluation-only "
            f"columns and must not be used as model inputs."
        )


def strip_leakage_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return a copy of df with all leakage columns removed, if present."""
    leakage_columns = [c for c in config.get("leakage_columns", []) if c in df.columns]
    return df.drop(columns=leakage_columns)
