"""Normalization utilities for waypoint VLA RLDS data.

Supports q99 (quantile) and normal (z-score) normalization,
with optional per-dimension masks.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def normalize_q99(
    values: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map [q01, q99] -> [-1, 1], clip outliers, zero constant dims."""
    eps = 1e-8
    normed = np.clip(2 * (values - q01) / (q99 - q01 + eps) - 1, -1, 1)
    if mask is not None:
        normed = np.where(mask, normed, values)
    normed = np.where(np.equal(q01, q99), 0.0, normed)
    return normed.astype(np.float32)


def unnormalize_q99(
    values: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse of normalize_q99: [-1,1] -> [q01, q99]."""
    values_01 = 0.5 * (values + 1)  # [-1,1] -> [0,1]
    unnormed = values_01 * (q99 - q01) + q01
    if mask is not None:
        unnormed = np.where(mask, unnormed, values)
    return unnormed.astype(np.float32)


def normalize_normal(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Z-score normalization: (x - mean) / (std + eps)."""
    eps = 1e-8
    normed = (values - mean) / (std + eps)
    if mask is not None:
        normed = np.where(mask, normed, values)
    return normed.astype(np.float32)


def unnormalize_normal(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse of z-score normalization."""
    eps = 1e-8
    unnormed = values * (std + eps) + mean
    if mask is not None:
        unnormed = np.where(mask, unnormed, values)
    return unnormed.astype(np.float32)


def pad_to_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad the last dimension of arr to target_dim."""
    actual = arr.shape[-1]
    if actual >= target_dim:
        return arr[..., :target_dim]
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_dim - actual)]
    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


def make_dim_mask(actual_dim: int, model_dim: int) -> np.ndarray:
    """Create boolean mask: True for real dims, False for padding."""
    return np.array(
        [True] * actual_dim + [False] * (model_dim - actual_dim),
        dtype=bool,
    )


def extract_proprio_from_obs(step_obs: dict, state_obs_keys: list[str]) -> np.ndarray:
    """Extract and concatenate proprio from an RLDS observation dict."""
    parts = []
    for key in state_obs_keys:
        val = step_obs[key]
        if hasattr(val, "numpy"):
            val = val.numpy()
        val = np.asarray(val, dtype=np.float32).flatten()
        parts.append(val)
    return np.concatenate(parts)


class NormalizationHelper:
    """Wraps normalization statistics and provides normalize/unnormalize methods."""

    def __init__(self, dataset_statistics: dict[str, Any], norm_type: str = "q99"):
        self.norm_type = norm_type
        stats = self._find_stats(dataset_statistics)

        self.action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
        self.action_std = np.array(stats["action"]["std"], dtype=np.float32)
        self.action_q01 = np.array(stats["action"]["q01"], dtype=np.float32)
        self.action_q99 = np.array(stats["action"]["q99"], dtype=np.float32)
        self.action_norm_mask = np.array(
            stats["action"].get("mask", np.ones_like(self.action_mean, dtype=bool)),
            dtype=bool,
        )

        self.proprio_mean = np.array(stats["proprio"]["mean"], dtype=np.float32)
        self.proprio_std = np.array(stats["proprio"]["std"], dtype=np.float32)
        self.proprio_q01 = np.array(stats["proprio"]["q01"], dtype=np.float32)
        self.proprio_q99 = np.array(stats["proprio"]["q99"], dtype=np.float32)

    @staticmethod
    def _find_stats(dataset_statistics: dict) -> dict:
        for key, val in dataset_statistics.items():
            if key != "__total__" and isinstance(val, dict) and "action" in val:
                return val
        raise ValueError("Cannot find action/proprio stats in dataset_statistics")

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.norm_type in ("q99", "bounds_q99"):
            return normalize_q99(actions, self.action_q01, self.action_q99, self.action_norm_mask)
        return normalize_normal(actions, self.action_mean, self.action_std, self.action_norm_mask)

    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.norm_type in ("q99", "bounds_q99"):
            return unnormalize_q99(actions, self.action_q01, self.action_q99, self.action_norm_mask)
        return unnormalize_normal(actions, self.action_mean, self.action_std, self.action_norm_mask)

    def normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        if self.norm_type in ("q99", "bounds_q99"):
            return normalize_q99(proprio, self.proprio_q01, self.proprio_q99)
        return normalize_normal(proprio, self.proprio_mean, self.proprio_std)

    def unnormalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        if self.norm_type in ("q99", "bounds_q99"):
            return unnormalize_q99(proprio, self.proprio_q01, self.proprio_q99)
        return unnormalize_normal(proprio, self.proprio_mean, self.proprio_std)


def load_dataset_statistics(path: str | Path) -> dict:
    """Load dataset statistics JSON from a file path."""
    path = Path(path)
    if path.is_dir():
        candidates = list(path.glob("dataset_statistics*.json"))
        if not candidates:
            raise FileNotFoundError(f"No dataset_statistics*.json found in {path}")
        path = candidates[0]
    with open(path) as f:
        return json.load(f)
