"""Sliding-window sequences for the Phase 2 deep learning models.

Each sample is the last `window` cycles of one engine, ending at a chosen
cycle. Only current and earlier cycles are used, so there is no look-ahead.
Engines with fewer than `window` cycles so far are left-padded by repeating
their first observed row.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:
    from load_data import SENSORS
except ImportError:
    from src.load_data import SENSORS

DEFAULT_WINDOW = 30
FEATURES = list(SENSORS)  # 14 useful sensors; settings are not used


def fit_scaler(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sensor mean and std. Fit on training engines only."""
    values = train_df[FEATURES].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _engine_windows(values: np.ndarray, window: int) -> np.ndarray:
    """All windows of one engine, one per cycle, left-padded. Shape (T, window, F)."""
    padded = np.concatenate([np.repeat(values[:1], window - 1, axis=0), values])
    windows = sliding_window_view(padded, window, axis=0)  # (T, F, window)
    return windows.transpose(0, 2, 1)


def make_windows(
    df: pd.DataFrame,
    scaler: Tuple[np.ndarray, np.ndarray],
    window: int = DEFAULT_WINDOW,
    points: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """Build windows for every row of `df`, or only for `points`.

    `points` is a dataframe with `engine_id` and `cycle` (for example the 100
    fixed validation points). Returns (X, y, keys):
      X    float32 array (N, window, 14), scaled with `scaler`
      y    float32 RUL array (N,), or None when `df` has no `rul` column
      keys dataframe of engine_id and cycle, row-aligned with X
    """
    mean, std = scaler
    df = df.sort_values(["engine_id", "cycle"])
    xs, ys, keys = [], [], []
    for engine_id, g in df.groupby("engine_id", sort=True):
        values = ((g[FEATURES].to_numpy(dtype=np.float32) - mean) / std).astype(
            np.float32
        )
        xs.append(_engine_windows(values, window))
        keys.append(g[["engine_id", "cycle"]])
        if "rul" in g:
            ys.append(g["rul"].to_numpy(dtype=np.float32))
    X = np.concatenate(xs)
    keys = pd.concat(keys, ignore_index=True)
    y = np.concatenate(ys) if ys else None

    if points is not None:
        keys["_pos"] = np.arange(len(keys))
        order = points[["engine_id", "cycle"]].merge(keys, on=["engine_id", "cycle"], how="left")
        if order["_pos"].isna().any():
            raise ValueError("some requested points are not in the dataframe")
        pos = order["_pos"].to_numpy(dtype=int)
        X = X[pos]
        y = y[pos] if y is not None else None
        keys = keys.drop(columns="_pos").iloc[pos].reset_index(drop=True)
    return X, y, keys
