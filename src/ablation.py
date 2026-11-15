"""Ablation study helpers — Vikash Kumar.

Extra feature channels and ablation utilities.  Does not edit sequences.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

try:
    from sequences import fit_scaler, make_windows, DEFAULT_WINDOW, FEATURES
    from evaluation import mae, rmse, validation_points, split_train_val
    from load_data import load_train, SENSORS
    from dl_models import build_model, N_FEATURES
    from train import set_seed, fit, predict, get_device, RUL_SCALE
except ImportError:
    from src.sequences import fit_scaler, make_windows, DEFAULT_WINDOW, FEATURES
    from src.evaluation import mae, rmse, validation_points, split_train_val
    from src.load_data import load_train, SENSORS
    from src.dl_models import build_model, N_FEATURES
    from src.train import set_seed, fit, predict, get_device, RUL_SCALE

THRESHOLD: int = 125


def add_settings_features(
    X: np.ndarray,
    df: pd.DataFrame,
    scaler,
    window: int,
    points: pd.DataFrame | None = None,
) -> np.ndarray:
    """Append setting_1 and setting_2 channels to existing windows.

    Returns X with shape (N, window, n_features+2).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from sklearn.preprocessing import StandardScaler

    setting_cols = ["setting_1", "setting_2"]
    s_scaler = StandardScaler()
    train = load_train()
    train_df, _ = split_train_val(train)
    s_scaler.fit(train_df[setting_cols])

    df = df.sort_values(["engine_id", "cycle"])
    xs, keys = [], []
    for eid, g in df.groupby("engine_id", sort=True):
        values = s_scaler.transform(g[setting_cols]).astype(np.float32)
        padded = np.concatenate([np.repeat(values[:1], window - 1, axis=0), values])
        win = sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)
        xs.append(win)
        keys.append(g[["engine_id", "cycle"]])

    X_settings = np.concatenate(xs)
    keys = pd.concat(keys, ignore_index=True)

    if points is not None:
        keys["_pos"] = np.arange(len(keys))
        order = points[["engine_id", "cycle"]].merge(
            keys, on=["engine_id", "cycle"], how="left", sort=False
        )
        pos = order["_pos"].to_numpy(dtype=int)
        X_settings = X_settings[pos]

    return np.concatenate([X, X_settings], axis=2)


def add_diff_features(X: np.ndarray) -> np.ndarray:
    """Append first-difference channels.  Returns (N, window, 2*features)."""
    diff = np.diff(X, axis=1, prepend=X[:, :1, :])
    return np.concatenate([X, diff], axis=2)


def run_ablation_single(
    model_name: str,
    model_kwargs: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 42,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    patience: int = 8,
    rul_cap: float | None = None,
) -> dict[str, Any]:
    """Train one model config and return validation metrics.

    y_tr and y_val are both in **cycles** (as returned by make_windows).
    fit() expects y_val in cycles; predict() also returns cycles.
    """
    device = get_device()
    set_seed(seed)
    model = build_model(model_name, **model_kwargs)

    history = fit(
        model, X_tr, y_tr, X_val, y_val,
        lr=lr, weight_decay=weight_decay, epochs=epochs,
        patience=patience, seed=seed, device=device,
        rul_cap=rul_cap,
    )
    preds = predict(model, X_val)

    overall_mae = mae(y_val, preds)
    overall_rmse = rmse(y_val, preds)

    early = y_val > THRESHOLD
    late = ~early
    early_mae = mae(y_val[early], preds[early]) if early.sum() > 0 else np.nan
    late_mae = mae(y_val[late], preds[late]) if late.sum() > 0 else np.nan

    return {
        "point_mae": overall_mae,
        "point_rmse": overall_rmse,
        "early_mae": early_mae,
        "late_mae": late_mae,
        "epochs_run": history["epochs_run"][0],
        "seed": seed,
    }
