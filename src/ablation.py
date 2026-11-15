"""Ablation helpers for notebook 06.

Every run follows the project protocol: the scaler is fitted on the 80 training
engines, training uses all their windows, early stopping uses all windows of the
20 validation engines, and the 100 fixed validation points are only scored after
training. Runs are trained on CPU (deterministic) and cached one row per run in a
CSV, so an interrupted notebook can resume and a finished one re-executes quickly.
"""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

try:
    import train
    from dl_models import build_model
    from evaluation import split_train_val, validation_points
    from load_data import load_train
    from rnn_experiments import get_data
except ImportError:
    from src import train
    from src.dl_models import build_model
    from src.evaluation import split_train_val, validation_points
    from src.load_data import load_train
    from src.rnn_experiments import get_data

KEYS = ["ablation", "setting", "seed"]
SETTING_COLUMNS = ["setting_1", "setting_2"]  # the two operating settings that vary in FD001
FEATURE_SETS = ("sensors", "settings", "diff")
RESULT_COLUMNS = [
    "ablation", "setting", "seed", "val_window_mae", "point_mae", "point_rmse",
    "early_mae", "late_mae", "epochs_run", "source",
]


def fit_settings_scaler(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std of the operating settings, fitted on training engines only."""
    values = train_df[SETTING_COLUMNS].to_numpy(dtype=np.float32)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return values.mean(axis=0), std


def settings_windows(
    df: pd.DataFrame,
    scaler: Tuple[np.ndarray, np.ndarray],
    window: int,
    points: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Windows of the standardised operating settings, shape (N, window, 2).

    Rows follow `make_windows`: sorted by engine_id and cycle, left-padded with the
    first row of each engine, restricted to `points` (in that order) if given.
    """
    mean, std = scaler
    df = df.sort_values(["engine_id", "cycle"])
    blocks, keys = [], []
    for _, g in df.groupby("engine_id", sort=True):
        values = ((g[SETTING_COLUMNS].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
        padded = np.concatenate([np.repeat(values[:1], window - 1, axis=0), values])
        blocks.append(sliding_window_view(padded, window, axis=0).transpose(0, 2, 1))
        keys.append(g[["engine_id", "cycle"]])
    out = np.concatenate(blocks)
    if points is not None:
        keys = pd.concat(keys, ignore_index=True)
        keys["_pos"] = np.arange(len(keys))
        order = points[["engine_id", "cycle"]].merge(keys, on=["engine_id", "cycle"], how="left")
        if order["_pos"].isna().any():
            raise ValueError("some requested points are not in the dataframe")
        out = out[order["_pos"].to_numpy(dtype=int)]
    return out


def add_diff_features(X: np.ndarray) -> np.ndarray:
    """Append first-difference channels (zero at the first step): (N, window, 2 * features)."""
    diff = np.diff(X, axis=1, prepend=X[:, :1, :])
    return np.concatenate([X, diff], axis=2)


@lru_cache(maxsize=None)
def get_arrays(window: int, features: str) -> Dict[str, Any]:
    """Training, validation-engine and 100-point arrays for a window and feature set."""
    if features not in FEATURE_SETS:
        raise ValueError(f"features must be one of {FEATURE_SETS}")
    data = dict(get_data(window))
    if features == "sensors":
        return data
    if features == "diff":
        for name in ("X_tr", "X_va", "X_pt"):
            data[name] = add_diff_features(data[name])
        return data
    train_df, val_df = split_train_val(load_train())
    scaler = fit_settings_scaler(train_df)
    extra = {
        "X_tr": settings_windows(train_df, scaler, window),
        "X_va": settings_windows(val_df, scaler, window),
        "X_pt": settings_windows(val_df, scaler, window, points=validation_points(val_df)),
    }
    for name, add in extra.items():
        assert len(add) == len(data[name]), name
        data[name] = np.concatenate([data[name], add], axis=2)
    return data


def run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Train one configuration and return one result row (see RESULT_COLUMNS).

    `cfg` needs ablation, setting, seed, model and kw (model constructor arguments);
    optional window (30), features ("sensors"), lr, weight_decay, rul_cap, patience.
    """
    window, features = cfg.get("window", 30), cfg.get("features", "sensors")
    data = get_arrays(window, features)
    train.set_seed(cfg["seed"])  # before building: initial weights depend on the seed
    model = build_model(cfg["model"], n_features=data["X_tr"].shape[2], **cfg["kw"])
    hist = train.fit(
        model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"],
        lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4),
        patience=cfg.get("patience", 8), seed=cfg["seed"], device=torch.device("cpu"),
        rul_cap=cfg.get("rul_cap"),
    )
    points = train.evaluate_points(model, data["X_pt"], data["y_pt"])
    return {
        "ablation": cfg["ablation"], "setting": cfg["setting"], "seed": cfg["seed"],
        "val_window_mae": hist["val_mae"][hist["best_epoch"][0] - 1],
        "point_mae": points["mae"], "point_rmse": points["rmse"],
        "early_mae": points["early_mae"], "late_mae": points["late_mae"],
        "epochs_run": hist["epochs_run"][0], "source": "run",
    }


def _init_worker() -> None:
    torch.set_num_threads(1)


def run_missing(
    configs: Sequence[Dict[str, Any]], cache: Path, workers: int = 6
) -> pd.DataFrame:
    """Run the configs not yet in the cache CSV, append them, return the full table."""
    done = pd.read_csv(cache) if Path(cache).exists() else pd.DataFrame(columns=RESULT_COLUMNS)
    have = set(map(tuple, done[KEYS].to_numpy().tolist()))
    todo = [c for c in configs if (c["ablation"], c["setting"], c["seed"]) not in have]
    if todo:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            rows = list(pool.map(run_one, todo))
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)[RESULT_COLUMNS]
        done.to_csv(cache, index=False)
    return done


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std (ddof=1, NaN for one seed) per ablation setting, for both scores."""
    grouped = results.groupby(["ablation", "setting"], sort=False)
    out = grouped.agg(
        n_seeds=("seed", "nunique"),
        point_mae_mean=("point_mae", "mean"), point_mae_std=("point_mae", lambda s: s.std(ddof=1)),
        point_rmse_mean=("point_rmse", "mean"), point_rmse_std=("point_rmse", lambda s: s.std(ddof=1)),
        val_window_mae_mean=("val_window_mae", "mean"),
        val_window_mae_std=("val_window_mae", lambda s: s.std(ddof=1)),
    )
    return out.reset_index()
