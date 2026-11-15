"""Hybrid ML + DL model for Phase 2 (stacking a tree on a deep-model estimate).

Hybrid A: a deep sequence model maps each window to a RUL estimate `d`. A tree
(Phase 1 Random Forest or Gradient Boosting settings) predicts RUL from the
Phase 1 rolling features plus `d`. The training values of `d` are out-of-fold
by engine: every training row is predicted by a model that never saw its engine.

Hybrid B is the plain mean of a DL prediction and a tree prediction. Hybrid C
lets the tree predict the residual `y - d`.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from torch import nn

try:
    import dl_models
    import features
    import sequences
    import train
    from evaluation import VAL_ENGINES, mae, rmse
except ImportError:
    from src import dl_models, features, sequences, train
    from src.evaluation import VAL_ENGINES, mae, rmse

KEYS = ["engine_id", "cycle"]
FEATURE_GROUPS = ("rolling", "dl")  # Phase 1 rolling features, DL estimate(s)

RUL_BANDS = (0, 50, 100, 150, 200, np.inf)  # true-RUL bands for the signed-error check

# Phase 1 tree settings, copied from notebooks/04_tree_models.ipynb (not retuned).
TREE_PARAMS: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 300, "min_samples_leaf": 20, "max_features": 0.33, "n_jobs": -1,
    },
    "gradient_boosting": {
        "learning_rate": 0.05, "max_iter": 200, "max_depth": 3, "early_stopping": False,
    },
}


@dataclass
class FoldModel:
    """One fold's DL model, its scaler and the engines it was allowed to see."""

    model: nn.Module
    scaler: Tuple[np.ndarray, np.ndarray]
    fit_engines: np.ndarray  # engines whose windows were used for gradient steps
    stop_engines: np.ndarray  # engines used only for early stopping
    holdout_engines: np.ndarray  # engines this model predicts out-of-fold
    history: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def seen_engines(self) -> np.ndarray:
        """Every engine that influenced this model (fit or early stopping)."""
        return np.union1d(self.fit_engines, self.stop_engines)


def engine_folds(engine_ids: Sequence[int], k: int = 5, seed: int = 42) -> List[np.ndarray]:
    """Split the unique engine ids into `k` disjoint folds (random, seeded, by engine)."""
    engines = np.unique(np.asarray(engine_ids))
    if not 2 <= k <= len(engines):
        raise ValueError("k must be between 2 and the number of engines")
    shuffled = np.random.default_rng(seed).permutation(engines)
    return [np.sort(fold) for fold in np.array_split(shuffled, k)]


def split_fit_stop(
    engines: Sequence[int], n_stop: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a fold's training engines into (fit engines, early-stopping engines)."""
    engines = np.unique(np.asarray(engines))
    if not 1 <= n_stop < len(engines):
        raise ValueError("n_stop must be at least 1 and leave engines to fit on")
    stop = np.random.default_rng(seed).choice(engines, size=n_stop, replace=False)
    return np.setdiff1d(engines, stop), np.sort(stop)


def train_dl(
    kind: str,
    model_kwargs: Dict[str, Any],
    fit_df: pd.DataFrame,
    stop_df: pd.DataFrame,
    window: int = sequences.DEFAULT_WINDOW,
    seed: int = 42,
    fit_kwargs: Dict[str, Any] | None = None,
) -> Tuple[nn.Module, Tuple[np.ndarray, np.ndarray], Dict[str, List[float]]]:
    """Train one DL model on `fit_df`, early-stopping on `stop_df`, on CPU.

    The scaler is fitted on `fit_df` only. Returns (model, scaler, history).
    """
    scaler = sequences.fit_scaler(fit_df)
    X_fit, y_fit, _ = sequences.make_windows(fit_df, scaler, window)
    X_stop, y_stop, _ = sequences.make_windows(stop_df, scaler, window)
    train.set_seed(seed)  # before building: initial weights depend on the seed
    model = dl_models.build_model(kind, **model_kwargs)
    history = train.fit(
        model, X_fit, y_fit, X_stop, y_stop,
        seed=seed, device=torch.device("cpu"), **(fit_kwargs or {}),
    )
    return model.eval(), scaler, history


def dl_predict(
    model: nn.Module,
    scaler: Tuple[np.ndarray, np.ndarray],
    df: pd.DataFrame,
    window: int = sequences.DEFAULT_WINDOW,
) -> pd.DataFrame:
    """DL prediction `d` (cycles) for every row of `df`, keyed by engine_id and cycle."""
    X, _, keys = sequences.make_windows(df, scaler, window)
    out = keys.copy()
    out["d"] = train.predict(model, X)
    return out


def oof_predictions(
    train_df: pd.DataFrame,
    kind: str,
    model_kwargs: Dict[str, Any],
    window: int = sequences.DEFAULT_WINDOW,
    k: int = 5,
    seed: int = 42,
    n_stop: int = 8,
    fit_kwargs: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, List[FoldModel]]:
    """Out-of-fold DL predictions for every row of the training engines.

    Engines are split into `k` folds. For each fold a fresh model is trained on
    the other folds (minus `n_stop` engines kept for early stopping) and predicts
    the held-out fold. Returns (oof, fold_models): `oof` has engine_id, cycle,
    fold and d, sorted by engine_id and cycle.
    """
    if "rul" not in train_df:
        raise ValueError("train_df must be labelled training data (test data has no rul)")
    if train_df["engine_id"].isin(VAL_ENGINES).any():
        raise ValueError("validation engines must not be used to build out-of-fold predictions")
    folds = engine_folds(train_df["engine_id"], k=k, seed=seed)
    parts, fold_models = [], []
    for i, holdout in enumerate(folds):
        rest = np.setdiff1d(np.concatenate(folds), holdout)
        fit_engines, stop_engines = split_fit_stop(rest, n_stop, seed=seed + i)
        model, scaler, history = train_dl(
            kind, model_kwargs,
            train_df[train_df["engine_id"].isin(fit_engines)],
            train_df[train_df["engine_id"].isin(stop_engines)],
            window=window, seed=seed, fit_kwargs=fit_kwargs,
        )
        part = dl_predict(model, scaler, train_df[train_df["engine_id"].isin(holdout)], window)
        part.insert(2, "fold", i)
        parts.append(part)
        fold_models.append(
            FoldModel(model, scaler, fit_engines, stop_engines, holdout, history)
        )
    oof = pd.concat(parts, ignore_index=True).sort_values(KEYS, ignore_index=True)
    assert len(oof) == len(train_df) and not oof.duplicated(KEYS).any()
    return oof, fold_models


def fold_mean_predict(
    fold_models: Sequence[FoldModel],
    df: pd.DataFrame,
    window: int = sequences.DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Mean of the fold models' predictions `d` for every row of unseen engines `df`."""
    preds = [dl_predict(fm.model, fm.scaler, df, window) for fm in fold_models]
    for p in preds[1:]:
        assert p[KEYS].equals(preds[0][KEYS])
    out = preds[0][KEYS].copy()
    out["d"] = np.mean([p["d"].to_numpy() for p in preds], axis=0)
    return out


def build_hybrid_features(
    df: pd.DataFrame,
    dl_prediction: Optional[pd.DataFrame],
    groups: Sequence[str] = FEATURE_GROUPS,
    points: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Hybrid input table: Phase 1 rolling features and/or DL estimates, one row per key.

    `df` holds raw rows of whole engines (rolling features are computed here, per
    engine, from current and earlier cycles). `dl_prediction` has engine_id,
    cycle and one or more DL columns (every column except the keys and `fold`).
    Rows are joined on (engine_id, cycle), never by position. Returns the feature
    columns only, indexed by (engine_id, cycle), in the order of `points` if given,
    else of `df` sorted by engine_id and cycle. (`cycle` is also a rolling feature.)
    """
    unknown = set(groups) - set(FEATURE_GROUPS)
    if not groups or unknown:
        raise ValueError(f"groups must be a non-empty subset of {FEATURE_GROUPS}")
    rows = features.add_rolling_features(df.sort_values(KEYS)).reset_index(drop=True)
    if points is not None:
        rows = points[KEYS].merge(rows, on=KEYS, how="left", sort=False)
        if rows[features.FEATURES].isna().any().any():
            raise ValueError("some requested points are not in df")
    columns = list(features.FEATURES) if "rolling" in groups else []
    if "dl" in groups:
        if dl_prediction is None:
            raise ValueError("the dl group needs dl_prediction")
        dl_cols = [c for c in dl_prediction.columns if c not in KEYS + ["fold"]]
        rows = rows.merge(
            dl_prediction[KEYS + dl_cols], on=KEYS, how="left", sort=False,
            validate="one_to_one",
        )
        if rows[dl_cols].isna().any().any():
            raise ValueError("dl_prediction is missing some (engine_id, cycle) rows")
        columns += [c for c in dl_cols if c not in columns]
    out = rows.set_index(KEYS, drop=False)[columns]
    expected = points[KEYS] if points is not None else df.sort_values(KEYS)[KEYS]
    assert out.index.equals(pd.MultiIndex.from_frame(expected)), "row alignment broken"
    return out


def make_tree(kind: str = "random_forest", seed: int = 42) -> RegressorMixin:
    """Unfitted tree regressor with the Phase 1 hyperparameters and `seed`."""
    if kind == "random_forest":
        return RandomForestRegressor(random_state=seed, **TREE_PARAMS[kind])
    if kind == "gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, **TREE_PARAMS[kind])
    raise ValueError(f"unknown tree: {kind}")


@dataclass
class HybridModel:
    """A fitted tree on hybrid features; `residual` means it predicts y - d."""

    tree: RegressorMixin
    columns: List[str]
    residual: bool = False
    d_column: str = "d"


def fit_hybrid(
    X: pd.DataFrame,
    y: np.ndarray,
    tree: str = "random_forest",
    residual: bool = False,
    seed: int = 42,
    d_column: str = "d",
) -> HybridModel:
    """Fit the tree on the hybrid table `X` (from `build_hybrid_features`).

    Hybrid A: `residual=False`, the tree predicts RUL. Hybrid C: `residual=True`,
    the tree predicts the DL error `y - X[d_column]`. `d` must be out-of-fold.
    """
    y = np.asarray(y, dtype=float)
    if len(y) != len(X):
        raise ValueError("X and y must have the same number of rows")
    if residual and d_column not in X:
        raise ValueError(f"residual hybrid needs the DL column {d_column!r}")
    target = y - X[d_column].to_numpy() if residual else y
    model = make_tree(tree, seed)
    model.fit(X, target)
    return HybridModel(model, list(X.columns), residual, d_column)


def predict_hybrid(model: HybridModel, X: pd.DataFrame) -> np.ndarray:
    """RUL predictions in cycles, clipped at 0."""
    if list(X.columns) != model.columns:
        raise ValueError("X must have the same columns, in the same order, as in training")
    pred = model.tree.predict(X)
    if model.residual:
        pred = pred + X[model.d_column].to_numpy()
    return np.clip(pred, 0, None)


def average_hybrid(dl_pred: np.ndarray, tree_pred: np.ndarray) -> np.ndarray:
    """Hybrid B: the plain mean of a DL and a tree prediction (no fitted weights)."""
    dl_pred, tree_pred = np.asarray(dl_pred, dtype=float), np.asarray(tree_pred, dtype=float)
    if dl_pred.shape != tree_pred.shape:
        raise ValueError("predictions must have the same shape")
    return (dl_pred + tree_pred) / 2


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, and MAE of the early (true RUL > 125) and late (<= 125) points."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    early = y_true > train.EARLY_LATE_SPLIT
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "early_mae": mae(y_true[early], y_pred[early]) if early.any() else float("nan"),
        "late_mae": mae(y_true[~early], y_pred[~early]) if (~early).any() else float("nan"),
    }


def band_errors(
    y_true: np.ndarray, predictions: Dict[str, np.ndarray], bands: Sequence[float] = RUL_BANDS
) -> pd.DataFrame:
    """Mean signed error (prediction - truth, cycles) per true-RUL band, with counts.

    Bands are left-closed: [0, 50), [50, 100), ..., [200, inf). Negative means the
    model underpredicts the remaining life.
    """
    y_true = np.asarray(y_true, dtype=float)
    labels = [
        f"{lo:g}+" if np.isinf(hi) else f"{lo:g}-{hi:g}" for lo, hi in zip(bands[:-1], bands[1:])
    ]
    band = pd.cut(y_true, bins=list(bands), right=False, labels=labels)
    table = pd.DataFrame({"band": band})
    for name, pred in predictions.items():
        table[name] = np.asarray(pred, dtype=float) - y_true
    out = table.groupby("band", observed=False).mean()
    out.insert(0, "n", table.groupby("band", observed=False).size())
    return out.reset_index()
