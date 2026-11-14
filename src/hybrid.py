"""Hybrid ML + DL model for Phase 2 (stacking a tree on a deep-model estimate).

Hybrid A: a deep sequence model maps each window to a RUL estimate `d`. A tree
(Phase 1 Random Forest or Gradient Boosting settings) predicts RUL from the
Phase 1 rolling features plus `d`. The training values of `d` are out-of-fold
by engine: every training row is predicted by a model that never saw its engine.

Hybrid B is the plain mean of a DL prediction and a tree prediction. Hybrid C
lets the tree predict the residual `y - d`.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import dl_models
    import sequences
    import train
    from evaluation import VAL_ENGINES
except ImportError:
    from src import dl_models, sequences, train
    from src.evaluation import VAL_ENGINES

KEYS = ["engine_id", "cycle"]


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
