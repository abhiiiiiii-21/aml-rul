"""Helpers to train and score one LSTM/GRU configuration (used by notebook 03).

`run_one` is a plain top-level function so that runs can be spread over worker
processes (`run_many`). Selection metrics use the 20 validation engines' windows.
The 100 fixed validation points and the test points are only scored or predicted
when `cfg["final"]` is true; no test metric is ever computed here.
"""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import time
from typing import Any, Dict, List

import numpy as np
import torch

from dl_models import build_model
from evaluation import split_train_val, test_points, validation_points
from load_data import load_test, load_train
from sequences import DEFAULT_WINDOW, fit_scaler, make_windows
import train

DEFAULTS: Dict[str, Any] = {
    "kind": "gru", "hidden": 64, "layers": 2, "dropout": 0.2,
    "weight_decay": 1e-4, "patience": 8, "epochs": 60, "seed": 42,
    "lr": 1e-3, "grad_clip": 1.0, "restore_best": True,
    "window": DEFAULT_WINDOW, "final": False,
}


@lru_cache(maxsize=None)
def get_data(window: int) -> Dict[str, Any]:
    """Windows for training engines, validation engines, 100 points, test points."""
    train_df, val_df = split_train_val(load_train())
    scaler = fit_scaler(train_df)  # training engines only
    X_tr, y_tr, _ = make_windows(train_df, scaler, window)
    X_va, y_va, _ = make_windows(val_df, scaler, window)
    pts = validation_points(val_df)
    X_pt, y_pt, keys_pt = make_windows(val_df, scaler, window, points=pts)
    test_df = load_test()
    X_te, _, keys_te = make_windows(test_df, scaler, window, points=test_points(test_df))
    return {
        "X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va,
        "X_pt": X_pt, "y_pt": y_pt, "keys_pt": keys_pt,
        "X_te": X_te, "keys_te": keys_te,
    }


def model_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Constructor arguments of `RNNRegressor` taken from a config."""
    return {k: cfg[k] for k in ("hidden", "layers", "dropout")}


def run_one(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Train one config on CPU and return validation-engine metrics and history."""
    cfg = {**DEFAULTS, **overrides}
    data = get_data(cfg["window"])
    device = torch.device("cpu")  # deterministic and as fast as MPS for this size
    train.set_seed(cfg["seed"])  # before building: initial weights depend on the seed
    model = build_model(cfg["kind"], **model_kwargs(cfg))
    start = time.perf_counter()
    hist = train.fit(
        model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"],
        epochs=cfg["epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        patience=cfg["patience"], seed=cfg["seed"], device=device,
        grad_clip=cfg["grad_clip"], restore_best=cfg["restore_best"],
    )
    elapsed = time.perf_counter() - start
    epochs_run = hist["epochs_run"][0]
    used = hist["best_epoch"][0] - 1 if cfg["restore_best"] else epochs_run - 1
    train_mae = float(np.mean(np.abs(train.predict(model, data["X_tr"]) - data["y_tr"])))
    out: Dict[str, Any] = {
        "cfg": cfg,
        "val_window_mae": hist["val_mae"][used],
        "train_loss_final": hist["train_loss"][used],
        "train_mae": train_mae,
        "best_epoch": hist["best_epoch"][0],
        "epochs_run": epochs_run,
        "sec_per_epoch": elapsed / epochs_run,
        "n_params": sum(p.numel() for p in model.parameters()),
        "history": hist,
    }
    if cfg["final"]:
        out["points"] = train.evaluate_points(model, data["X_pt"], data["y_pt"])
        out["val_pred"] = train.predict(model, data["X_pt"])
        out["test_pred"] = train.predict(model, data["X_te"])
        out["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    return out


def _init_worker() -> None:
    torch.set_num_threads(1)


def run_many(configs: List[Dict[str, Any]], workers: int = 6) -> List[Dict[str, Any]]:
    """Run configs in parallel worker processes; results keep the input order."""
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        return list(pool.map(run_one, configs))
