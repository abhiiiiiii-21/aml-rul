"""Helpers to train and score one CNN or Transformer configuration (notebook 04).

Same protocol as `rnn_experiments.py`: the scaler is fitted on the 80 training
engines, training uses all their windows, early stopping and model selection use
all windows of the 20 validation engines (`val_window_mae`). The 100 fixed
validation points and the test points are scored or predicted only when
`cfg["final"]` is true, and no test metric is ever computed here.
"""
from concurrent.futures import ProcessPoolExecutor
import time
from typing import Any, Dict, List

import numpy as np
import torch

from dl_models import build_model
from rnn_experiments import get_data
from sequences import DEFAULT_WINDOW
import train

DEFAULTS: Dict[str, Any] = {
    "model": "cnn1d", "kw": {}, "weight_decay": 1e-4, "patience": 8, "epochs": 60,
    "seed": 42, "lr": 1e-3, "grad_clip": 1.0, "restore_best": True,
    "window": DEFAULT_WINDOW, "final": False,
}


def run_one(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Train one config on CPU; return validation-engine metrics and history."""
    cfg = {**DEFAULTS, **overrides}
    data = get_data(cfg["window"])
    device = torch.device("cpu")  # deterministic; the data is small
    train.set_seed(cfg["seed"])  # before building: initial weights depend on the seed
    model = build_model(cfg["model"], **cfg["kw"])
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
    out: Dict[str, Any] = {
        "cfg": cfg,
        "val_window_mae": hist["val_mae"][used],
        "train_loss_final": hist["train_loss"][used],
        "train_mae": float(np.mean(np.abs(train.predict(model, data["X_tr"]) - data["y_tr"]))),
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
