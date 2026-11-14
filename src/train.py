"""Training loop and prediction helpers for the Phase 2 models.

Training uses all windows of the 80 training engines. Early stopping watches
MAE on all windows of the 20 validation engines, so the validation scores
reported afterwards are slightly optimistic (the epoch was picked on those
engines). The held-out test set is the unbiased check. Final validation
numbers are reported on the 100 fixed validation points (see `evaluation.py`).
"""
import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn

try:
    from dl_models import build_model
    from evaluation import mae, rmse
except ImportError:
    from src.dl_models import build_model
    from src.evaluation import mae, rmse

RUL_SCALE = 100.0
EARLY_LATE_SPLIT = 125.0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """RUL predictions in cycles, clipped at 0."""
    device = next(model.parameters()).device
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.clip(np.concatenate(out) * RUL_SCALE, 0, None)


def evaluate_points(
    model: nn.Module, X_points: np.ndarray, y_points: np.ndarray
) -> Dict[str, float]:
    """MAE, RMSE, and MAE of the early (true RUL > 125) and late (<= 125) points."""
    y_true = np.asarray(y_points, dtype=float)
    y_pred = predict(model, X_points)
    early = y_true > EARLY_LATE_SPLIT
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "early_mae": mae(y_true[early], y_pred[early]) if early.any() else float("nan"),
        "late_mae": mae(y_true[~early], y_pred[~early]) if (~early).any() else float("nan"),
    }


def save_model(model: nn.Module, path: Union[str, Path]) -> None:
    """Save the model `state_dict` (CPU tensors) to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)


def load_model(name: str, path: Union[str, Path], **kwargs) -> nn.Module:
    """Build `name` with `kwargs` (see `build_model`) and load weights from `path`."""
    model = build_model(name, **kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def fit(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    seed: int = 42,
    device: Optional[torch.device] = None,
    rul_cap: Optional[float] = None,
    lr_schedule: bool = False,
    grad_clip: Optional[float] = 1.0,
    restore_best: bool = True,
) -> Dict[str, List[float]]:
    """Train with AdamW and early stopping; restore the best weights.

    Early stopping uses MAE (cycles, true uncapped RUL) on the validation-engine
    windows. `rul_cap` caps only the training target. `lr_schedule` halves the
    learning rate when validation MAE stops improving. `grad_clip=None` disables
    clipping; the gradient norm is logged either way. `restore_best=False` keeps
    the last epoch (use with a large `patience` to switch early stopping off).

    Call `set_seed` before building the model if its initial weights must be
    reproducible; `fit` seeds only the shuffling and dropout that follow.

    Returns a history dict: per-epoch `train_loss` (MSE on RUL/100, dropout on),
    `val_mae` (cycles), `val_loss` (MSE on RUL/100), `grad_norm` (mean pre-clip
    norm), plus `best_epoch` and `epochs_run` (1-based, single-element lists).
    """
    set_seed(seed)
    device = device or get_device()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        if lr_schedule
        else None
    )
    loss_fn = nn.MSELoss()
    clip_at = float("inf") if grad_clip is None else grad_clip

    y_target = y_train if rul_cap is None else np.minimum(y_train, rul_cap)
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(np.asarray(y_target, dtype=np.float32) / RUL_SCALE)
    gen = torch.Generator().manual_seed(seed)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_mae": [], "val_loss": [], "grad_norm": [],
    }
    best_mae, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total, norms = 0.0, []
        for i in range(0, len(Xt), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), clip_at)))
            opt.step()
            total += loss.item() * len(idx)
        pred = predict(model, X_val)
        val_mae = float(np.mean(np.abs(pred - y_val)))
        history["train_loss"].append(total / len(Xt))
        history["val_mae"].append(val_mae)
        history["val_loss"].append(float(np.mean(((pred - y_val) / RUL_SCALE) ** 2)))
        history["grad_norm"].append(float(np.mean(norms)))
        if sched is not None:
            sched.step(val_mae)
        if val_mae < best_mae:
            best_mae, best_epoch, bad = val_mae, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = [best_epoch]
    history["epochs_run"] = [len(history["val_mae"])]
    return history
