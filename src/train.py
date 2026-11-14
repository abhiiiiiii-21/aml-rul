"""Training loop and prediction helpers for the Phase 2 models.

Training uses all windows of the 80 training engines. Early stopping watches
MAE on all windows of the 20 validation engines. Final numbers are always
reported on the 100 fixed validation points (see `evaluation.py`).
"""
import copy
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

RUL_SCALE = 100.0


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
) -> Dict[str, List[float]]:
    """Train with AdamW and early stopping. Restores the best weights.

    Returns a history dict with train_loss and val_mae (cycles) per epoch.
    """
    set_seed(seed)
    device = device or get_device()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train / RUL_SCALE)
    gen = torch.Generator().manual_seed(seed)

    history: Dict[str, List[float]] = {"train_loss": [], "val_mae": []}
    best_mae, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        val_mae = float(np.mean(np.abs(predict(model, X_val) - y_val)))
        history["train_loss"].append(total / len(Xt))
        history["val_mae"].append(val_mae)
        if val_mae < best_mae:
            best_mae, best_state, bad = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
