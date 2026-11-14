"""Hybrid ML + DL model for Phase 2 (stacking a tree on a deep-model estimate).

Hybrid A: a deep sequence model maps each window to a RUL estimate `d`. A tree
(Phase 1 Random Forest or Gradient Boosting settings) predicts RUL from the
Phase 1 rolling features plus `d`. The training values of `d` are out-of-fold
by engine: every training row is predicted by a model that never saw its engine.

Hybrid B is the plain mean of a DL prediction and a tree prediction. Hybrid C
lets the tree predict the residual `y - d`.
"""
from typing import List, Sequence

import numpy as np


def engine_folds(engine_ids: Sequence[int], k: int = 5, seed: int = 42) -> List[np.ndarray]:
    """Split the unique engine ids into `k` disjoint folds (random, seeded, by engine)."""
    engines = np.unique(np.asarray(engine_ids))
    if not 2 <= k <= len(engines):
        raise ValueError("k must be between 2 and the number of engines")
    shuffled = np.random.default_rng(seed).permutation(engines)
    return [np.sort(fold) for fold in np.array_split(shuffled, k)]
