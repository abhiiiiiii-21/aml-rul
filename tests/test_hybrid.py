import numpy as np
import pytest

from src.evaluation import VAL_ENGINES, split_train_val
from src.hybrid import engine_folds
from src.load_data import load_train


def test_folds_partition_the_engines():
    ids = np.repeat(np.arange(1, 81), 3)  # several rows per engine
    folds = engine_folds(ids, k=5, seed=42)
    assert len(folds) == 5
    assert [len(f) for f in folds] == [16] * 5
    joined = np.concatenate(folds)
    assert sorted(joined.tolist()) == list(range(1, 81))
    assert len(set(joined.tolist())) == 80  # disjoint


def test_folds_are_deterministic_and_seed_dependent():
    ids = np.arange(1, 81)
    a = engine_folds(ids, k=5, seed=42)
    b = engine_folds(ids, k=5, seed=42)
    c = engine_folds(ids, k=5, seed=43)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_folds_reject_bad_k():
    with pytest.raises(ValueError):
        engine_folds([1, 2, 3], k=4)
    with pytest.raises(ValueError):
        engine_folds([1, 2, 3], k=1)


def test_folds_of_training_engines_never_contain_validation_engines():
    train_df, _ = split_train_val(load_train())
    folds = engine_folds(train_df["engine_id"], k=5, seed=42)
    in_folds = set(np.concatenate(folds).tolist())
    assert len(in_folds) == 80
    assert in_folds.isdisjoint(VAL_ENGINES)
