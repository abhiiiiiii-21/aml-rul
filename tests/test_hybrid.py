import numpy as np
import pandas as pd
import pytest

from src.evaluation import VAL_ENGINES, split_train_val
from src.features import FEATURES as ROLLING
from src.hybrid import (
    KEYS, build_hybrid_features, dl_predict, engine_folds, fold_mean_predict,
    oof_predictions, split_fit_stop,
)
from src.load_data import load_test, load_train


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


TINY = {"hidden": 8, "layers": 1, "dropout": 0.0}
FAST = {"epochs": 1, "patience": 1}


def small_train(n_engines: int = 10):
    train_df, _ = split_train_val(load_train())
    keep = np.unique(train_df["engine_id"])[:n_engines]
    return train_df[train_df["engine_id"].isin(keep)]


def test_split_fit_stop_is_disjoint_and_complete():
    fit, stop = split_fit_stop(np.arange(1, 65), n_stop=8, seed=0)
    assert len(stop) == 8 and len(fit) == 56
    assert set(fit).isdisjoint(stop)
    assert sorted(set(fit) | set(stop)) == list(range(1, 65))


def test_oof_rows_are_predicted_by_a_model_that_never_saw_their_engine():
    df = small_train()
    oof, models = oof_predictions(
        df, "lstm", TINY, window=5, k=5, n_stop=2, fit_kwargs=FAST
    )
    assert oof[KEYS].equals(df.sort_values(KEYS)[KEYS].reset_index(drop=True))
    for i, fm in enumerate(models):
        held = oof.loc[oof["fold"] == i, "engine_id"].unique()
        assert set(held) == set(fm.holdout_engines)
        assert set(fm.seen_engines).isdisjoint(fm.holdout_engines)
        assert set(fm.seen_engines).isdisjoint(VAL_ENGINES)
        assert set(fm.seen_engines) <= set(df["engine_id"])  # training engines only
    assert oof["d"].notna().all() and (oof["d"] >= 0).all()


def test_oof_refuses_validation_or_unlabelled_data():
    train_df, val_df = split_train_val(load_train())
    with pytest.raises(ValueError):
        oof_predictions(pd.concat([train_df, val_df]), "lstm", TINY)
    with pytest.raises(ValueError):
        oof_predictions(load_test(), "lstm", TINY)  # test engines have no rul


def test_fold_mean_predict_averages_the_fold_models():
    df = small_train()
    _, models = oof_predictions(df, "lstm", TINY, window=5, k=5, n_stop=2, fit_kwargs=FAST)
    _, val_df = split_train_val(load_train())
    val_one = val_df[val_df["engine_id"] == VAL_ENGINES[0]]
    mean = fold_mean_predict(models, val_one, window=5)
    each = [dl_predict(fm.model, fm.scaler, val_one, window=5)["d"] for fm in models]
    np.testing.assert_allclose(mean["d"], np.mean(each, axis=0), rtol=1e-6)
    assert len(mean) == len(val_one)


def fake_dl(df: pd.DataFrame) -> pd.DataFrame:
    """A DL prediction whose value encodes its own key, shuffled out of order."""
    out = df[KEYS].copy()
    out["d"] = out["engine_id"] * 1000 + out["cycle"]
    return out.sample(frac=1.0, random_state=0)


def test_features_join_dl_by_key_not_by_position():
    df = small_train(3)
    table = build_hybrid_features(df.sample(frac=1.0, random_state=1), fake_dl(df))
    keys = table.index.to_frame(index=False)
    assert keys.equals(df.sort_values(KEYS)[KEYS].reset_index(drop=True))
    np.testing.assert_array_equal(table["d"], keys["engine_id"] * 1000 + keys["cycle"])
    np.testing.assert_array_equal(table["cycle"], keys["cycle"])
    assert list(table.columns) == list(ROLLING) + ["d"]


def test_features_follow_points_order():
    df = small_train(3)
    points = df[KEYS].sample(n=7, random_state=2)
    table = build_hybrid_features(df, fake_dl(df), points=points)
    keys = table.index.to_frame(index=False)
    assert keys.equals(points.reset_index(drop=True))
    np.testing.assert_array_equal(table["d"], keys["engine_id"] * 1000 + keys["cycle"])


def test_feature_groups_select_columns():
    df = small_train(2)
    assert list(build_hybrid_features(df, fake_dl(df), groups=["dl"]).columns) == ["d"]
    rolling_only = build_hybrid_features(df, None, groups=["rolling"])
    assert list(rolling_only.columns) == list(ROLLING)
    with pytest.raises(ValueError):
        build_hybrid_features(df, None, groups=["dl"])
    with pytest.raises(ValueError):
        build_hybrid_features(df, None, groups=["embedding"])


def test_features_reject_missing_dl_rows():
    df = small_train(2)
    with pytest.raises(ValueError):
        build_hybrid_features(df, fake_dl(df).iloc[1:])
