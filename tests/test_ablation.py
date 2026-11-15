"""Alignment and leakage checks for the ablation feature helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ablation as ab  # noqa: E402
from evaluation import split_train_val, validation_points  # noqa: E402
from load_data import load_train  # noqa: E402


@pytest.fixture(scope="module")
def split():
    return split_train_val(load_train())


def test_settings_scaler_uses_training_engines_only(split):
    train_df, val_df = split
    mean, std = ab.fit_settings_scaler(train_df)
    assert np.allclose(mean, train_df[ab.SETTING_COLUMNS].mean().to_numpy(), atol=1e-4)
    assert not np.allclose(mean, val_df[ab.SETTING_COLUMNS].mean().to_numpy(), atol=1e-9)


def test_settings_windows_causal_and_ordered(split):
    train_df, val_df = split
    scaler = ab.fit_settings_scaler(train_df)
    points = validation_points(val_df)
    window = 30
    full = ab.settings_windows(val_df, scaler, window)
    picked = ab.settings_windows(val_df, scaler, window, points=points)
    assert full.shape == (len(val_df), window, 2) and picked.shape == (100, window, 2)
    sorted_df = val_df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    for k in (0, 7, 55):
        row = sorted_df[(sorted_df.engine_id == points.engine_id[k]) & (sorted_df.cycle == points.cycle[k])]
        eng = sorted_df[sorted_df.engine_id == points.engine_id[k]]
        end = int(row.index[0] - eng.index[0])  # position of the point inside its engine
        values = (eng[ab.SETTING_COLUMNS].to_numpy(np.float32) - scaler[0]) / scaler[1]
        expected = values[max(0, end - window + 1): end + 1]
        if len(expected) < window:  # left padding repeats the first row
            expected = np.concatenate([np.repeat(values[:1], window - len(expected), axis=0), expected])
        assert np.allclose(picked[k], expected, atol=1e-5)


def test_missing_point_raises(split):
    train_df, val_df = split
    scaler = ab.fit_settings_scaler(train_df)
    bad = validation_points(val_df).copy()
    bad.loc[0, "cycle"] = 10_000
    with pytest.raises(ValueError):
        ab.settings_windows(val_df, scaler, 30, points=bad)


def test_diff_features_first_step_zero_and_values():
    X = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    out = ab.add_diff_features(X)
    assert out.shape == (2, 5, 6)
    assert np.all(out[:, 0, 3:] == 0)
    assert np.allclose(out[:, 1:, 3:], X[:, 1:] - X[:, :-1])


def test_get_arrays_channel_counts_and_unknown_set():
    assert ab.get_arrays(10, "settings")["X_pt"].shape == (100, 10, 16)
    assert ab.get_arrays(10, "diff")["X_pt"].shape == (100, 10, 28)
    with pytest.raises(ValueError):
        ab.get_arrays(10, "everything")
