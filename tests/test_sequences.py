import numpy as np
import pandas as pd
import pytest

from src.sequences import FEATURES, cap_rul, fit_scaler, make_windows


def frame() -> pd.DataFrame:
    rows = []
    for engine_id, length in [(1, 4), (2, 3)]:
        for cycle in range(1, length + 1):
            row = {"engine_id": engine_id, "cycle": cycle, "rul": length - cycle}
            row.update({name: engine_id * 100 + cycle * 10 + i for i, name in enumerate(FEATURES)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_shapes_and_targets():
    df = frame()
    X, y, keys = make_windows(df, fit_scaler(df[df.engine_id == 1]), window=3)
    assert X.shape == (7, 3, 14)
    assert y.shape == (7,)
    assert keys[["engine_id", "cycle"]].to_numpy().tolist() == [
        [1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3]
    ]
    np.testing.assert_array_equal(y, df.sort_values(["engine_id", "cycle"]).rul)


def test_window_is_scaled_history_without_lookahead_or_cross_engine_rows():
    df = frame()
    scaler = fit_scaler(df[df.engine_id == 1])
    X, _, _ = make_windows(df, scaler, window=3)
    values = (df[df.engine_id == 1].sort_values("cycle")[FEATURES].to_numpy() - scaler[0]) / scaler[1]
    np.testing.assert_allclose(X[2], values[:3])

    changed = df.copy()
    changed.loc[(changed.engine_id == 1) & (changed.cycle == 4), FEATURES] = 999999
    changed_X, _, _ = make_windows(changed, scaler, window=3)
    np.testing.assert_allclose(X[2], changed_X[2])
    assert not np.all(X[0, :, 0] == X[4, :, 0])


def test_left_padding_repeats_first_observation():
    df = frame()
    scaler = fit_scaler(df[df.engine_id == 1])
    X, _, _ = make_windows(df[df.engine_id == 1], scaler, window=3)
    np.testing.assert_allclose(X[0, 0], X[0, 1])
    np.testing.assert_allclose(X[1, 0], X[1, 1])
    np.testing.assert_allclose(X[1, 1], X[0, 0])


def test_points_preserve_requested_order_and_missing_points_fail():
    df = frame()
    scaler = fit_scaler(df[df.engine_id == 1])
    points = pd.DataFrame({"engine_id": [2, 1], "cycle": [2, 4]})
    X, y, keys = make_windows(df, scaler, window=2, points=points)
    assert keys[["engine_id", "cycle"]].to_records(index=False).tolist() == [(2, 2), (1, 4)]
    np.testing.assert_array_equal(y, [1, 0])
    assert X.shape == (2, 2, 14)
    with pytest.raises(ValueError, match="requested points"):
        make_windows(df, scaler, points=pd.DataFrame({"engine_id": [9], "cycle": [1]}))


def test_scaler_uses_only_input_rows_and_constant_columns_are_safe():
    df = frame()
    train = df[df.engine_id == 1].copy()
    train[FEATURES[0]] = 7
    mean, std = fit_scaler(train)
    assert mean[0] == 7
    assert std[0] == 1
    assert mean[1] != df[df.engine_id == 2][FEATURES[1]].mean()


def test_test_dataframe_without_rul_returns_none():
    df = frame().drop(columns="rul")
    X, y, keys = make_windows(df, fit_scaler(df[df.engine_id == 1]), window=2)
    assert X.shape == (7, 2, 14)
    assert y is None
    assert len(keys) == 7


def test_cap_rul_is_explicit_training_only_helper():
    np.testing.assert_array_equal(cap_rul(np.array([10, 200]), 125), [10, 125])
    with pytest.raises(ValueError):
        cap_rul(np.array([1]), -1)
