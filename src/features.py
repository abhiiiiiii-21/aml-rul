from pathlib import Path

import pandas as pd

try:
    from load_data import SENSORS
except ImportError:
    from src.load_data import SENSORS

WINDOW = 10

FEATURES = (
    ["cycle"]
    + SENSORS
    + [f"{s}_{stat}" for s in SENSORS for stat in ["mean", "std", "slope"]]
)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with rolling mean, std, and slope for each sensor.

    Computes features using only the current and earlier cycles of the same engine
    over a window of up to WINDOW cycles (min_periods=1).
    """
    df = df.copy()
    engine = df["engine_id"]

    def rolling_mean(values):
        return values.groupby(engine).transform(
            lambda v: v.rolling(WINDOW, min_periods=1).mean()
        )

    def rolling_std(values):
        return values.groupby(engine).transform(
            lambda v: v.rolling(WINDOW, min_periods=1).std()
        )

    t = df["cycle"].astype(float)
    mean_t = rolling_mean(t)
    var_t = rolling_mean(t * t) - mean_t * mean_t

    for sensor in SENSORS:
        x = df[sensor]
        mean_x = rolling_mean(x)
        std_x = rolling_std(x).fillna(0.0)
        cov_tx = rolling_mean(t * x) - mean_t * rolling_mean(x)
        slope_x = (cov_tx / var_t.where(var_t > 1e-9)).fillna(0.0)

        df[f"{sensor}_mean"] = mean_x
        df[f"{sensor}_std"] = std_x
        df[f"{sensor}_slope"] = slope_x

    return df
