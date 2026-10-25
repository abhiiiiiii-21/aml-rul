from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from load_data import DATA_DIR
except ImportError:
    from src.load_data import DATA_DIR

CUTOFFS: pd.DataFrame = pd.read_csv(DATA_DIR / "validation_cutoffs.csv")
VAL_ENGINES: list[int] = sorted(CUTOFFS["engine_id"].unique())


def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and validation by engine_id.

    Returns (train_df, val_df). val_df contains rows whose engine_id is in
    VAL_ENGINES; train_df contains all other rows. Original row order is preserved.
    """
    val_mask = df["engine_id"].isin(VAL_ENGINES)
    train_df = df[~val_mask].copy()
    val_df = df[val_mask].copy()
    return train_df, val_df


def validation_points(val_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the 100 fixed validation cutoff points in CUTOFFS order."""
    return CUTOFFS[["engine_id", "cycle"]].merge(
        val_df, on=["engine_id", "cycle"], how="left"
    )


def test_points(test_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the last observed cycle for each test engine (100 rows)."""
    return test_df.groupby("engine_id").tail(1)


def mae(y_true, y_pred) -> float:
    """Calculate Mean Absolute Error (MAE) using NumPy."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse(y_true, y_pred) -> float:
    """Calculate Root Mean Squared Error (RMSE) using NumPy."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
