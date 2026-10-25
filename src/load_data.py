from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SETTINGS = ["setting_1", "setting_2", "setting_3"]
ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["engine_id", "cycle"] + SETTINGS + ALL_SENSORS
FLAT_COLUMNS = [
    "setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]
SENSORS = [s for s in ALL_SENSORS if s not in FLAT_COLUMNS]


def load_train() -> pd.DataFrame:
    """Load the training observations and compute each row's RUL."""
    train = pd.read_csv(
        DATA_DIR / "train_FD001.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    train["rul"] = (
        train.groupby("engine_id")["cycle"].transform("max") - train["cycle"]
    )
    return train


def load_test() -> pd.DataFrame:
    """Load the test observations without adding a training-only RUL label."""
    return pd.read_csv(
        DATA_DIR / "test_FD001.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )


def load_test_rul() -> pd.DataFrame:
    """Load the supplied RUL for each test engine."""
    rul = pd.read_csv(
        DATA_DIR / "RUL_FD001.txt",
        sep=r"\s+",
        header=None,
        names=["rul"],
    )
    rul.insert(0, "engine_id", range(1, len(rul) + 1))
    return rul
