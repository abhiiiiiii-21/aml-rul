# AML Project — Aryan: Data loading and EDA

**Your part:** load the raw files, create the RUL label, explore the data, and find the flat columns to remove. Everyone else's code starts from your `load_train()`.

- **Your files:** `src/load_data.py`, `notebooks/01_eda.ipynb`
- **Branch:** `aryan/data-eda`
- **Start:** now. You are first. Akhil starts when your work is merged.

## The data in 4 lines

- One row = one engine at one cycle (one flight). 26 columns, no header: `engine_id`, `cycle`, `setting_1`–`setting_3`, `sensor_1`–`sensor_21`.
- `train_FD001.txt`: 100 engines, each runs until it fails.
- `test_FD001.txt`: 100 different engines, stopped some time before failure.
- RUL (Remaining Useful Life) = how many cycles are left before failure.

## Setup (once)

Python 3.12 or newer.

```bash
git clone https://github.com/Abhayanthk/aml-rul-fd001.git
cd aml-rul-fd001
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git checkout -b aryan/data-eda
```

## Step 1 — `src/load_data.py`

Use exactly these names. Other people import them.

| Name              | What it must be                                                                       |
| ----------------- | ------------------------------------------------------------------------------------- |
| `DATA_DIR`        | `Path(__file__).resolve().parent.parent / "data"`                                     |
| `SETTINGS`        | `["setting_1", "setting_2", "setting_3"]`                                             |
| `ALL_SENSORS`     | `["sensor_1", ..., "sensor_21"]`, built with `[f"sensor_{i}" for i in range(1, 22)]`  |
| `COLUMNS`         | `["engine_id", "cycle"] + SETTINGS + ALL_SENSORS` (26 names)                          |
| `FLAT_COLUMNS`    | The 8 flat columns you find in Step 2, typed in as a list                             |
| `SENSORS`         | `[s for s in ALL_SENSORS if s not in FLAT_COLUMNS]` (14 sensors)                      |
| `load_train()`    | Reads `train_FD001.txt`, adds the column `rul`, returns the DataFrame                 |
| `load_test()`     | Reads `test_FD001.txt`, returns the DataFrame (no `rul`)                              |
| `load_test_rul()` | Reads `RUL_FD001.txt`, returns a DataFrame with columns `engine_id`, `rul` (100 rows) |

1. Read every file like this (the columns are separated by spaces):
   ```python
   pd.read_csv(DATA_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
   ```
2. RUL in `load_train()` = last cycle of that engine − current cycle. No cap.
   ```python
   df["rul"] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
   ```
3. `load_test_rul()`: read with `sep=r"\s+", header=None, names=["rul"]`. Then add the engine number as the first column. Line 1 of the file belongs to engine 1.
   ```python
   rul.insert(0, "engine_id", range(1, len(rul) + 1))
   ```
4. Don't sort, drop or change any rows. The files are already ordered by engine, then cycle.

## Step 2 — `notebooks/01_eda.ipynb`

First cell:
```python
import sys
sys.path.append("../src")
from load_data import ALL_SENSORS, FLAT_COLUMNS, SENSORS, SETTINGS, load_test, load_test_rul, load_train
```

Sections, in this order. Give each one a short markdown heading and 1–2 sentences on what it shows.

1. **Shapes:** shape of train and test, number of engines in each, total missing values.
2. **Engine lifetimes:** last cycle of each train engine. Print min, mean and max, and plot a histogram.
3. **RUL label:** show the first 3 and last 3 rows of engine 1 (`engine_id`, `cycle`, `rul`).
4. **Sensor trends:** a 2×2 figure of `sensor_2`, `sensor_4`, `sensor_11`, `sensor_15` against `rul` for engines 1–5. Invert the x-axis, so failure (RUL 0) is on the right. Note what you see: flat for most of the life, then changing near failure.
5. **Flat columns:** for each of the 24 settings and sensors, compute the share of rows that have its most common value:
   ```python
   train[col].value_counts(normalize=True).iloc[0]
   ```
   Show it as a table sorted from high to low. A column is **flat if the share is 0.95 or more**. Copy the result into `FLAT_COLUMNS` in `load_data.py`.
6. **Correlation:** heatmap of `train[SENSORS].corr()`. List the sensor pairs with absolute correlation ≥ 0.8.
7. **Operating condition:** `train[SETTINGS].describe()`. Write one sentence: `setting_1` and `setting_2` barely change and `setting_3` is constant, so FD001 has one operating condition and needs no per-condition normalisation.
8. **Test file:** cycles seen per test engine (min and max), and `load_test_rul().shape`. Don't print, plot or describe the test RUL values. They are the test answers and are used only in notebook 05.

Finally, run **Kernel → Restart & Run All**, so the outputs are saved in the notebook.

## Check: your numbers must match

| Check | Expected |
| --- | --- |
| `load_train().shape` | (20631, 27) |
| `load_test().shape` | (13096, 26) |
| Engines in train / test | 100 / 100 |
| Missing values | 0 |
| Engine lifetime min / mean / max | 128 / 206.31 / 362 |
| RUL of engine 1, cycle 1 | 191 |
| RUL at the last row of every engine | 0 |
| Most common value share | 1.00 for `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`; 0.98 for `sensor_6`; next highest is `sensor_17` at 0.264 |
| `FLAT_COLUMNS` | `setting_3, sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19` |
| `SENSORS` (14) | `sensor_2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21` |
| Sensor pairs with abs correlation ≥ 0.8 | 7; strongest `sensor_9` and `sensor_14` at 0.96 |
| `setting_1` min / max | −0.0087 / 0.0087 |
| Test cycles seen per engine min / max | 31 / 303 |
| `load_test_rul().shape` | (100, 2) |

## Submit

```bash
git add src/load_data.py notebooks/01_eda.ipynb
git commit -m "Add data loading, RUL label and EDA"
git push -u origin aryan/data-eda
```

Open a pull request on GitHub. Touch only your two files.

## Questions you should be able to answer

1. Why can RUL be computed as "last cycle − current cycle" on train but not on test? (Train engines run to failure. Test engines stop early, so NASA gives their RUL in a separate file.)
2. How did you decide a column is flat? (Most common value in 95% or more of rows. The next highest share is 26%, so the exact threshold doesn't matter.)
3. Why remove flat columns? (They carry no information about wear; they only add noise.)
4. Why is there no normalisation per operating condition, as the brief suggests? (FD001 has a single operating condition.)
5. What do the sensor plots show? (Flat for most of the life, then a clear change near failure. This is why predicting far from failure is hard.)

Course topics in your part: ML project lifecycle (data understanding), feature selection.
