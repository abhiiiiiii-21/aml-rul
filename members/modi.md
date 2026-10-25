# AML Project — Akhil: Rolling features

**Your part:** turn raw sensor readings into features that describe the recent trend: for each sensor, the mean, standard deviation and slope over the last 10 cycles of the same engine. Every model uses your `FEATURES` list.

- **Your files:** `src/features.py`, `notebooks/02_features.ipynb`
- **Branch:** `akhil/features`
- **Start:** after Aryan's work is merged (you need `src/load_data.py`). Run `git pull` on `main` first. Abhayanth starts after your work is merged.

## Why these features

- A single reading is noisy. The **mean** over 10 cycles smooths it.
- The **std** shows how much the sensor has been jumping around recently.
- The **slope** shows how fast the sensor is drifting, which is the wear signal.
- The window uses **only the current and earlier cycles of the same engine**. A test engine is cut off before failure, so a feature must never look ahead or borrow cycles from another engine.

## Setup (once)

Python 3.12 or newer.

```bash
git clone https://github.com/Abhayanthk/aml-rul-fd001.git
cd aml-rul-fd001
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git checkout -b akhil/features
```

## Step 1 — `src/features.py`

Use exactly these names:

| Name | What it must be |
| --- | --- |
| `WINDOW` | `10` |
| `FEATURES` | `["cycle"] + SENSORS + [f"{s}_{stat}" for s in SENSORS for stat in ["mean", "std", "slope"]]`, which is 57 names. Keep this exact order: the tree models' results depend on it. |
| `add_rolling_features(df)` | Returns a **copy** of `df` with 42 new columns: `<sensor>_mean`, `<sensor>_std`, `<sensor>_slope` for each of the 14 `SENSORS` |

Import `SENSORS` from Aryan's file: `from load_data import SENSORS`.

Rules inside `add_rolling_features(df)`:

1. Start with `df = df.copy()`. Don't sort or reorder rows.
2. Each window = the current cycle and up to 9 earlier cycles of the **same engine**. The first cycles use whatever history exists (`min_periods=1`). Use these two helpers exactly:
   ```python
   engine = df["engine_id"]

   def rolling_mean(values):
       return values.groupby(engine).transform(lambda v: v.rolling(WINDOW, min_periods=1).mean())

   def rolling_std(values):
       return values.groupby(engine).transform(lambda v: v.rolling(WINDOW, min_periods=1).std())
   ```
3. With `t = df["cycle"].astype(float)` and `x = df[sensor]`:
   - **mean** = `rolling_mean(x)`
   - **std** = `rolling_std(x).fillna(0.0)`. The first cycle has only one value, so its std is NaN; replace it with 0.
   - **slope** = the least-squares slope of `x` against `t` inside the window:
     ```python
     mean_t = rolling_mean(t)
     var_t = rolling_mean(t * t) - mean_t * mean_t
     cov_tx = rolling_mean(t * x) - mean_t * rolling_mean(x)
     slope = (cov_tx / var_t.where(var_t > 1e-9)).fillna(0.0)
     ```
     At the first cycle `var_t` is 0, so the slope is set to 0.
4. Compute `mean_t` and `var_t` once, before the loop over sensors.
5. Write the slope with this exact formula. Don't use `np.polyfit`: tiny rounding differences would change the tree models' numbers.

## Step 2 — `notebooks/02_features.ipynb`

First cell:
```python
import sys
sys.path.append("../src")
from load_data import load_test, load_train
from features import FEATURES, WINDOW, add_rolling_features
```

Sections, in this order. Give each one a markdown heading and 1–2 sentences of explanation.

1. **Build:** `train = add_rolling_features(load_train())`. Print `train.shape`, `len(FEATURES)`, the first 6 and last 3 names in `FEATURES`, and the number of NaN values in `train[FEATURES]`.
2. **Look at the numbers:** engine 1, cycles 1–12, columns `cycle, sensor_2, sensor_2_mean, sensor_2_std, sensor_2_slope`.
3. **Plot:** a 2-panel figure for engine 1 over its whole life: (a) `sensor_11` raw and `sensor_11_mean` on the same axes, (b) `sensor_11_slope`.
4. **The window stays inside one engine:** show engine 2, cycle 1. Its mean equals its raw value, and its std and slope are 0, so nothing leaked from engine 1's last cycles.
5. **Features use only the past:** compute features on engine 1 cut at cycle 100:
   ```python
   cut = add_rolling_features(load_train().query("engine_id == 1 and cycle <= 100"))
   ```
   Compare its last row with engine 1, cycle 100 from the full table (`FEATURES` columns). They must be identical. Write one sentence on why this matters: test engines are cut off before failure.
6. **Which features relate to RUL:** the 10 features with the largest absolute correlation with `rul`:
   ```python
   train[FEATURES].corrwith(train["rul"]).abs().sort_values(ascending=False).head(10)
   ```
7. **Test file works too:** `add_rolling_features(load_test()).shape`.

Finally, run **Kernel → Restart & Run All**, so the outputs are saved.

## Check: your numbers must match

| Check | Expected |
| --- | --- |
| `add_rolling_features(load_train()).shape` | (20631, 69) |
| `len(FEATURES)` | 57 |
| First 6 `FEATURES` | `cycle, sensor_2, sensor_3, sensor_4, sensor_7, sensor_8` |
| Last 3 `FEATURES` | `sensor_21_mean, sensor_21_std, sensor_21_slope` |
| NaN in `train[FEATURES]` | 0 |
| Engine 1, cycle 1: `sensor_2` mean / std / slope | 641.8200 / 0.0000 / 0.0000 |
| Engine 1, cycle 2 | 641.9850 / 0.2333 / 0.3300 |
| Engine 1, cycle 10 | 642.2010 / 0.2759 / −0.0002 |
| Engine 1, cycle 11 | 642.2470 / 0.2416 / −0.0213 |
| Engine 1, `sensor_11_slope` at cycles 1 / 2 / 10 / 11 | 0.0000 / 0.0200 / −0.0299 / −0.0234 |
| Engine 2, cycle 1: `sensor_2` mean / std / slope | 641.89 / 0 / 0 |
| Engine 1 cut at cycle 100: last row equals the full-table row | True |
| Top correlations with `rul` | `cycle` 0.736, `sensor_4_mean` 0.733, `sensor_21_mean` 0.730, `sensor_11_mean` 0.729, `sensor_17_mean` 0.728 |
| `add_rolling_features(load_test()).shape` | (13096, 68) |

## Submit

```bash
git add src/features.py notebooks/02_features.ipynb
git commit -m "Add rolling mean, std and slope features"
git push -u origin akhil/features
```

Open a pull request on GitHub. Touch only your two files.

## Questions you should be able to answer

1. Why must the window stay inside one engine? (Otherwise engine 2's first cycles would use engine 1's last cycles, which are near failure: wrong and leaky.)
2. Why only past cycles, not a centred window? (At prediction time the future doesn't exist; test engines stop before failure.)
3. What does the slope measure? (How fast the sensor drifts per cycle over the last 10 cycles, i.e. the speed of wear.)
4. Why 10 cycles? (Long enough to smooth noise, shorter than the shortest test history of 31 cycles.)
5. Why are std and slope 0 at the first cycle? (One value has no spread and no trend.)
6. What can a 10-cycle window **not** see? (The long history: when degradation started and how it built up over 100+ cycles. This is the "gap" in the project brief, and the reason for the Phase 2 LSTM.)

Course topics in your part: time series analysis (rolling windows), feature engineering.
