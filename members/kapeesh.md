# AML Project — Abhayanth: Split, metrics and linear models

**Your part:** decide how models are scored fairly (engine split, fixed validation points, MAE/RMSE), and build the linear baseline every other model is compared against.

- **Your files:** `src/evaluation.py`, `notebooks/03_linear_models.ipynb`
- **Branch:** `abhayanth/evaluation-linear`
- **Start:** after Akhil's work is merged (you need `load_data.py` and `features.py`). Run `git pull` on `main` first. Srijan starts when your `evaluation.py` is merged.

## Why it is set up this way

- Rows of one engine are near-copies of each other. A random **row** split would put the same engine on both sides and make models look better than they are, so we split by **engine**.
- A test engine is scored **once**, at a cut-off before failure. Validation copies this: 20 engines × 5 cut-off cycles = 100 points, saved in `data/validation_cutoffs.csv` so every model is scored on the same points.

## Setup (once)

Python 3.12 or newer.

```bash
git clone https://github.com/Abhayanthk/aml-rul-fd001.git
cd aml-rul-fd001
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git checkout -b abhayanth/evaluation-linear
```

## Step 1 — `src/evaluation.py`

Use exactly these names. Srijan and Ashrith import them.

| Name | What it must be |
| --- | --- |
| `CUTOFFS` | `pd.read_csv(DATA_DIR / "validation_cutoffs.csv")`: 100 rows, columns `engine_id, cycle, rul` |
| `VAL_ENGINES` | `sorted(CUTOFFS["engine_id"].unique())`: the 20 validation engines |
| `split_train_val(df)` | Returns `(train_df, val_df)`. `val_df` = rows whose `engine_id` is in `VAL_ENGINES`; `train_df` = all other rows. Keep the row order. |
| `validation_points(val_df)` | `CUTOFFS[["engine_id", "cycle"]].merge(val_df, on=["engine_id", "cycle"], how="left")`: the 100 scored rows, in `CUTOFFS` order |
| `test_points(test_df)` | `test_df.groupby("engine_id").tail(1)`: the last row of each test engine (100 rows) |
| `mae(y_true, y_pred)` | Mean of absolute errors, returned as a `float` |
| `rmse(y_true, y_pred)` | Square root of the mean squared error, returned as a `float` |

Import `DATA_DIR` from `load_data`. Write `mae` and `rmse` with NumPy (`np.mean`, `np.abs`, `np.sqrt`), not with sklearn, so you can explain each step.

## Step 2 — `notebooks/03_linear_models.ipynb`

First cell:
```python
import sys
sys.path.append("../src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from load_data import ALL_SENSORS, SETTINGS, load_test, load_train
from features import FEATURES, add_rolling_features
from evaluation import VAL_ENGINES, mae, rmse, split_train_val, test_points, validation_points
```

Sections, in this order:

1. **Data:**
   ```python
   train = add_rolling_features(load_train())
   test = add_rolling_features(load_test())
   train_df, val_df = split_train_val(train)
   val_pts = validation_points(val_df)
   test_pts = test_points(test)
   ```
   Print the row and engine counts and `VAL_ENGINES`.
2. **Two feature sets:**
   - `V1 = SETTINGS + ALL_SENSORS`: 24 raw columns, current cycle only.
   - `V2 = FEATURES`: 57 columns.
3. **Two models, fixed settings.** Fit on `train_df` (all rows of the 80 training engines), target `train_df["rul"]`:
   - **OLS on V1:** `make_pipeline(StandardScaler(), LinearRegression())`
   - **Elastic Net on V2:** `make_pipeline(StandardScaler(), ElasticNet(alpha=1, l1_ratio=0.5, max_iter=50_000))`
   
   The pipeline fits the scaler on training rows only.
4. **Validation scores:** a small table with model, features, MAE and RMSE at the 100 points (`val_pts[cols]` against `val_pts["rul"]`).
5. **Underfitting check:** RMSE on all `train_df` rows vs all `val_df` rows for both models. Write one sentence: the training error is high too, so the linear model mostly underfits (high bias).
6. **Diagnostics for Elastic Net:** a 2×2 figure at the 100 validation points:
   - predicted vs actual, with a diagonal line;
   - residuals vs predicted;
   - residual histogram;
   - normal Q-Q plot (`stats.probplot(residuals, dist="norm", plot=ax)`).
   
   Count the negative predictions. Don't clip them.
7. **Coefficients:** Elastic Net coefficients (`model[-1].coef_`) as a Series indexed by `FEATURES`. Show the top 10 by absolute value, and count how many are exactly 0 (L1 removed them).
8. **Save predictions.** No test scoring here; Ashrith does that.
   - `results/val_predictions_linear.csv`, columns exactly `engine_id, cycle, ols_v1, elastic_net_v2` (100 rows, `val_pts` order)
   - `results/test_predictions_linear.csv`, columns exactly `engine_id, ols_v1, elastic_net_v2` (100 rows)
   
   Use `index=False`.

Finally, run **Kernel → Restart & Run All**.

## Check: your numbers must match

| Check | Expected |
| --- | --- |
| `train_df` rows / engines | 16390 / 80 |
| `val_df` rows / engines | 4241 / 20 |
| `val_pts` / `test_pts` rows | 100 / 100 |
| `VAL_ENGINES` | 8, 9, 13, 18, 37, 46, 48, 50, 55, 62, 64, 69, 72, 74, 76, 84, 85, 87, 90, 95 |
| OLS v1, validation MAE / RMSE | 32.86 / 42.92 |
| Elastic Net v2, validation MAE / RMSE | 31.66 / 41.77 |
| RMSE on all training rows, OLS / EN | 43.31 / 39.30 |
| RMSE on all validation rows, OLS / EN | 49.72 / 47.25 |
| Negative predictions at the 100 points, OLS / EN | 4 / 3 |
| EN coefficients exactly 0 | 7 of 57 |
| Largest EN coefficient | `cycle` −14.35 |
| Test predictions for engines 1, 2, 3, OLS | 158.47, 121.28, 85.13 |
| Test predictions for engines 1, 2, 3, EN | 157.46, 135.79, 83.91 |

## Submit

```bash
git add src/evaluation.py
git commit -m "Add engine split, validation points and MAE/RMSE"
git add notebooks/03_linear_models.ipynb results/val_predictions_linear.csv results/test_predictions_linear.csv
git commit -m "Add linear baseline: OLS and Elastic Net"
git push -u origin abhayanth/evaluation-linear
```

Open a pull request on GitHub. Srijan starts after it is merged.

## Questions you should be able to answer

1. Why split by engine, not by row? (Rows of one engine are near-copies; a row split leaks the engine into validation.)
2. Why 5 cut-off points per validation engine, not all rows? (It copies the test set: one point per engine before failure. Using all rows would let long-lived engines dominate.)
3. MAE vs RMSE? (MAE = average miss in cycles. RMSE squares errors first, so big misses count more.)
4. What do `alpha` and `l1_ratio` do? (`alpha` = penalty strength. `l1_ratio` = mix of L1, which sets coefficients to 0, and L2, which shrinks them.)
5. Why scale, and why only on training rows? (The penalty treats all coefficients the same, so features need the same scale. Fitting the scaler on validation or test rows would leak information.)
6. How were `alpha=1, l1_ratio=0.5` chosen? (In our earlier work, a grid on validation: alpha 0.001–1, l1_ratio 0.2/0.5/0.8; lowest validation RMSE.)
7. Is the model underfitting? (Yes: training RMSE ~39–43 is already high. A straight line can't follow "flat, then steep" degradation.)

Course topics in your part: OLS, regularisation (Ridge/Lasso/Elastic Net), evaluation metrics, assumptions of linear regression (residuals, Q-Q), bias–variance.
