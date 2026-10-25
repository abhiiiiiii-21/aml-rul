# AML Project — Srijan: Tree models

**Your part:** train the two non-linear models the brief asks for, Random Forest and gradient boosting, on the same features and split as the linear baseline. Show what they rely on and save their predictions.

- **Your files:** `notebooks/04_tree_models.ipynb` (plus the 2 prediction files it writes)
- **Branch:** `srijan/tree-models`
- **Start:** after Abhayanth's work is merged (you need `load_data.py`, `features.py`, `evaluation.py`). Run `git pull` on `main` first. Ashrith starts after your work is merged.

## Why trees

The sensors stay flat for most of an engine's life, then change quickly near failure. A straight line can't follow "flat, then steep"; trees can, because they split the data into regions.

- **Random Forest:** many trees on random samples, averaged (bagging). Mainly lowers variance.
- **Gradient boosting:** small trees added one by one, each fixing the previous errors. Mainly lowers bias.
  - The brief names XGBoost. We use scikit-learn's `HistGradientBoostingRegressor`, which works the same way.

## Setup (once)

Python 3.12 or newer.

```bash
git clone https://github.com/Abhayanthk/aml-rul-fd001.git
cd aml-rul-fd001
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git checkout -b srijan/tree-models
```

## `notebooks/04_tree_models.ipynb`

First cell:
```python
import sys
sys.path.append("../src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from load_data import load_test, load_train
from features import FEATURES, add_rolling_features
from evaluation import mae, rmse, split_train_val, test_points, validation_points
```

Sections, in this order:

1. **Data**, exactly the same as notebook 03:
   ```python
   train = add_rolling_features(load_train())
   test = add_rolling_features(load_test())
   train_df, val_df = split_train_val(train)
   val_pts = validation_points(val_df)
   test_pts = test_points(test)
   ```
2. **Two models, fixed settings.** Copy them exactly; `random_state=42` is what makes the numbers repeat.
   ```python
   rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.33,
                              random_state=42, n_jobs=-1)
   gb = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, max_depth=3,
                                      early_stopping=False, random_state=42)
   ```
   Fit both on `train_df[FEATURES]`, target `train_df["rul"]`. No scaling; trees don't need it.
   - `early_stopping=False`: built-in early stopping would hold out random **rows**, putting cycles of the same engine on both sides. That is the same leak the engine split prevents.
3. **Validation scores:** a table with model, MAE and RMSE at the 100 points (`val_pts[FEATURES]` against `val_pts["rul"]`).
4. **Fit check:** RMSE on all `train_df` rows vs all `val_df` rows for both models. Write one sentence: Random Forest fits the training rows much better than validation (some overfitting), and boosting less so.
5. **Predicted vs actual:** one figure with 2 panels (RF, GB). Each shows predicted RUL against true RUL at the 100 validation points, with a diagonal line.
6. **Feature importance:** `pd.Series(rf.feature_importances_, index=FEATURES)`, top 10 as a horizontal bar chart. Write 1–2 sentences: `cycle` leads, then sensor rolling means. Importance shows what the model **uses**, not what **causes** failure, and correlated features share credit.
7. **Save predictions.** No test scoring here; Ashrith does that.
   - `results/val_predictions_trees.csv`, columns exactly `engine_id, cycle, random_forest, gradient_boosting` (100 rows, `val_pts` order)
   - `results/test_predictions_trees.csv`, columns exactly `engine_id, random_forest, gradient_boosting` (100 rows)
   
   Use `index=False`.

Finally, run **Kernel → Restart & Run All**.

## Check: your numbers must match

| Check | Expected |
| --- | --- |
| Random Forest, validation MAE / RMSE | 25.03 / 40.17 |
| Gradient boosting, validation MAE / RMSE | 26.12 / 40.17 |
| RMSE on all training rows, RF / GB | 21.15 / 29.66 |
| RMSE on all validation rows, RF / GB | 42.62 / 43.02 |
| RF importance, top 5 | `cycle` 0.325, `sensor_4_mean` 0.125, `sensor_3_mean` 0.090, `sensor_11_mean` 0.067, `sensor_15_mean` 0.062 |
| Test predictions for engines 1, 2, 3, RF | 166.04, 154.59, 51.96 |
| Test predictions for engines 1, 2, 3, GB | 164.49, 157.63, 58.25 |

If the RF numbers are slightly off, check three things: the `FEATURES` order, `random_state=42`, and that you installed from `requirements.txt` (same scikit-learn version).

## Submit

```bash
git add notebooks/04_tree_models.ipynb results/val_predictions_trees.csv results/test_predictions_trees.csv
git commit -m "Add Random Forest and gradient boosting"
git push -u origin srijan/tree-models
```

Open a pull request on GitHub. Touch only these files.

## Questions you should be able to answer

1. Why do trees beat the linear model? (Degradation is flat, then steep. Trees split it into regions; a line can't bend.)
2. Random Forest vs boosting? (RF: many independent trees averaged, which lowers variance. Boosting: trees in sequence, each fixing the last one's errors, which lowers bias.)
3. What does `min_samples_leaf=20` do? (Every leaf must hold at least 20 rows, so the trees can't memorise single rows. Less overfitting.)
4. What does `max_features=0.33` do? (Each split looks at a random third of the features, so the trees differ more and averaging helps more.)
5. Why `early_stopping=False`? (Its internal hold-out picks random rows, which leaks the same engine.)
6. Why is `cycle` the most important feature? (Older engines have less life left on average. The model leans on age as much as on sensor history, which is part of the gap Phase 2 targets.)
7. How were the settings chosen? (In our earlier work, a small grid on validation. RF: `min_samples_leaf` 1/5/20, `max_features` 0.33/1.0. GB: `learning_rate` 0.05/0.1, `max_iter` 200/500, `max_depth` 3/None. Lowest validation RMSE won.)

Course topics in your part: bias–variance trade-off, evaluation metrics. Random Forest and boosting are the models the project brief names.
