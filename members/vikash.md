# AML Project — Ashrith: Final evaluation

**Your part:** score every model in one place, on validation **and** on the test set. Break the error into early and late points (the project brief asks for "error by early versus late degradation stage"), and show the gap that motivates Phase 2.

- **Your files:** `notebooks/05_final_evaluation.ipynb` (plus `results/final_results.csv` and `results/the_gap.png`, which it writes)
- **Branch:** `ashrith/final-evaluation`
- **Start:** after Abhayanth's and Srijan's work is merged (you need their 4 prediction files in `results/`). Run `git pull` on `main` first. You can write the code earlier and run it once the files are there.

## Key ideas

- **Early point:** true RUL > 125, so the engine is still far from failure. **Late point:** true RUL ≤ 125, close to failure.
- Points are grouped by their **true** RUL, never by the prediction. Grouping by prediction would let the model choose its own group.
- 125 is borrowed from C-MAPSS papers (they cap RUL at 125–130). It is a convention, not a measured start of degradation, and it was not tuned.
- The **late** error matters most: near failure, a wrong RUL is the costly mistake.
- **The test set is scored only here, once.** After you see the test numbers, nobody changes any model.

## Setup (once)

Python 3.12 or newer.

```bash
git clone https://github.com/Abhayanthk/aml-rul-fd001.git
cd aml-rul-fd001
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git checkout -b ashrith/final-evaluation
```

## `notebooks/05_final_evaluation.ipynb`

First cell:
```python
import sys
sys.path.append("../src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_test_rul
from evaluation import CUTOFFS, mae, rmse

MODELS = ["ols_v1", "elastic_net_v2", "random_forest", "gradient_boosting"]
THRESHOLD = 125
```

Sections, in this order:

1. **Load the answers and the predictions:**
   ```python
   val = (CUTOFFS
          .merge(pd.read_csv("../results/val_predictions_linear.csv"), on=["engine_id", "cycle"])
          .merge(pd.read_csv("../results/val_predictions_trees.csv"), on=["engine_id", "cycle"]))
   test = (load_test_rul()
           .merge(pd.read_csv("../results/test_predictions_linear.csv"), on="engine_id")
           .merge(pd.read_csv("../results/test_predictions_trees.csv"), on="engine_id"))
   ```
   Print both shapes. Each must have 100 rows: the column `rul` (true answer) plus the 4 model columns.
2. **`score_by_band(y_true, y_pred)` function** (write it in the notebook):
   - Convert both inputs to NumPy arrays.
   - `early = y_true > THRESHOLD`, `late = ~early`.
   - Return `{"early": {"n": ..., "mae": ..., "rmse": ...}, "late": {...}}`, using `mae` and `rmse` from `evaluation.py`.
   - If a group has 0 points, give `n = 0` with MAE and RMSE `np.nan`. Don't call `mae` on an empty array.
3. **Final table:** one row per (set, model), 8 rows in all: validation × 4 models and test × 4 models. Columns exactly:
   `set, model, n, mae, rmse, early_n, early_mae, early_rmse, late_n, late_mae, late_rmse, late_over_pct`
   - `late_over_pct` = % of late points where prediction > true RUL, i.e. the model says more life is left than there is.
   
   Save to `results/final_results.csv` (`index=False`) and show it rounded to 2 decimals.
4. **The gap figure (validation):** a 2×2 figure, one panel per model. Each panel shows predicted RUL vs true RUL at the 100 validation points, with a diagonal line (perfect prediction) and a dashed vertical line at 125. Save it to `results/the_gap.png`.
5. **Where the models stop:** for the validation points with true RUL > 200, print how many there are, their mean true RUL, and each model's mean prediction for them. Also print each model's largest validation prediction.
6. **Findings:** a markdown cell with 4–5 short bullet points:
   - which model is best overall, and on late points;
   - early vs late error;
   - over-prediction near failure;
   - validation vs test;
   - the gap.
   
   Use only numbers from your table.

Finally, run **Kernel → Restart & Run All**.

## Check: your numbers must match

Validation (from the 4 prediction files):

| Model | MAE | RMSE | early n | early MAE | early RMSE | late n | late MAE | late RMSE | late over % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ols_v1 | 32.86 | 42.92 | 22 | 47.31 | 64.87 | 78 | 28.79 | 34.27 | 88 |
| elastic_net_v2 | 31.66 | 41.77 | 22 | 46.31 | 66.02 | 78 | 27.52 | 31.74 | 86 |
| random_forest | 25.03 | 40.17 | 22 | 49.33 | 69.59 | 78 | 18.17 | 26.50 | 73 |
| gradient_boosting | 26.12 | 40.17 | 22 | 50.92 | 69.54 | 78 | 19.13 | 26.54 | 69 |

Test (overall MAE / RMSE, the same as our earlier published results):

| Model | MAE | RMSE |
| --- | --- | --- |
| ols_v1 | 24.91 | 31.17 |
| elastic_net_v2 | 23.61 | 29.04 |
| random_forest | 18.95 | 25.88 |
| gradient_boosting | 18.54 | 25.33 |

Test early/late are new numbers, computed for the first time here. Check that `early_n + late_n = 100`, and that the counts are the same for all 4 models (the groups depend only on the true RUL).

Validation points with true RUL > 200: 5 points, mean true RUL 268.0. Mean prediction: OLS 147.7, EN 146.2, RF 144.7, GB 146.4. Largest validation prediction: OLS 177.1, EN 173.6, RF 187.2, GB 177.5.

## The gap (what the brief means)

The brief says: *"Fixed rolling features may miss long degradation trajectories and interactions whose timing differs across engines."*

In simple words:
- Our features summarise only the **last 10 cycles**, with formulas **we** chose (mean, std, slope).
- Wear builds up over 100–300 cycles and starts at a different time in each engine. A 10-cycle snapshot can't see how long an engine has been degrading or the shape of its history.

Your numbers show it:
1. **Predictions stop at about 175–190.** For engines with about 268 cycles left, every model predicts about 145, so they can't tell "200 left" from "300 left". That is why early RMSE is about 65–70.
2. **The models lean on age.** `cycle` is the top Random Forest feature (Srijan's notebook), so "how old is the engine" counts as much as the sensor history.

Phase 2 fix: an LSTM reads the raw sensor **sequence** and learns its own patterns over a longer window, instead of using our fixed summaries.

## Submit

```bash
git add notebooks/05_final_evaluation.ipynb results/final_results.csv results/the_gap.png
git commit -m "Add final evaluation: validation and test, early/late, the gap"
git push -u origin ashrith/final-evaluation
```

Open a pull request on GitHub. Touch only these files.

## Questions you should be able to answer

1. Why group by true RUL, not by the prediction? (Otherwise the model decides its own group, and a bad prediction could move a point into the easier group.)
2. Why 125? (Borrowed from papers that cap RUL at 125–130. It's a convention, not tuned on our results.)
3. Why is the test error lower than validation? (Validation includes very large RUL points, up to 329, which every model misses badly; the early points cause about two-thirds of the squared error. The likely reason is that test cut-offs sit closer to failure.)
4. Why does over-prediction matter? (It says there is more life left than there is. Near failure, that means maintenance comes too late.)
5. Why don't we cap RUL? (The brief says cap only if stated. Earlier tests showed capping helps late points but makes overall error worse. The early/late breakdown shows the same weakness without changing the answers.)
6. What is the gap? (Fixed 10-cycle summaries can't see the long history, so predictions stop at about 175–190. This is the reason for the Phase 2 LSTM.)

Course topics in your part: evaluation metrics (MAE, RMSE), bias–variance and error analysis.
