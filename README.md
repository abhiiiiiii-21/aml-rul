# Remaining Useful Life Prediction — NASA C-MAPSS FD001 (Phase 1, AML)

Predict the **Remaining Useful Life (RUL)** of an aircraft engine: how many more cycles it can run before it fails, using only the sensor readings seen so far.

Example: an engine fails at cycle 200 and we are at cycle 150, so its RUL is 50.

Phase 1 compares engineered-feature regression models: a linear baseline (OLS, Elastic Net), Random Forest and gradient boosting.

## Data (`data/`)

| File | What it is |
| --- | --- |
| `train_FD001.txt` | 100 engines, each run until it fails. 20,631 rows (one row = one engine at one cycle). |
| `test_FD001.txt` | 100 other engines, stopped some time before failure. 13,096 rows. |
| `RUL_FD001.txt` | The true RUL at the last row of each test engine (line 1 = engine 1). Used only in notebook 05. |
| `validation_cutoffs.csv` | The 100 fixed validation points: 20 engines × 5 cycles, chosen at random (seed 42) in our earlier work. Every model is scored on these same points. |

Each row has 26 columns: `engine_id`, `cycle`, 3 operating settings, 21 sensors.
Source: NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set (Saxena and Goebel, 2008).

## Method

```
load data ──> RUL label ──> rolling features ──> split by engine ──> models ──> final evaluation
   01             01               02                  03            03, 04          05
```

- **Target:** RUL = last cycle of the engine − current cycle. No cap.
- **Split:** by engine, never by row. 80 training engines, 20 validation engines.
- **Features:** cycle, 14 useful sensors, and each sensor's mean, std and slope over the last 10 cycles of the same engine (57 features). Features only use the current and earlier cycles.
- **Metrics:** MAE and RMSE in cycles, plus the same split into early points (true RUL > 125) and late points (true RUL ≤ 125).
- **Test set:** scored once, in notebook 05.
- **Columns removed:** `setting_3` and sensors 1, 5, 6, 10, 16, 18, 19 are constant or nearly constant. No missing values in the training or test data.

## Results

Validation scores on the 100 fixed points (lower is better, in cycles):

| Model | MAE | RMSE |
| --- | --- | --- |
| OLS Linear Regression | 32.86 | 42.92 |
| Elastic Net | 31.66 | 41.77 |
| Random Forest | **25.03** | 40.17 |
| Gradient Boosting | 26.12 | **40.17** |

- Random Forest has the lowest MAE, about 24% below OLS. Random Forest and Gradient Boosting tie on RMSE.
- Tree models cut MAE by about 5 to 8 cycles versus the linear baselines, so nonlinear sensor patterns matter.
- RMSE is much higher than MAE for every model: a few large errors dominate.
- Held-out test MAE so far (linear models only): OLS 24.91, Elastic Net 23.61. Tree-model test results are not yet recorded.

### Limitations

- All four models underpredict engines with a high true RUL (above 200). For the linear models, the mean error in the highest RUL band is about −120 cycles.
- The 10-cycle rolling window sees only recent behaviour. It does not capture the engine's longer degradation history.
- The high-RUL subset is small (5 validation points), so treat that result with caution.

### Next steps

1. Run notebook 05 to complete and verify the test evaluation for all models.
2. Try longer rolling windows and more temporal features.
3. Test a sequence model such as an LSTM for longer sensor histories (later phase).

## Structure

```
data/                         dataset (do not edit)
src/
  load_data.py                load files, RUL label, flat columns          Prateek
  features.py                 rolling mean / std / slope                    Yogesh
  evaluation.py               engine split, validation points, MAE / RMSE   Kapish
notebooks/
  01_eda.ipynb                explore the data                              Prateek
  02_features.ipynb           build and explain the features                Yogesh
  03_linear_models.ipynb      OLS and Elastic Net                           Kapish
  04_tree_models.ipynb        Random Forest and gradient boosting           Abhishek
  05_final_evaluation.ipynb   all models on validation and test, early/late Vikash Kumar
results/                      predictions and final results (written by the notebooks)
members/                      each member's instructions and check numbers
```

## How to run

Needs Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order: 01 → 02 → 03 → 04 → 05.

## Team

| Member | Part | Instructions |
| --- | --- | --- |
| Prateek | Data loading, RUL label, exploratory analysis, flat-sensor removal | [members/prateek.md](members/prateek.md) |
| Abhishek | Tree models: Random Forest and gradient boosting | [members/abhishek.md](members/abhishek.md) |
| Kapish | Engine split, validation points, metrics, linear models | [members/kapish.md](members/kapish.md) |
| Yogesh | Rolling features | [members/yogesh.md](members/yogesh.md) |
| Vikash Kumar | Final evaluation: validation and test, early/late breakdown | [members/vikash.md](members/vikash.md) |

Work order (each part needs the previous one merged): Prateek → Yogesh → Kapish → Abhishek → Vikash Kumar.

## Deliverables

- [Report.pdf](Report.pdf): full project report (method, results, team contributions, conclusions).
- [Presentation.pdf](Presentation.pdf): project slides (problem, data, exploration, features, models, results, next steps).

## Earlier work

Model settings were chosen on the validation set in our development repo, which also holds the earlier experiments (tuning grids, RUL cap test, feature comparisons):
https://github.com/Abhishek76-code/aml_rul_project
