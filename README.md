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

## Structure

```
data/                         dataset (do not edit)
src/
  load_data.py                load files, RUL label, flat columns          Aryan
  features.py                 rolling mean / std / slope                    Akhil
  evaluation.py               engine split, validation points, MAE / RMSE   Abhayanth
notebooks/
  01_eda.ipynb                explore the data                              Aryan
  02_features.ipynb           build and explain the features                Akhil
  03_linear_models.ipynb      OLS and Elastic Net                           Abhayanth
  04_tree_models.ipynb        Random Forest and gradient boosting           Srijan
  05_final_evaluation.ipynb   all models on validation and test, early/late Ashrith
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
| Aryan | Data loading, RUL label, exploratory analysis, flat-sensor removal | [members/aryan.md](members/aryan.md) |
| Srijan | Tree models: Random Forest and gradient boosting | [members/srijan.md](members/srijan.md) |
| Abhayanth | Engine split, validation points, metrics, linear models | [members/abhayanth.md](members/abhayanth.md) |
| Akhil | Rolling features | [members/akhil.md](members/akhil.md) |
| Ashrith | Final evaluation: validation and test, early/late breakdown | [members/ashrith.md](members/ashrith.md) |

Work order (each part needs the previous one merged): Aryan → Akhil → Abhayanth → Srijan → Ashrith.

## Earlier work

Model settings were chosen on the validation set in our development repo, which also holds the earlier experiments (tuning grids, RUL cap test, feature comparisons):
https://github.com/Srijan76-code/aml_rul_project
