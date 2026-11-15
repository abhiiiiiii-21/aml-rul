# Remaining Useful Life Prediction on NASA C-MAPSS FD001

Predicting how many operating cycles a turbofan engine can still run before it fails, from the sensor readings seen so far. The project compares classical machine learning on hand-crafted rolling features (Phase 1) with deep sequence models and a hybrid stacking model (Phase 2), all evaluated under one strict protocol.

| | |
| --- | --- |
| **Task** | Regression of RUL in cycles, no cap on the target |
| **Data** | NASA C-MAPSS FD001: 100 training and 100 test engines, 14 informative sensors |
| **Models** | OLS, Elastic Net, Random Forest, Gradient Boosting; LSTM, GRU, 1D-CNN, Transformer; hybrid stacking |
| **Metrics** | MAE and RMSE in cycles, early/late split, NASA score, bootstrap confidence intervals |
| **Stack** | Python 3.12+, PyTorch, scikit-learn, pandas, Jupyter |

## Results

MAE and RMSE in cycles (lower is better). Validation is the same 100 fixed points for every model; test is the last observed cycle of each of the 100 test engines, scored once. Phase 2 validation and test numbers are for the seed-42 models that were saved.

| Phase | Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
| --- | --- | --- | --- | --- | --- |
| 1 | OLS regression | 32.86 | 42.92 | 24.91 | 31.17 |
| 1 | Elastic Net | 31.66 | 41.77 | 23.61 | 29.04 |
| 1 | Random Forest | 25.03 | 40.17 | 18.95 | 25.88 |
| 1 | Gradient Boosting | 26.12 | 40.17 | 18.54 | 25.33 |
| 2 | LSTM (1 layer) | 19.41 | 34.45 | 16.10 | 23.66 |
| 2 | GRU (2 layers) | 23.18 | 36.34 | 19.99 | 28.22 |
| 2 | 1D-CNN (dilated) | 24.32 | 36.70 | 20.25 | 27.86 |
| 2 | Transformer (3 layers) | 20.59 | 34.79 | 18.57 | 25.57 |
| 2 | Hybrid stacking | 21.48 | 34.38 | 17.07 | 24.13 |
| 2 | Hybrid average | 21.50 | 35.83 | 16.50 | 22.96 |

What the results support:

- **The single-layer LSTM is the best model.** Validation MAE is 19.41 for the saved model and 20.14 ± 1.46 over three seeds, against 25.03 for the best Phase 1 model (Random Forest). Test MAE is 16.10 against 18.95.
- **That advantage is not statistically established on the test set.** Choosing the best models on validation, the paired bootstrap 95% interval of the test MAE difference is [-0.14, 5.84] cycles, which includes 0. With 100 engines the intervals are wide.
- **Phase 2 is not uniformly better than Phase 1.** On the test set the GRU (19.99) and the CNN (20.25) are worse than both tree models, and the Transformer (18.57) is level with them.
- **The hybrid does not beat the best deep model**, and stacking is no better than a plain average. It improves a tree model: Random Forest on rolling features alone is 25.08 on validation, the stacked hybrid is 21.72 (three seeds). Its low NASA score (4,312) comes with over-prediction of 84% of the late test points, so it is not evidence of safer predictions.
- **The high-RUL gap is not closed.** Engines with a true RUL above 200 are underpredicted by 110 to 125 cycles by every model (5 validation points; the test set has no such engine).

Ablations (100-point MAE, three seeds, reference LSTM 20.14 ± 1.46): a window of 10 or 20 cycles is clearly worse (27.60, 23.39) and 30 to 50 cycles are indistinguishable; one recurrent layer is best and extra depth hurts; first-difference channels and target caps (125, 150) are worse; early stopping is the regularizer that matters, while dropout and weight decay are within seed noise. Full tables are in `results/phase2/ablation_summary.csv` and the report.

## Method

**Protocol (the same for every model).**
- Split by engine, never by row: 80 training engines and 20 validation engines.
- The 100 validation points (20 engines x 5 cycles) are fixed in `data/validation_cutoffs.csv`.
- Scalers, encoders and models are fitted on the 80 training engines only. Validation engines are used for early stopping and model selection.
- Windows contain only the current and earlier cycles of one engine (no look-ahead). Evaluation always uses the true, uncapped RUL.
- The test set is scored once, in notebook 07, and is never used to choose a model.
- Seed 42 for the main runs; seeds 42, 43, 44 for spread estimates.

**Phase 1.** Rolling mean, standard deviation and slope over the last 10 cycles of 14 sensors (57 features) feed OLS, Elastic Net, Random Forest and Gradient Boosting.

**Phase 2.**
- *Sequence models:* 30-cycle windows of the 14 standardised sensors feed an LSTM, a GRU, a dilated 1D-CNN or a small Transformer encoder. Training uses AdamW, MSE on RUL/100, gradient clipping and early stopping on the validation engines.
- *Hybrid:* out-of-fold RUL estimates from the LSTM and GRU (5 folds by engine, so no training row is predicted by a model that saw its engine) are stacked with the Phase 1 rolling features in a Random Forest. A plain average and a residual variant serve as comparisons.
- *Ablations:* one factor at a time around the LSTM reference (window, features, target cap, capacity, regularization, architecture, hybrid components).

## Repository structure

```
data/                 NASA C-MAPSS FD001 files and the 100 fixed validation cutoffs
src/
  load_data.py        loading, RUL label, the 14 useful sensors
  features.py         Phase 1 rolling features
  evaluation.py       engine split, validation and test points, MAE/RMSE
  sequences.py        sliding windows and scaler for the deep models
  dl_models.py        LSTM/GRU, 1D-CNN and Transformer regressors
  train.py            training loop, early stopping, prediction, save/load
  rnn_experiments.py  runner used by notebook 03
  cnn_experiments.py  runner used by notebook 04
  hybrid.py           out-of-fold stacking pipeline
  ablation.py         feature variants and cached ablation runner
tests/                pytest suite (windowing, models, hybrid leakage, ablation helpers)
notebooks/            Phase 1 notebooks 01-05
notebooks/phase2/     Phase 2 notebooks 01-07
results/phase1/       Phase 1 predictions and metrics
results/phase2/       Phase 2 predictions, configs, weights, tables, figures
report/               IEEE-format LaTeX report (main.tex, sections/, references.bib)
Report.pdf, Presentation.pdf   Phase 1 report and slides
```

## Getting started

```bash
git clone https://github.com/abhiiiiiii-21/aml-rul.git
cd aml-rul
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # unit tests
```

Run the notebooks from inside their folder, in order:

```bash
cd notebooks/phase2
jupyter nbconvert --to notebook --execute 01_dl_literature_and_eda.ipynb --inplace
jupyter nbconvert --to notebook --execute 02_sequence_windows.ipynb --inplace
jupyter nbconvert --to notebook --execute 03_lstm_gru.ipynb --inplace          # about 9 min
jupyter nbconvert --to notebook --execute 04_cnn_transformer.ipynb --inplace   # about 21 min
jupyter nbconvert --to notebook --execute 05_hybrid.ipynb --inplace
jupyter nbconvert --to notebook --execute 06_ablations.ipynb --inplace         # about 11 min first run, seconds from the cache
jupyter nbconvert --to notebook --execute 07_final_evaluation.ipynb --inplace  # seconds
```

Timings are from an 8-core laptop with 6 worker processes. Notebook 07 only reads the prediction files written by the earlier notebooks. The trained weights are in `results/phase2/models/`; load them with `train.load_model` and the settings in the matching `config_*.json`.

## Reproducibility

- Training runs on CPU with fixed seeds and reproduces the stored numbers (the ablation notebook checks that its seed-42 runs match the model notebooks).
- Each result file records its configuration (`config_*.json`).
- The LaTeX report is in `report/`; build it with a TeX distribution (`pdflatex`/`bibtex` or `latexmk -pdf main.tex` inside `report/`). It was not compiled in the environment used to write it.

## Limitations

- One dataset subset (FD001, one operating condition, one fault mode); FD002-FD004 are not tested.
- 80 training, 20 validation and 100 test engines: wide confidence intervals; validation scores are slightly optimistic because early stopping and selection used the validation engines.
- Standard deviations rest on three seeds.
- Underprediction of healthy engines (true RUL above 200) remains, and the test set cannot measure it.

## Team

| Member | Phase 1 | Phase 2 |
| --- | --- | --- |
| Prateek Shakya | Data loading, RUL label, exploration | Literature review, sequence EDA, windowing pipeline and tests |
| Yogesh Modi | Rolling features | LSTM and GRU, training loop, regularization study |
| Abhishek Kumar Patel | Tree models | 1D-CNN and Transformer |
| Kapish Rohilla | Engine split, metrics, linear models | Hybrid stacking model |
| Vikash Kumar | Final evaluation | Ablations, final evaluation, report integration |

## Data and references

- A. Saxena and K. Goebel, "Turbofan Engine Degradation Simulation Data Set," NASA Ames Prognostics Data Repository, 2008. The data files in `data/` come from this repository.
- The full bibliography is in `report/references.bib`.
