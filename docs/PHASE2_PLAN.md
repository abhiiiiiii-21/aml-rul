# Phase 2 plan: deep learning on FD001

Phase 2 is graded on: architecture logic, hybrid innovation, ablation studies, deep learning literature review, DL dataset and regularization, technical validation, theoretical rigor, repository and code quality, presentation, viva (individual) and an IEEE LaTeX report.

## Ground rules (keep results comparable with Phase 1)

- Same split: 80 training engines, 20 validation engines (`src/evaluation.py`).
- Same 100 fixed validation points. Every model is scored on them with MAE and RMSE, plus early (RUL > 125) and late (RUL <= 125) breakdown.
- Scaler is fit on the 80 training engines only. Never fit on validation or test.
- Test set is scored once, in `07_final_evaluation.ipynb`.
- Seed 42 everywhere. Save predictions to `results/phase2/` in the same shape as Phase 1.
- Phase 1 numbers to beat (validation MAE): Random Forest 25.03, Gradient Boosting 26.12, Elastic Net 31.66, OLS 32.86.

## Shared code (`src/`)

| File | Purpose |
| --- | --- |
| `sequences.py` | Scaler and sliding windows (default 30 cycles, left-padded). |
| `dl_models.py` | `RNNRegressor` (LSTM, GRU), `CNN1DRegressor`, `build_model`. Add new models here. |
| `train.py` | `fit` (AdamW, weight decay, early stopping, grad clipping), `predict`, `set_seed`. |

## Notebooks and proposed owners

Owners are a proposal. Change them as a team and update this table and the README.

| Notebook | Content | Proposed owner |
| --- | --- | --- |
| `01_dl_literature_and_eda` | DL literature review for RUL, sequence-level EDA | Prateek Shakya |
| `02_sequence_windows` | Windowing, scaling, sanity checks (no look-ahead) | Prateek Shakya |
| `03_lstm_gru` | LSTM and GRU baselines, regularization (dropout, weight decay, early stopping) | Kapish Rohilla |
| `04_cnn_transformer` | 1D-CNN and Transformer encoder | Yogesh Modi |
| `05_hybrid` | Hybrid ML + DL (see ideas below) | Abhishek Kumar Patel |
| `06_ablations` | Window length, depth, hidden size, dropout, RUL cap, with/without features | Vikash Kumar |
| `07_final_evaluation` | All models on validation and test, early/late breakdown | Vikash Kumar |
| `report/` | IEEE LaTeX report, slides | everyone, one section each |

## Hybrid ideas (must be justified and actually work)

- Learned sequence embedding from the LSTM/GRU/CNN fed to Random Forest or Gradient Boosting together with the Phase 1 rolling features.
- Stack or blend the best DL model with the best tree model.
- Use a DL model where Phase 1 failed: engines with true RUL above 200, which all four ML models underpredicted.

## Viva

Every member must be able to explain the whole pipeline, not only their notebook: windowing, model equations (LSTM gates, backpropagation through time), loss, regularization, validation protocol, results.
