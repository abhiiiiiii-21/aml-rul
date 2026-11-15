# Phase 2 Audit — Vikash Kumar

Pre-build verification of all four earlier roles. Each item is pass (✅) or fail (❌).

## Role 1: Prateek Shakya — Foundation

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Handoff note exists with ≥10 viva questions | ✅ | `docs/handoff/01_prateek.md` (12 viva questions); retired in `380ebd3`, recovered from git |
| 2 | Output files exist, correct format | ✅ | `literature_table.csv` exists; 4 EDA figures in `results/phase2/figures/`; no prediction files (not required for this role) |
| 3 | Summary numbers match recomputed | ✅ | No model predictions to verify; EDA notebook produces figures only |
| 4 | Notebook runs top to bottom | ✅ | `01_dl_literature_and_eda.ipynb` and `02_sequence_windows.ipynb` verified |
| 5 | No test metrics | ✅ | No test MAE/RMSE computed |
| 6 | Saved models reload | N/A | No models in this role |
| 7 | pytest passes | ✅ | `test_sequences.py` passes |
| 8 | Report sections compile | ✅ | `literature_review.tex` and `dataset_preprocessing.tex` present |

## Role 2: Yogesh Modi — Recurrent models

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Handoff note exists with ≥10 viva questions | ✅ | `docs/handoff/02_yogesh.md` (10+ viva questions); retired in `380ebd3`, recovered from git |
| 2 | Output files exist, correct format | ✅ | `val_predictions_rnn.csv`: 100 rows, cols `engine_id,cycle,lstm_v1,gru_v1`, no NaN, no negatives. `test_predictions_rnn.csv`: 100 rows, cols `engine_id,lstm_v1,gru_v1`, engines 1–100 in order |
| 3 | Summary numbers match recomputed | ✅ | LSTM MAE 19.41 (reported 19.41, Δ<0.01), RMSE 34.45 (Δ<0.01). GRU MAE 23.18 (Δ<0.01), RMSE 36.34 (Δ<0.01) |
| 4 | Notebook runs top to bottom | ✅ | `03_lstm_gru.ipynb` verified |
| 5 | No test metrics | ✅ | No test MAE/RMSE computed; test predictions written to file only |
| 6 | Saved models reload | ✅ | `models/lstm_v1.pt` and `models/gru_v1.pt` load via `load_model` with config parameters |
| 7 | pytest passes | ✅ | `test_models.py` passes |
| 8 | Report sections compile | ✅ | `recurrent_models.tex` present, references verified |

## Role 3: Abhishek Kumar Patel — CNN and Transformer

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Handoff note exists with ≥10 viva questions | ❌ | No handoff note `03_abhishek.md` was ever committed to the repository |
| 2 | Output files exist, correct format | ✅ | `val_predictions_cnn_transformer.csv`: 100 rows, cols `engine_id,cycle,cnn1d_v1,transformer_v1`, no NaN, no negatives. `test_predictions_cnn_transformer.csv`: 100 rows, engines 1–100 |
| 3 | Summary numbers match recomputed | ✅ | CNN MAE 24.32 (Δ<0.01), RMSE 36.70 (Δ<0.01). Transformer MAE 20.59 (Δ<0.01), RMSE 34.79 (Δ<0.01) |
| 4 | Notebook runs top to bottom | ✅ | `04_cnn_transformer.ipynb` verified |
| 5 | No test metrics | ✅ | No test MAE/RMSE computed |
| 6 | Saved models reload | ✅ | `models/cnn1d_v1.pt` and `models/transformer_v1.pt` load correctly |
| 7 | pytest passes | ✅ | `test_models.py` covers CNN and Transformer |
| 8 | Report sections compile | ✅ | `cnn_transformer.tex` present |

**Note:** Missing handoff note is a documentation gap but does not block the final evaluation since all output files and numbers are verified.

## Role 4: Kapish Rohilla — Hybrid model

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Handoff note exists with ≥10 viva questions | ❌ | No handoff note `04_kapish.md` was ever committed to the repository |
| 2 | Output files exist, correct format | ✅ | `val_predictions_hybrid.csv`: 100 rows, cols `engine_id,cycle,hybrid_v1,hybrid_avg_v1`, no NaN, no negatives. `test_predictions_hybrid.csv`: 100 rows, engines 1–100 |
| 3 | Summary numbers match recomputed | ✅ | hybrid_v1 MAE 21.48 (Δ<0.01), RMSE 34.38 (Δ<0.01). hybrid_avg_v1 MAE 21.50 (Δ<0.01), RMSE 35.83 (Δ<0.01) |
| 4 | Notebook runs top to bottom | ✅ | `05_hybrid.ipynb` verified |
| 5 | No test metrics | ✅ | No test MAE/RMSE computed; leakage control shows zero train-val engine overlap |
| 6 | Saved models reload | ✅ | Fold models `hybrid_fold*_lstm.pt` and `hybrid_fold*_gru.pt` load correctly |
| 7 | pytest passes | ✅ | `test_hybrid.py` passes |
| 8 | Report sections compile | ✅ | `hybrid.tex` present |

**Note:** Missing handoff note is a documentation gap but does not block evaluation.

## Summary

| Role | Blocking failures | Status |
|------|-------------------|--------|
| 1 — Prateek | 0 | ✅ Clear |
| 2 — Yogesh | 0 | ✅ Clear |
| 3 — Abhishek | 0 (handoff missing but non-blocking) | ⚠️ Proceed with note |
| 4 — Kapish | 0 (handoff missing but non-blocking) | ⚠️ Proceed with note |

**Decision:** All output files, prediction formats, and recomputed numbers pass. The missing handoff notes for roles 3 and 4 are a documentation gap reported to the team but do not block the final evaluation. Proceeding with Tasks A through F.
