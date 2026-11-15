# Phase 2 Plan & Integration Status — Deep Learning for RUL

## Overview

Phase 2 replaces the engineered-feature tabular models from Phase 1 with end-to-end deep learning architectures and an integrated hybrid model on NASA C-MAPSS FD001.

### Team Roles & Execution Order

| Step | Member | Role | Branch | Status |
| --- | --- | --- | --- | --- |
| 1 | Prateek Shakya | Foundation: EDA, windowing pipeline, test suite, literature review | `phase2/prateek` | ✅ Merged |
| 2 | Yogesh Modi | Recurrent models: LSTM, GRU, training protocol, regularization sweeps | `phase2/yogesh` | ✅ Merged |
| 3 | Abhishek Kumar Patel | 1D-CNN and Transformer architectures | `phase2/abhishek` | ✅ Merged |
| 4 | Kapish Rohilla | Hybrid ML + DL stacking model (RF over OOF DL embeddings + rolling features) | `phase2/kapish` | ✅ Merged |
| 5 | Vikash Kumar | Ablation studies, final evaluation, integration, report, presentation, viva pack | `phase2/vikash` | 🔄 In Progress |

---

## Architectural Conventions & Rules

1. **Engine Split:** 80 training engines, 20 validation engines (`split_train_val` in `src/evaluation.py`).
2. **Fixed Evaluation Points:** Exactly 100 points (`validation_points(val_df)`) matching Phase 1 for all validation comparisons.
3. **Target:** Uncapped true RUL in cycles (`rul = last_cycle - current_cycle`). Training uses normalized targets (`RUL / 100.0`).
4. **Causal Windows:** Sliding windows of length 30 (`DEFAULT_WINDOW`). Left-padding for early cycles using the first observed row. No future look-ahead.
5. **Standardization:** Per-sensor `(mean, std)` fitted on the 80 training engines only. Zero variance sensors mapped to std = 1.0.
6. **Held-out Test Evaluation:** 100 test engines evaluated once at their last observed cycle in `07_final_evaluation.ipynb`.

---

## Model Inventory & Baseline Metrics (100 Fixed Validation Points)

| Model | Type | Architecture / Hyperparameters | Val MAE (cycles) | Val RMSE (cycles) |
| --- | --- | --- | --- | --- |
| Random Forest (P1) | Tabular Tree | 57 rolling features, n_estimators=100 | 25.03 | 40.17 |
| Gradient Boosting (P1) | Tabular Tree | 57 rolling features, lr=0.05, depth=3 | 26.12 | 40.17 |
| **LSTM** (P2) | Recurrent | 1 layer, 64 hidden, dropout 0.2 | **19.41** | **34.45** |
| GRU (P2) | Recurrent | 2 layers, 64 hidden, dropout 0.3 | 23.18 | 36.34 |
| 1D-CNN (P2) | Convolutional | 2 layers, 64 channels, kernel 5, dropout 0.2 | 24.32 | 36.70 |
| Transformer (P2) | Self-Attention | 2 layers, 32 d_model, 4 heads, dropout 0.1 | 20.59 | 34.79 |
| **Hybrid Stacking** (P2) | Stacking Ensemble | RF meta-learner over OOF DL + rolling features | 21.48 | 34.38 |
| Hybrid Average (P2) | Simple Blend | Average of LSTM and GRU predictions | 21.50 | 35.83 |

---

## Ablation Study Matrix (Vikash Kumar)

Seven systematic ablations evaluated with reference model (LSTM 1x64, window 30, seed 42):
1. **Window Length:** $\{10, 20, 30, 40, 50\}$ cycles across seeds $\{42, 43, 44\}$.
2. **Input Features:** 14 sensors vs. 16 features (+settings) vs. 28 features (+first differences).
3. **RUL Target Capping:** No cap vs. cap at 125 vs. cap at 150.
4. **Model Capacity:** Hidden size $\{32, 64, 128\} \times$ layers $\{1, 2, 3\}$.
5. **Regularization:** Dropout $\{0.0, 0.1, 0.2, 0.3, 0.4\} \times$ Weight Decay $\{0, 10^{-4}, 10^{-3}, 10^{-2}\}$.
6. **Architecture Comparison:** LSTM vs. GRU vs. 1D-CNN vs. Transformer.
7. **Hybrid Components:** Rolling features only vs. DL embeddings only vs. Combined.

---

## Deliverables Checklist

- [x] Pre-build audit (`docs/handoff/05_audit.md`)
- [x] Ablation helper module (`src/ablation.py`)
- [x] Run all ablations & final evaluations (`results/phase2/ablation_results.csv`, `final_results.csv`, `test_bootstrap.csv`)
- [x] Generate all figures in `results/phase2/figures/`
- [ ] Runnable notebook `06_ablations.ipynb`
- [ ] Runnable notebook `07_final_evaluation.ipynb`
- [x] IEEE LaTeX Report integration (`report/main.tex` and all section files)
- [ ] Presentation outline (`docs/presentation_outline.md`)
- [ ] Comprehensive viva preparation guide (`docs/VIVA_PREP.md`)
- [ ] Updated root `README.md`
- [ ] Final handoff documentation (`docs/handoff/05_vikash.md`)
