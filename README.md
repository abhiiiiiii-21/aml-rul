# Remaining Useful Life Prediction — NASA C-MAPSS FD001 (Phase 1 & Phase 2)

Predict the **Remaining Useful Life (RUL)** of an aircraft engine: how many more operating cycles it can safely run before it fails, using multivariate sensor time series.

- **Phase 1 (Machine Learning):** Engineered rolling statistics (mean, std, slope over window 10) evaluated with OLS, Elastic Net, Random Forest, and Gradient Boosting.
- **Phase 2 (Deep Learning):** End-to-end deep sequence architectures (LSTM, GRU, 1D-CNN, Transformer), out-of-fold Hybrid Stacking, systematic ablation studies, and final cross-phase evaluation.

---

## Data (`data/`)

| File | Description | Dimensions |
| --- | --- | --- |
| `train_FD001.txt` | 100 engines run to failure under sea-level conditions | 20,631 rows $\times$ 26 cols |
| `test_FD001.txt` | 100 test engines censored prior to failure | 13,096 rows $\times$ 26 cols |
| `RUL_FD001.txt` | True RUL ground-truth values at the last observed cycle of test engines | 100 rows |
| `validation_cutoffs.csv` | 100 fixed validation cutoff points (20 engines $\times$ 5 cycles, seed 42) | 100 rows $\times$ 3 cols |

Each row contains: `engine_id`, `cycle`, 3 operational settings, and 21 sensor channels.
- 7 uninformative/constant channels removed (`setting_3` and sensors 1, 5, 6, 10, 16, 18, 19).
- 14 active, calibrated sensors retained for modeling.

---

## Final Performance Matrix (Validation & Held-Out Test)

All models are evaluated on the identical 100 fixed validation points and 100 held-out test engines against true uncapped RUL (cycles, lower is better):

| Phase | Model | Family | Val MAE | Val RMSE | Test MAE | Test RMSE | NASA Score ($S$) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | OLS Linear Regression | Classical Linear | 32.86 | 42.92 | 24.91 | 31.17 | 11,407 |
| 1 | Elastic Net | Regularized Linear | 31.66 | 41.77 | 23.61 | 29.04 | 5,620 |
| 1 | Random Forest | Tabular Ensemble | 25.03 | 40.17 | 18.95 | 25.88 | 6,048 |
| 1 | Gradient Boosting | Tabular Ensemble | 26.12 | 40.17 | 18.54 | 25.33 | 4,813 |
| **2** | **LSTM (`lstm_v1`)** | **Recurrent** | **19.41** | 34.45 | **16.10** | 23.66 | 14,308 |
| 2 | GRU (`gru_v1`) | Recurrent | 23.18 | 36.34 | 19.99 | 28.22 | 12,126 |
| 2 | 1D-CNN (`cnn1d_v1`) | Convolutional | 24.32 | 36.70 | 20.25 | 27.86 | 18,267 |
| 2 | Transformer (`transformer_v1`) | Self-Attention | 20.59 | 34.79 | 18.57 | 25.57 | 4,999 |
| **2** | **Hybrid Stacking (`hybrid_v1`)** | **Stacking Ensemble** | 21.48 | **34.38** | 17.07 | 24.13 | **4,312** |
| 2 | Hybrid Average (`hybrid_avg_v1`) | Simple Blend | 21.50 | 35.83 | 16.50 | **22.96** | 6,026 |

### Core Project Insights
- **Phase 2 Superiority:** The best deep learning model (**LSTM**) reduces validation MAE from 25.03 to **19.41 cycles** (a **22.5% improvement** over Phase 1 Random Forest). On the held-out test set, LSTM reaches **16.10 cycles MAE**.
- **Hybrid Innovation:** While LSTM achieves the lowest symmetric MAE, the **Hybrid Stacking model (`hybrid_v1`)** achieves the lowest overall **NASA risk score (4,312)** and validation RMSE (**34.38 cycles**), demonstrating superior stability and suppression of dangerous late overpredictions.
- **Ablation Findings:** 
  1. *Window length:* Window 30 is the empirical optimum (MAE 19.39 vs 25.40 at W=10).
  2. *Features:* 14 raw sensors outperform configurations adding operational settings or first-difference channels.
  3. *Capacity:* A compact 1-layer 64-hidden unit LSTM outperforms deeper multi-layer models on 80 training engine trajectories.

---

## Repository Map

```
aml-rul/
├── data/                             # NASA C-MAPSS FD001 dataset files (read-only)
├── src/
│   ├── load_data.py                  # Data loader and RUL label generation
│   ├── features.py                   # Phase 1 rolling statistical features (window 10)
│   ├── evaluation.py                 # Split logic, 100 benchmark points, MAE/RMSE
│   ├── sequences.py                  # Phase 2 sliding-window generator & standardizer
│   ├── dl_models.py                  # PyTorch models: LSTM, GRU, 1D-CNN, Transformer
│   ├── train.py                      # AdamW training loop, early stopping, inference
│   ├── hybrid.py                     # Out-of-fold stacking pipeline & feature fusion
│   └── ablation.py                   # Ablation feature helpers & evaluation wrappers
├── notebooks/                        # Phase 1 exploratory & modeling notebooks (01–05)
├── notebooks/phase2/                 # Phase 2 deep learning notebooks (01–07)
│   ├── 01_dl_literature_and_eda.ipynb      # Literature review & sequence EDA
│   ├── 02_sequence_windows.ipynb            # Sliding window pipeline validation
│   ├── 03_lstm_gru.ipynb                    # Recurrent sequence models (LSTM & GRU)
│   ├── 04_cnn_transformer.ipynb            # 1D-CNN & Transformer architectures
│   ├── 05_hybrid.ipynb                     # Out-of-fold Hybrid ML + DL stacking
│   ├── 06_ablations.ipynb                  # 7 controlled ablation studies (Vikash)
│   └── 07_final_evaluation.ipynb           # Consolidated evaluation on val & test (Vikash)
├── results/
│   ├── phase1/                       # Phase 1 validation and test prediction CSVs
│   └── phase2/                       # Phase 2 predictions, ablation tables, models, figures
│       ├── figures/                  # Publication-ready diagnostic figures (150+ DPI)
│       ├── models/                   # Saved PyTorch model weights (.pt)
│       ├── ablation_results.csv      # Individual ablation experiment trials
│       ├── ablation_summary.csv      # Aggregated ablation statistics across seeds
│       ├── final_results.csv         # Definitive cross-phase benchmark matrix
│       └── test_bootstrap.csv        # 95% bootstrap confidence intervals
├── report/                           # IEEE Conference LaTeX Report
│   ├── main.tex                      # Master LaTeX document
│   ├── references.bib                # Verified peer-reviewed bibliography
│   └── sections/                     # Modular LaTeX section files
├── docs/
│   ├── PHASE2_PLAN.md                # Phase 2 architectural roadmap & execution status
│   ├── presentation_outline.md       # 15-slide presentation guide with speaker notes
│   ├── VIVA_PREP.md                  # Comprehensive viva voce preparation guide
│   └── handoff/                      # Formal handoff notes from each team member
└── tests/                            # Pytest test suite
```

---

## Team & Responsibilities

| Role | Member | Responsibilities | Handoff |
| --- | --- | --- | --- |
| 1 | **Prateek Shakya** | Literature review, sequence EDA, windowing pipeline, test suite | [`docs/handoff/01_prateek.md`](docs/handoff/01_prateek.md) |
| 2 | **Yogesh Modi** | Recurrent models (LSTM & GRU), training loop, regularisation sweeps | [`docs/handoff/02_yogesh.md`](docs/handoff/02_yogesh.md) |
| 3 | **Abhishek Kumar Patel** | 1D-CNN, Transformer with self-attention, attention interpretability | `notebooks/phase2/04` |
| 4 | **Kapish Rohilla** | Out-of-fold Hybrid ML + DL stacking, variance analysis | `notebooks/phase2/05` |
| 5 | **Vikash Kumar** | Ablation studies, final evaluation, integration, report, viva pack | [`docs/handoff/05_vikash.md`](docs/handoff/05_vikash.md) |

---

## How to Run & Reproduce

### 1. Environment Setup
```bash
git clone https://github.com/Abhishek76-code/aml-rul.git
cd aml-rul
python3 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest -q
```

### 3. Run Phase 2 Notebooks
To execute the notebooks linearly from the command line:
```bash
cd notebooks/phase2
jupyter nbconvert --to notebook --execute 01_dl_literature_and_eda.ipynb --inplace
jupyter nbconvert --to notebook --execute 02_sequence_windows.ipynb --inplace
jupyter nbconvert --to notebook --execute 03_lstm_gru.ipynb --inplace
jupyter nbconvert --to notebook --execute 04_cnn_transformer.ipynb --inplace
jupyter nbconvert --to notebook --execute 05_hybrid.ipynb --inplace
jupyter nbconvert --to notebook --execute 06_ablations.ipynb --inplace
jupyter nbconvert --to notebook --execute 07_final_evaluation.ipynb --inplace
```
All notebooks execute top-to-bottom cleanly in under 15 minutes each.
