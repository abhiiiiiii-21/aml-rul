# 02 Yogesh Handoff

## What was built

- Extended [`src/train.py`](../../src/train.py), backward compatible: `fit` gained `rul_cap`, `lr_schedule` (ReduceLROnPlateau, default off), `grad_clip` (None disables clipping, the gradient norm is always logged) and `restore_best`; its history now holds `val_loss`, `grad_norm`, `best_epoch` and `epochs_run`. New helpers: `evaluate_points`, `save_model`, `load_model`.
- Added [`src/rnn_experiments.py`](../../src/rnn_experiments.py): `run_one` trains and scores one LSTM/GRU configuration on CPU; `run_many` runs configurations in parallel worker processes. Used only by notebook 03.
- Notebook [`notebooks/phase2/03_lstm_gru.ipynb`](../../notebooks/phase2/03_lstm_gru.ipynb): baselines, one-factor regularization and capacity sweeps, gradient-norm study, selection, three seeds, outputs, analysis.
- Results in `results/phase2/`: `val_predictions_rnn.csv`, `test_predictions_rnn.csv` (`lstm_v1`, `gru_v1`), `config_lstm.json`, `config_gru.json`, `models/lstm_v1.pt`, `models/gru_v1.pt`, `summary_rnn.csv`, `rnn_regularization.csv`, and five `figures/rnn_*.png`.
- Report section [`report/sections/recurrent_models.tex`](../../report/sections/recurrent_models.tex) and four verified bibliography entries appended to `report/references.bib`.
- `RNNRegressor` in `src/dl_models.py` was not changed; `encode` still returns the last-step hidden vector, shape `(batch, hidden)`.

## Key numbers

All from `results/phase2/` (100 fixed validation points, cycles). Phase 1 Random Forest for reference: MAE 25.03, RMSE 40.17.

| Model | Configuration | MAE (3 seeds) | RMSE (3 seeds) | Early MAE | Late MAE |
| --- | --- | --- | --- | --- | --- |
| `lstm_v1` | 1 layer, 64 hidden, dropout 0.2 | 20.14 +/- 1.46 | 33.18 +/- 2.46 | 43.73 | 12.55 |
| `gru_v1` | 2 layers, 64 hidden, dropout 0.3 | 22.98 +/- 0.49 | 36.61 +/- 0.90 | 43.34 | 17.49 |

Early and late MAE and the saved predictions are for the seed 42 runs (MAE 19.41 and 23.18). Standard deviations use `ddof=1` over seeds 42, 43, 44.

- Window MAE on the 20 validation engines (the selection score): baseline GRU 26.87, baseline LSTM 28.42; chosen GRU 26.78, chosen LSTM 26.38.
- No early stopping (60 epochs, last epoch kept): 31.15 (GRU), 31.95 (LSTM). Spread over dropout 0.38 / 0.68 and weight decay 0.07 / 0.04 cycles (GRU / LSTM).
- Mean error for true RUL above 200 (5 points): LSTM -124.7, GRU -115.8 cycles.
- Parameters: LSTM 20,545 (chosen) and 53,825 (2 x 64); GRU 40,385. Seconds per epoch at 2 x 64, one CPU thread: LSTM 1.48, GRU 1.14.
- Peak mean gradient norm with clipping at 1.0: GRU 1.57, LSTM 1.31; without clipping 1.13 and 0.99.

## Decisions and why

- **CPU only, seeded before model creation.** CPU runs are deterministic and as fast as MPS for these sizes (about 1.4 s per epoch). A first run built the model before seeding, so initial weights differed per process and results did not reproduce; `run_one` now calls `set_seed` first and a full rerun reproduced every number.
- **Parallel runs.** About 40 training runs take around 9 minutes with 6 worker processes of one thread each; serially they would exceed the 15-minute target.
- **Selection rule written first** (see the notebook Method): lowest all-windows MAE of the 20 validation engines, ties under 0.05 cycles to the smaller model, pool = baseline, one-factor runs, one combined run. This chose a one-layer LSTM, so the LSTM and GRU in the comparison differ in depth; the matched 2 x 64 timing and baselines are shown for a fair cell-type comparison.
- **`rnn_regularization.csv` has extra columns** (`model`, `train_mae`, `best_epoch`, `epochs_run`, `n_params`) beyond the four the prompt lists, and `summary_rnn.csv` has `mae_mean_3seeds` and `rmse_mean_3seeds`. The required columns come first, in the required order.
- **No-clipping run is a diagnostic**, excluded from the selection pool.
- `docs/` is git-ignored; this note was added with `git add -f`, like the previous handoff.

## How to reproduce

From `aml-rul/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
cd notebooks/phase2
../../.venv/bin/jupyter nbconvert --to notebook --execute 03_lstm_gru.ipynb --inplace --ExecutePreprocessor.timeout=3000
```

The notebook takes about 9 minutes on an 8-core laptop. To load a saved model: `train.load_model("lstm", "results/phase2/models/lstm_v1.pt", hidden=64, layers=1, dropout=0.2)` (values are in `config_lstm.json`); GRU: `hidden=64, layers=2, dropout=0.3`.

## Known problems or open questions

- The LaTeX section was not compiled: no TeX installation was available here. Figure paths are relative to `report/` (`../results/phase2/figures/...`); check them when `main.tex` is assembled.
- Validation scores are slightly optimistic (early stopping and selection used the 20 validation engines, and the winner of about 16 single-seed runs per model is chosen). The standard deviation reflects training seeds only, not the sampling noise of 100 points from 20 engines.
- Single-factor differences below about 1 cycle are within seed noise (chosen-configuration window MAE varies by about 0.4 to 0.8 across seeds).
- Patience 3, 8 and 15 follow the same training path (same seed); they differ only when training stops before the best epoch.
- Underprediction above 200 cycles is not fixed (5 points); likely an information limit of the 30-cycle window. A hybrid model or a longer window may help.
- No test metric was computed anywhere; test predictions are only written to `test_predictions_rnn.csv`.
- No file of another member was edited.

## Viva preparation

1. **Why does the LSTM have a cell state?** The cell state `c_t = f_t * c_{t-1} + i_t * g_t` is an additive path. Its derivative along time is the forget gate (between 0 and 1) instead of a repeated weight matrix, so gradients over 30 cycles are not forced to vanish. The gates decide what to keep, write and read.
2. **Why do gradients vanish or explode in a plain RNN?** BPTT multiplies one Jacobian per step: the product over 30 steps shrinks geometrically if each factor has norm below 1 and grows if above 1. Distant cycles then stop influencing the weights (vanishing) or training diverges (exploding).
3. **What does gradient clipping do, and did it matter here?** It rescales the gradient to norm at most 1.0 when it is larger, keeping the direction. It protects against exploding steps. Here the mean gradient norm peaked at 1.57 (GRU) and 1.31 (LSTM) and validation MAE with and without clipping differed by less than seed noise, so it was cheap insurance, not a deciding factor.
4. **Why is weight decay in AdamW "decoupled"?** The decay `lambda * theta` is subtracted from the weights directly, outside the adaptive `m / sqrt(v)` scaling. With L2 added to the loss, the penalty would be rescaled by Adam's per-weight step size, so its strength would depend on the gradient history. Decoupling makes it independent of the learning-rate scaling.
5. **Why did weight decay change almost nothing?** The shrink per step is learning rate times decay, 1e-3 x 1e-4 = 1e-7 of each weight. Over a few thousand steps this is negligible compared with the gradient updates, so validation MAE moved by 0.04 to 0.07 cycles.
6. **Why split by engine and not by row?** Windows of one engine overlap by 29 of 30 cycles and follow one degradation path, so row-split validation windows would have near-copies in training and the error would be far too optimistic. The effective sample size is about 80 engines, not 16,390 windows.
7. **Why does early stopping make validation scores optimistic?** The epoch (and the configuration) is chosen by the score on the 20 validation engines, so those engines influence the model. The score is the best of many looks. The test set, never used for any choice, is the unbiased check.
8. **LSTM versus GRU?** The LSTM has a cell state and three gates plus a candidate (four weight blocks); the GRU merges cell and hidden state and has an update and a reset gate (three blocks), so it has about 25% fewer parameters (40,385 against 53,825 at 2 x 64) and trained faster (1.14 against 1.48 s per epoch). Accuracy was close at the same size (seed 42 MAE 22.83 GRU, 21.83 LSTM) and the chosen LSTM was better (20.14 against 22.98), but with large seed spread the evidence is weak.
9. **Why was no early stopping so much worse?** With 80 engines the network memorises their trajectories: training MAE fell to about 7 cycles while validation-window MAE rose from 26.87 to 31.15 (GRU). Early stopping picks the epoch before that happens.
10. **How was the final configuration chosen, and why not with the 100 points?** By the lowest MAE over all windows of the 20 validation engines, ties under 0.05 cycles to the smaller model, rule written before results. 100 points are too few and noisy to select from, and selecting on them would also contaminate the number reported for them.
11. **Why 20.14 +/- 1.46 and not just 19.41?** The saved model is the seed 42 run (19.41). The mean and standard deviation over seeds 42 to 44 show how much one training run can vary: 1.46 cycles for the LSTM, which is as large as many sweep differences.
12. **Do the models beat Random Forest and do they fix the high-RUL problem?** They beat it on validation: MAE 20.14 and 22.98 against 25.03, RMSE 33.18 and 36.61 against 40.17, gaps larger than the seed standard deviation. They do not fix the high-RUL underprediction: the mean error above 200 cycles is -124.7 and -115.8 on 5 points, as in Phase 1.
13. **What bug did the first run have?** The model was built before the seed was set, so initial weights differed between runs and the same configuration gave different scores (26.20 and 27.02). Seeding before `build_model` fixed it, and a full rerun reproduced every number.

## Report sections written

- [`report/sections/recurrent_models.tex`](../../report/sections/recurrent_models.tex)
