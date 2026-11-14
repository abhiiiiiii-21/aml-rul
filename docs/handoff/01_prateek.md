# 01 Prateek Handoff

## What was built

- Hardened [`src/sequences.py`](../../src/sequences.py): training-only scaler, causal per-engine windows, left padding, ordered point selection, and explicit `cap_rul`.
- Added [`tests/test_sequences.py`](../../tests/test_sequences.py) covering shapes, targets, causality, padding, ordering, missing points, scaler scope, constants, and test data without RUL.
- Added literature and preprocessing report sections and the verified-source bibliography.
- Added runnable Phase 2 notebooks 01 and 02; notebook execution writes EDA figures and the literature table.

## Key numbers

The default window is 30 cycles and each sample has shape `(30, 14)`. The
pipeline produces one window per observed row, so padding does not change the
number of samples. Exact EDA quantities are computed and saved by notebook 01.

## Decisions and why

The scaler is a `(mean, std)` tuple to preserve the existing public API.
Windows are built separately per engine and sorted by cycle. Requested points
are selected by a left merge with `sort=False`, preserving caller order.
RUL capping is opt-in and never used by default because evaluation must use
the uncapped target.

## How to reproduce

From `aml-rul/`, install dependencies and run:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
jupyter nbconvert --to notebook --execute notebooks/phase2/01_dl_literature_and_eda.ipynb --output /tmp/phase2_01.ipynb
jupyter nbconvert --to notebook --execute notebooks/phase2/02_sequence_windows.ipynb --output /tmp/phase2_02.ipynb
```

## Known problems or open questions

The bibliography includes only sources with stable bibliographic metadata;
paper-specific RMSE, window, and regularization values are recorded as `n/a`
unless unambiguous in the source. Full model tuning and final test scoring
belong to later roles.

## Viva preparation

1. **Why use a window?** A single row cannot show a trend. A causal window exposes recent degradation while retaining cycle order.
2. **Why split by engine?** Rows from one engine are correlated. An engine split tests generalization to unseen degradation trajectories.
3. **Why fit the scaler on training engines only?** Validation and test statistics are information about held-out engines and would leak distribution information.
4. **What is look-ahead?** Including a row after the prediction cycle. The implementation builds each window from the current and earlier rows only.
5. **How is padding done?** The first observed sensor vector is repeated on the left until the requested length is reached.
6. **Why not drop short engines?** Dropping them changes the test population and wastes valid observations; padding preserves one sample per observed row.
7. **What does `points` do?** It selects exact engine-cycle keys after all windows are built and preserves the requested order.
8. **Why can `y` be `None`?** Test data intentionally has no RUL column, so windows can be made without inventing labels.
9. **Why protect zero standard deviation?** A constant sensor would otherwise create division by zero; its scale is set to one.
10. **Why is RUL capping optional?** Capping can stabilize learning on early-life uncertainty, but it discards high-RUL information and must not alter evaluation labels.
11. **Why are sequence models useful here?** They can learn temporal interactions instead of relying only on fixed rolling summaries.
12. **What is the main window-length risk?** Short windows miss slow degradation; long windows increase computation, padding, and overfitting.

## Report sections written

- [`report/sections/literature_review.tex`](../../report/sections/literature_review.tex)
- [`report/sections/dataset_preprocessing.tex`](../../report/sections/dataset_preprocessing.tex)
