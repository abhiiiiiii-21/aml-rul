# Phase 2 Individual Viva Voce Comprehensive Preparation Pack

This document is the master study guide for the individual viva voce examinations. Every team member must understand their own section in depth and be familiar with the cross-cutting principles.

---

## Master Key Numbers Table (Quick Reference)

### Validation Set (100 Fixed Points, Cycles, Lower is Better)
| Model | Family | Parameters | MAE | RMSE | Early MAE (>125) | Late MAE (≤125) |
| --- | --- | --- | --- | --- | --- | --- |
| OLS (Phase 1) | Linear | 58 | 32.86 | 42.92 | ~120.0 | ~20.0 |
| Elastic Net (Phase 1) | Regularized Linear | 58 | 31.66 | 41.77 | ~115.0 | ~18.5 |
| Random Forest (Phase 1) | Tabular Ensemble | ~500k | 25.03 | 40.17 | 49.33 | 18.17 |
| Gradient Boosting (Phase 1) | Tabular Ensemble | ~100k | 26.12 | 40.17 | 50.12 | 19.34 |
| **LSTM** (`lstm_v1`, P2) | Recurrent | 20,545 | **19.41** (20.14±1.46) | **34.45** (33.18±2.46) | 43.73 | **12.55** |
| GRU (`gru_v1`, P2) | Recurrent | 40,385 | 23.18 (22.98±0.49) | 36.34 (36.61±0.90) | 43.34 | 17.49 |
| 1D-CNN (`cnn1d_v1`, P2) | Convolutional | ~35,000 | 24.32 (24.38±0.14) | 36.70 (36.86±0.17) | 41.33 | 19.53 |
| Transformer (`transformer_v1`, P2) | Self-Attention | ~25,000 | 20.59 (22.80±2.28) | 34.79 (36.97±2.60) | **41.10** | 14.80 |
| **Hybrid Stacking** (`hybrid_v1`, P2) | Stacking Ensemble | Base DL + RF | 21.48 (21.72±0.31) | **34.38** (35.34±0.91) | 41.52 | 15.83 |
| Hybrid Average (`hybrid_avg_v1`, P2) | Simple Blend | None | 21.50 (21.50±0.03) | 35.83 (35.91±0.08) | 46.18 | 14.55 |

---

## Cross-Cutting Core Principles (All Members Must Know)

### 1. Why split by engine and never by row?
- **Answer:** In CMAPSS, consecutive rows of the same engine share 29 of 30 cycles (97% overlap) and follow the same unobserved degradation trajectory. A row-wise random split would place nearly identical sliding windows in both training and validation sets, causing catastrophic data leakage and artificially optimistic error metrics. The effective sample size is 80 independent engine degradation histories, not 16,390 rows.

### 2. What is temporal look-ahead and how was it strictly prevented?
- **Answer:** Look-ahead occurs when information from cycle $t' > t$ is used to predict the RUL at cycle $t$. It was prevented by constructing sliding windows $X_t$ strictly using observations from cycle $\le t$ of that engine. Any feature standardizer was fitted solely on training engines.

### 3. Why evaluate on uncapped true RUL?
- **Answer:** Piecewise linear capping (e.g. at 125 cycles) assumes that initial healthy operation exhibits zero measurable degradation. While useful as a regularized training target, evaluating against a capped RUL would artificially lower MAE by ignoring errors when true RUL is high. Real-world maintenance planning requires unbiased estimation across the engine's entire lifespan.

### 4. What is the asymmetric NASA Scoring Function?
- **Answer:** Proposed by Saxena et al. (2008):
  $$S = \sum_{i=1}^{N} s_i, \quad s_i = \begin{cases} e^{-d_i / 13} - 1 & \text{if } d_i < 0 \text{ (early prediction)} \\ e^{d_i / 10} - 1 & \text{if } d_i \ge 0 \text{ (late prediction)} \end{cases}$$
  where $d_i = \hat{y}_i - y_i$. It penalizes late predictions (overestimating RUL, risking catastrophic in-flight failure) exponentially more severely than early predictions (calling for premature inspection).

---

## Role 1: Prateek Shakya (Foundation, EDA, Windowing Pipeline)

1. **Why use a sliding window instead of single-cycle rows?** A single sensor row lacks trend information. A temporal window of length 30 captures rate-of-change, acceleration, and cumulative drift across 14 sensors simultaneously.
2. **How does left-padding work for engines shorter than 30 cycles?** The first observed row of that engine is repeated on the left ($t=1$ state repeated) until length 30 is reached. This preserves one prediction per observed cycle without introducing zero-padding artifacts.
3. **Why fit the scaler on training engines only?** Computing mean and variance across the full dataset leaks information about the test engine distributions into the training representation.
4. **Why are 7 sensors dropped?** Sensors 1, 5, 6, 10, 16, 18, 19 have zero or near-zero variance across all engines and operating cycles; keeping them introduces numerical instability in standard scaling and adds uninformative dimensions.
5. **Why protect zero standard deviation in scaling?** If a sensor has zero variance, dividing by $\sigma = 0$ causes `NaN`. We set $\sigma = 1.0$ when $\sigma = 0$.
6. **What does `validation_points(val_df)` do?** It extracts the exact 100 engine-cycle cutoffs (5 random points per engine across 20 validation engines) fixed in Phase 1, ensuring identical evaluation conditions across all models.
7. **Why can `y` be `None` in `make_windows`?** The raw test set (`test_FD001.txt`) contains no RUL column. The pipeline supports label-free inference cleanly.
8. **What is the risk of a window that is too short or too long?** Short windows ($<20$) miss slow degradation trends; long windows ($>40$) require heavy padding for short trajectories and introduce uninformative early healthy cycles.
9. **How many total samples are generated from the 80 training engines?** 16,390 causal windows of shape $(30, 14)$.
10. **Why did we not drop the first 29 cycles of each engine?** Dropping them would discard early-life observations and distort the test set where some engines are stopped before cycle 30.
11. **What are the primary degradation indicators in FD001?** Sensor 2 (HPC inlet temp), Sensor 3 (HPC outlet temp), Sensor 4 (LPT outlet temp), and Sensor 11 (Static pressure at HPC outlet) exhibit consistent monotonic upward drift.
12. **Why is RUL defined as $T - t$?** Linear run-to-failure assumes each operating cycle consumes one unit of operational life until the engine exceeds safe vibration or temperature thresholds.

---

## Role 2: Yogesh Modi (Recurrent Models: LSTM & GRU)

1. **Why does LSTM prevent vanishing gradients?** The cell state update $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ is purely additive. Its gradient backpropagating through time contains the forget gate $f_t \in [0, 1]$ directly rather than repeatedly multiplying by weight matrices $W$.
2. **Why do plain RNNs suffer from vanishing/exploding gradients?** Backpropagation Through Time (BPTT) requires multiplying the Jacobian $\frac{\partial h_t}{\partial h_{t-1}}$ across all 30 time steps. If eigenvalues are $<1$, gradients vanish exponentially; if $>1$, they explode.
3. **What is the structural difference between LSTM and GRU?** LSTM has 3 gates (input, forget, output) and distinct cell and hidden states (4 weight matrices). GRU merges cell and hidden state and uses 2 gates (reset, update), reducing parameter count by ~25%.
4. **Why did LSTM outperform GRU here?** The LSTM achieved validation MAE 19.41 vs GRU 23.18. The dedicated cell state maintains long-term baseline memory across 30 cycles better than the merged GRU hidden state.
5. **How does AdamW decoupled weight decay work?** In standard Adam with L2 regularization, the weight penalty is added to the gradient, which then gets scaled by the running second-moment estimate $1/\sqrt{v_t}$. In AdamW, weight decay $\lambda \theta_t$ is subtracted directly from the parameter update, making regularization rate-independent.
6. **Why did changing weight decay between $10^{-4}$ and $10^{-2}$ have minimal effect?** At learning rate $10^{-3}$, weight shrink per step is $10^{-3} \times 10^{-4} = 10^{-7}$. Over ~3,000 steps, gradient updates overwhelmingly dominate weight decay.
7. **What happened when early stopping was turned off?** Training loss continued to decrease (training MAE fell to 7 cycles), but validation MAE degraded from 26.8 to 31.2 cycles due to trajectory memorization of the 80 training engines.
8. **Why did gradient clipping matter?** Mean gradient norms peaked at 1.57 (GRU) and 1.31 (LSTM). Clipping at 1.0 prevented disruptive parameter updates during early epochs when loss was steep.
9. **Why report 3-seed mean and std ($20.14 \pm 1.46$)?** A single seed run (19.41) can be lucky or unlucky. Multi-seed evaluation proves the architecture's stability and quantifies training stochasticity.
10. **Why was selection performed on all validation windows rather than the 100 points?** The 100 fixed points represent a sparse, noisy subsample (5 per engine). Evaluating on all validation windows provided a lower-variance signal for model selection without contaminating the reported benchmark points.

---

## Role 3: Abhishek Kumar Patel (1D-CNN & Transformer)

1. **What is the inductive bias of a 1D-CNN for RUL?** Temporal convolution assumes translation equivariance and local stationarity: sensor patterns spanning 5 consecutive cycles (e.g. sudden pressure spikes or rapid temperature rises) carry diagnostic value regardless of when they occur in the window.
2. **Why did 1D-CNN lag behind LSTM (24.32 vs 19.41 MAE)?** A 2-layer CNN with kernel size 5 has a receptive field of $(5 - 1) + (5 - 1) + 1 = 9$ cycles (expanded slightly by pooling). It lacks the global temporal integration needed to track cumulative wear across the full 30-cycle window.
3. **Explain the Scaled Dot-Product Attention mechanism:**
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   It computes dynamic pairwise compatibility between time steps, allowing cycle 30 to query cycle 1 directly without routing through intermediate states.
4. **Why are positional encodings required in the Transformer?** Self-attention is permutation-invariant. Without positional embeddings, the network cannot distinguish whether a high temperature occurred at cycle 1 or cycle 30.
5. **Why did the Transformer perform so strongly (MAE 20.59)?** Direct $O(1)$ path length between any two time steps allows the Transformer to detect subtle long-range sensor drifts across the entire 30-cycle sequence.
6. **What do the learned attention maps show?** In healthy engine stages, attention is uniformly dispersed. Near failure, attention heads concentrate sharply on the final 5-8 cycles, capturing the exponential phase of bearing and seal wear.
7. **What is the parameter comparison between Transformer and CNN?** Transformer used 2 layers, 4 heads, $d_{model}=32$ (~25,000 params). CNN used 2 layers of 64 channels (~35,000 params). Transformer was both more compact and more accurate.
8. **Why did we use mean pooling over the sequence dimension in Transformer?** Mean pooling aggregates information across the entire sequence representation rather than relying solely on a single token (like `[CLS]`), which stabilized gradient flow.

---

## Role 4: Kapish Rohilla (Hybrid ML + DL Stacking Model)

1. **What is the architecture of the Hybrid Stacking model?** A two-stage ensemble:
   - Base stage: Out-of-fold predictions from LSTM and GRU.
   - Meta stage: Random Forest regressor trained on 58 features (base DL predictions + 57 Phase 1 rolling statistical features).
2. **How was target leakage strictly prevented in stacking?** Via 5-fold cross-validation partitioned strictly by `engine_id` (16 engines per fold). When generating training features for the meta-learner, base models never made predictions on engines they were trained on.
3. **What is the component ablation result?**
   - DL predictions only: MAE 24.93, RMSE 38.10
   - Rolling features only: MAE 25.03, RMSE 40.17
   - Combined Stacking (`hybrid_v1`): **MAE 23.08**, **RMSE 35.40**
   Proves super-additive synergy between sequential representations and engineered rolling features.
4. **Why does Hybrid Stacking achieve the best RMSE (34.38)?** Random Forest meta-learner penalizes variance and extreme outliers, pulling erroneous deep learning predictions back toward the empirical mean.
5. **How does simple blending (`hybrid_avg_v1`) compare to stacking (`hybrid_v1`)?**
   - Blending: simple average of LSTM and GRU predictions (MAE 21.50, RMSE 35.83).
   - Stacking: MAE 21.48, RMSE 34.38. Stacking improves RMSE by 1.45 cycles because it dynamically reweights based on rolling feature context.
6. **Why did we use Random Forest as the meta-learner instead of linear regression?** Random Forest can model nonlinear regime shifts (e.g. trusting deep learning in late life while relying on conservative rolling statistics in early life).

---

## Role 5: Vikash Kumar (Ablations, Final Evaluation, Integration)

1. **What were the key takeaways from the 7 ablation studies?**
   - **Window length:** Window 30 is optimal (MAE 19.39 vs 25.40 at W=10).
   - **Input features:** Raw 14 sensors are best; adding settings or differences added noise.
   - **RUL capping:** Capping at 125 cycles accelerates early convergence but alters evaluation calibration.
   - **Capacity:** Single layer 64 hidden units is optimal; deeper networks overfit the 80 engines.
   - **Regularization:** Moderate dropout (0.2) + AdamW is critical.
   - **Architecture:** LSTM > Transformer > GRU > CNN.
   - **Hybrid:** Combining tabular + DL reduces variance and cuts RMSE.
2. **What was the result of the paired bootstrap hypothesis test?** Over 2,000 bootstrap resamples on the test set, the 95% confidence interval for $(\text{MAE}_{\text{Phase 1}} - \text{MAE}_{\text{Phase 2}})$ was strictly positive and excluded zero, proving statistically significant superiority of Phase 2 deep sequence models ($p < 0.01$).
3. **Why did all models struggle on engines with true RUL > 200?** In FD001, degradation is an accelerated exponential process that only manifests physically in the final 80-100 cycles. Prior to cycle 100, sensors exhibit baseline noise with near-zero degradation gradient. Predicting whether an engine will last 220 or 340 cycles from non-degraded initial readings is physically ill-posed.
4. **How does deep learning change the NASA asymmetric score?** Because Phase 1 models severely underpredicted high RUL, they occasionally avoided late penalties at the cost of massive early penalties. Deep sequence models provide calibrated predictions across all bands, reducing both symmetric MAE and catastrophic late overpredictions near failure.
5. **How was full project reproducibility guaranteed?**
   - Global seeding (`seed=42`) across PyTorch, NumPy, and Python random.
   - Deterministic operations enabled on CPU and MPS.
   - All intermediate prediction CSVs, model weights, and configs committed to `results/phase2/`.
   - Notebooks execute linearly top-to-bottom without cached state dependencies.
