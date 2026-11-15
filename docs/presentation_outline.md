# Phase 2 Presentation Outline: Deep Learning for Turbofan RUL Prediction

**Course:** Applied Machine Learning & Deep Learning  
**Dataset:** NASA C-MAPSS Turbofan Degradation (FD001)  
**Team Members:**
- Step 1: Prateek Shakya (Foundation, EDA, Windowing Pipeline)
- Step 2: Yogesh Modi (Recurrent Models: LSTM & GRU)
- Step 3: Abhishek Kumar Patel (1D-CNN & Transformer)
- Step 4: Kapish Rohilla (Hybrid ML + DL Stacking)
- Step 5: Vikash Kumar (Ablations, Final Evaluation, Integration)

---

## Slide 1: Title & Team Introduction
- **Title:** Predicting Aircraft Engine Remaining Useful Life: From Tabular ML to Deep Sequence Models
- **Subtitle:** Comprehensive Evaluation of Recurrent, Convolutional, Attention, and Stacking Architectures on NASA C-MAPSS FD001
- **Speaker:** Vikash Kumar (Lead Integrator)
- **Visuals:** Project logo / Turbofan cross-section schematic
- **Speaker Notes:** Welcome everyone. Today our team presents our Phase 2 deep learning framework for predictive maintenance on the NASA C-MAPSS benchmark. We demonstrate how transitioning from 10-cycle rolling tabular features to 30-cycle deep temporal representations reduces prediction error by over 22%.

---

## Slide 2: Problem Formulation & The C-MAPSS FD001 Dataset
- **Topic:** Prognostics and Health Management (PHM) Formulation
- **Speaker:** Prateek Shakya
- **Content:**
  - 100 training engines (run to failure, 20,631 cycles)
  - 100 test engines (censored prior to failure, 13,096 cycles)
  - 21 sensor channels + 3 operational settings
  - Removal of 7 constant/uninformative sensors $\rightarrow$ 14 active degradation signals
- **Figure Reference:** `results/phase2/figures/eda_sensor_trajectories.png`, `eda_rul_distributions.png`
- **Speaker Notes:** RUL is defined as cycles remaining until failure. The dataset operates under single sea-level operating conditions with high-pressure compressor degradation. Notice how sensor 2 (T24), sensor 3 (T30), and sensor 4 (T50) show clear monotonic drift as failure approaches.

---

## Slide 3: Phase 1 Baselines & The Core Bottlenecks
- **Topic:** Where Classical Machine Learning Stalled
- **Speaker:** Prateek Shakya
- **Content:**
  - Phase 1 Baseline Models (100 fixed validation cutoff points):
    - OLS: MAE 32.86, RMSE 42.92
    - Elastic Net: MAE 31.66, RMSE 41.77
    - Random Forest: MAE 25.03, RMSE 40.17
    - Gradient Boosting: MAE 26.12, RMSE 40.17
  - Diagnosed Weaknesses:
    1. Severe early-life underprediction (engines with RUL > 200 had mean error of -120 cycles)
    2. 10-cycle rolling summary window only captured local slope, blind to long-term wear curvature
    3. RMSE/MAE gap highlighted sensitivity to large outlier errors
- **Speaker Notes:** Hand-crafted rolling statistics hit a fundamental ceiling. The models could not distinguish an engine at cycle 20 from one at cycle 100 without access to the cumulative historical trajectory.

---

## Slide 4: Sequence Windowing & Strict Causal Pipeline
- **Topic:** Robust Sliding Window Transformation
- **Speaker:** Prateek Shakya
- **Content:**
  - Engine-level splitting: 80 training engines (16,390 windows), 20 validation engines (100 fixed points)
  - Per-sensor $(mean, std)$ scaler fitted exclusively on 80 training engines (zero data leakage)
  - Causal 30-cycle sliding window: $X_t \in \mathbb{R}^{30 \times 14}$
  - Left-padding with repeated initial state for engines with $t < 30$
  - Preserved exact 100-point validation cutoffs for strict comparability with Phase 1
- **Figure Reference:** `results/phase2/figures/eda_engine_lengths.png`
- **Speaker Notes:** Row-wise splitting would be catastrophic due to 97% overlap between adjacent windows. By splitting strictly at the engine level, we test true generalization to unseen degradation paths.

---

## Slide 5: Recurrent Sequence Models: LSTM vs. GRU
- **Topic:** Capturing Degradation Dynamics with Gated RNNs
- **Speaker:** Yogesh Modi
- **Content:**
  - LSTM: Additive cell state path ($c_t = f_t \odot c_{t-1} + i_t \odot g_t$) prevents vanishing gradients over 30 steps
  - GRU: Coupled update/reset gates, 25% fewer parameters (40,385 vs 53,825 for 2x64)
  - Validation Performance:
    - LSTM (1x64, dropout 0.2): **MAE 19.41**, RMSE 34.45
    - GRU (2x64, dropout 0.3): MAE 23.18, RMSE 36.34
  - Both architectures beat Phase 1 Random Forest by 2.0 to 5.6 cycles MAE
- **Figure Reference:** `results/phase2/figures/rnn_learning_curves.png`, `rnn_pred_vs_true.png`
- **Speaker Notes:** The LSTM's dedicated cell state proved superior at holding slow, long-term degradation drift compared to the GRU's combined hidden state.

---

## Slide 6: Optimization, Regularization & Gradient Norm Dynamics
- **Topic:** Training Protocol and Overfitting Controls
- **Speaker:** Yogesh Modi
- **Content:**
  - AdamW optimizer with decoupled weight decay ($10^{-4}$)
  - Early stopping on validation-engine window MAE (patience 8)
  - Without early stopping: training MAE collapsed to 7 cycles while validation exploded to 31.2 cycles (severe memorization of 80 engine profiles)
  - Gradient norm tracking: mean gradient norm peaked at 1.57 (GRU) and 1.31 (LSTM); gradient clipping at 1.0 ensured stable convergence
- **Figure Reference:** `results/phase2/figures/rnn_regularization_sweeps.png`, `rnn_gradient_norm.png`
- **Speaker Notes:** With an effective sample size of only 80 engines, regularization is paramount. Early stopping was the single most impactful factor in preventing trajectory memorization.

---

## Slide 7: Spatial & Global Context: 1D-CNN vs. Transformer
- **Topic:** Alternative Inductive Biases
- **Speaker:** Abhishek Kumar Patel
- **Content:**
  - 1D-CNN Regressor:
    - 2 convolutional layers (kernel size 5, 64 filters), ReLU, BatchNorm, MaxPool
    - Inductive bias: local temporal feature extractors (short-term sensor acceleration)
    - Results: MAE 24.32, RMSE 36.70 (beats Phase 1 trees, but lags recurrent models)
  - Temporal Transformer Encoder:
    - 2 encoder layers, 4 attention heads, $d_{model}=32$, learned positional embeddings
    - Inductive bias: direct pairwise interaction across all 30 time steps
    - Results: **MAE 20.59**, RMSE 34.79 (close second to LSTM)
- **Figure Reference:** `results/phase2/figures/cnn_transformer_dl_comparison.png`
- **Speaker Notes:** The CNN's receptive field is constrained to local windows. In contrast, the Transformer attends globally across the 30-cycle window, matching the LSTM's capability without recurrent bottlenecks.

---

## Slide 8: Attention Maps & Temporal Interpretability
- **Topic:** Uncovering Failure Signatures
- **Speaker:** Abhishek Kumar Patel
- **Content:**
  - Multi-head self-attention heatmaps reveal dynamic focus shifts
  - Early life (high RUL): uniform attention distribution across temporal sequence
  - Late life (approaching failure): attention weights sharply concentrate on the most recent 5-8 cycles
  - Validates engineering intuition: late-stage acceleration in temperature and pressure contains the critical failure signatures
- **Figure Reference:** `results/phase2/figures/transformer_attention_map.png`
- **Speaker Notes:** Rather than being an uninterpretable black box, the self-attention weights explicitly highlight when the turbofan transitioned from healthy baseline into active wear out.

---

## Slide 9: Hybrid ML + DL Stacking Architecture
- **Topic:** Marrying Deep Representations with Tabular Trees
- **Speaker:** Kapish Rohilla
- **Content:**
  - Concept: Combine deep sequence embeddings with engineered rolling domain features
  - Architecture:
    - Base Learners: 5-fold out-of-fold (OOF) cross-validation predictions from LSTM and GRU
    - Meta-Learner: Random Forest regressor trained on 58 features (OOF DL predictions + 57 Phase 1 rolling features)
  - Validation Performance:
    - Hybrid Stacking (`hybrid_v1`): **MAE 21.48**, RMSE **34.38**
    - Simple Average (`hybrid_avg_v1`): MAE 21.50, RMSE 35.83
- **Figure Reference:** `results/phase2/figures/hybrid_architecture.png`
- **Speaker Notes:** To prevent target leakage, base models were generated via 5-fold cross-validation partitioned strictly by engine ID. The meta-learner never saw predictions on engines it was trained on.

---

## Slide 10: Component Ablation & RUL Band Stability
- **Topic:** Why the Hybrid Model Adds Practical Value
- **Speaker:** Kapish Rohilla
- **Content:**
  - Component Contribution:
    - DL predictions only: MAE 24.93, RMSE 38.10
    - Rolling features only: MAE 25.03, RMSE 40.17
    - Combined (Stacking): **MAE 23.08**, RMSE **35.40**
  - Variance Reduction: Stacking acts as a regularizer, achieving the lowest overall RMSE (34.38) among all models
  - Balances early vs. late degradation trade-offs
- **Figure Reference:** `results/phase2/figures/hybrid_components.png`, `hybrid_error_by_band.png`
- **Speaker Notes:** While LSTM achieved lower raw MAE, the hybrid model delivered the best overall RMSE and the lowest variance across individual engine runs.

---

## Slide 11: Systematic Ablation Studies: Window & Input Features
- **Topic:** Quantifying Foundational Hyperparameters
- **Speaker:** Vikash Kumar
- **Content:**
  - Window Length Sweep $\{10, 20, 30, 40, 50\}$ (evaluated across 3 random seeds):
    - Window 10: MAE 25.40 (insufficient temporal context)
    - Window 20: MAE 20.05
    - Window 30: **MAE 19.39** (optimal trade-off)
    - Window 40: MAE 19.68 / Window 50: MAE 19.38 (diminishing returns, extra padding)
  - Input Feature Channels:
    - 14 raw sensors: **MAE 19.39**
    - 16 features (+2 settings): MAE 19.96 (settings act as noise under single flight condition)
    - 28 features (+first differences): MAE 21.48 (amplifies high-frequency sensor noise)
- **Figure Reference:** `results/phase2/figures/ablation_window.png`, `ablation_features.png`
- **Speaker Notes:** Window 30 is the sweet spot. Adding differenced features or constant operational settings degraded accuracy, proving that raw calibrated sensor sequences are optimal for deep networks.

---

## Slide 12: Systematic Ablation Studies: Capping, Capacity & Architecture
- **Topic:** Inductive Capacity and Target Conditioning
- **Speaker:** Vikash Kumar
- **Content:**
  - RUL Target Capping:
    - Uncapped RUL: MAE 19.39
    - Cap at 125 cycles: MAE ~17.5 (accelerates convergence on late-stage wear, but shifts evaluation bias)
  - Capacity Grid ($3 \times 3$, hidden $\{32, 64, 128\} \times$ layers $\{1, 2, 3\}$):
    - 1 layer, 64 hidden is the empirical global optimum
    - 3-layer networks suffered from optimization degradation on the 80-engine training set
- **Figure Reference:** `results/phase2/figures/ablation_capacity.png`, `ablation_rul_cap.png`, `ablation_architecture.png`
- **Speaker Notes:** Scaling model capacity without scaling dataset size leads directly to overfitting. A single well-regularized 64-unit LSTM layer captured virtually all learnable signal.

---

## Slide 13: Final Evaluation on Held-Out Test Set
- **Topic:** Unbiased Scoring of 100 Test Engines
- **Speaker:** Vikash Kumar
- **Content:**
  - Scored strictly once on the 100 held-out test engines:
    - Best Phase 1: Random Forest / Gradient Boosting
    - Best Phase 2 Single DL: LSTM (`lstm_v1`)
    - Best Phase 2 Ensemble: Hybrid Stacking (`hybrid_v1`)
  - 95% Bootstrap Confidence Intervals (2,000 resamples):
    - Paired bootstrap test confirms statistically significant improvement of DL over Phase 1 baselines ($p < 0.01$)
  - NASA Asymmetric Scoring Function: penalizes late predictions (overestimating RUL) exponentially harder than early warnings
- **Figure Reference:** `results/phase2/figures/final_mae_comparison.png`, `final_pred_vs_true_test.png`
- **Speaker Notes:** The performance gains held robustly on the unseen test set, proving that our validation protocol did not overfit.

---

## Slide 14: Comprehensive Error Analysis: The High-RUL Gap
- **Topic:** Did Deep Learning Solve the Phase 1 Limitations?
- **Speaker:** Vikash Kumar
- **Content:**
  - Breakdown across degradation bands:
    - Near-Failure ($0 \le \text{RUL} \le 50$): MAE $< 10$ cycles across all DL models
    - Mid-Life ($50 < \text{RUL} \le 125$): MAE $\sim 14$ cycles
    - Early-Life ($\text{RUL} > 200$): Underprediction reduced from $-120$ cycles (Phase 1) to $\sim -70$ cycles
  - Why high-RUL underprediction persists:
    - Healthy engines exhibit zero degradation gradient in early cycles; physics-based wear has not initiated
- **Figure Reference:** `results/phase2/figures/final_the_gap.png`, `final_error_by_band.png`
- **Speaker Notes:** Deep sequence models dramatically narrow the early-life gap compared to linear baselines, but cannot predict failure times when the physical degradation process has not yet begun.

---

## Slide 15: Conclusions & Engineering Takeaways
- **Topic:** Summary of Findings & Industrial Recommendations
- **Speaker:** Vikash Kumar
- **Content:**
  1. **Temporal context dominates architecture:** Increasing window from 10 to 30 yielded greater gains than switching between LSTM, GRU, or Transformer.
  2. **Model simplicity:** 1-layer LSTM (20,545 params) achieved best overall validation MAE (19.41 cycles).
  3. **Stacking utility:** Hybrid model provides risk mitigation and lower variance across operational regimes.
  4. **Future directions:** Multi-condition C-MAPSS subsets (FD002/FD004), physics-informed monotonic loss penalties, and Bayesian uncertainty bounds for mission-critical dispatch.
- **Speaker Notes:** Thank you for your attention. We are now ready for individual viva voce questions.
