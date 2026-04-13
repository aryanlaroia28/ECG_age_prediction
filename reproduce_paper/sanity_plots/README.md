**Sanity Plots — Interpretation Guide**

This file explains the sanity-check plots and CSVs produced by the repository scripts (`sanity_check.py` and `error_vs_age.py`). Use this guide to quickly interpret failures and likely causes.

- **Age Distribution (phase2_age_distribution.png):**
  - **What:** Histograms of `age` for Train / Val / Test splits.
  - **Why it matters:** Shows class imbalance across ages and whether validation/test reflect train distribution.
  - **Plausible explanations:** A skew towards younger ages indicates dataset imbalance; mismatched train/val/test shapes suggests sampling issues.
  - **Checks:** Look for empty bins, extreme outliers, or very different means across splits.

- **Sample ECG Waveforms (phase2_sample_ecg.png):**
  - **What:** 12-lead ECG traces for a single record (first sample).
  - **Why it matters:** Verifies channel ordering, orientation (lead × time), and obvious corruption.
  - **Plausible explanations:** Flat-lines → missing data or indexing bug; very noisy / huge amplitude → unit mismatch (µV vs mV) or missing normalization.

- **ECG Amplitude Histogram (phase2_ecg_amplitude_hist.png):**
  - **What:** Distribution of per-sample amplitudes (first 100 records).
  - **Why it matters:** Confirms global signal scale and presence of NaNs or outliers.
  - **Plausible explanations:** Very small variance → overly normalized data; extremely large tails → units or artifact contamination.

- **Loss Curves (phase3_loss_curves.png):**
  - **What:** Training loss and validation L1 per epoch for grouped methods/backbones.
  - **Why it matters:** Detects convergence, underfitting, and overfitting.
  - **Plausible explanations:** Train >> Val (no learning) → bug in training loop; Val >> Train (overfit) → reduce capacity or use regularization; noisy curves → unstable LR or batch-size issues.

- **Validation MSE Curves (phase3_val_mse_curves.png):**
  - **What:** Validation MSE across epochs for runs grouped by method/backbone.
  - **Why it matters:** Confirms relative performance and whether early stopping / checkpointing selected the proper model.
  - **Checks:** Look for large final variance between runs of the same setting (reproducibility issue).

- **Per-run Test Summaries / Log CSVs:**
  - **What:** `analyze_logs.py` produces `log_analysis/summary.csv` summarizing best-epoch val metrics and test metrics parsed from training logs.
  - **Why it matters:** Quick table to compare runs; use it to cross-check MAE/MSE reported in paper vs reproduced.

- **Error vs Age (error_vs_age_*.png & error_vs_age_*.csv):**
  - **What:** MAE per 5-year age bin (0–5, 5–10, …, 95–100). CSV contains `age_bin`, `count`, and `mae` columns.
  - **Why it matters:** Key fairness / reliability diagnostic — shows whether errors concentrate at certain ages.
  - **Plausible explanations:** High MAE at older ages often caused by low sample counts (see `count` column) or distributional shift; bimodal peaks could indicate labeling errors or dataset mix-up.
  - **Checks:** Compare per-bin `count` to MAE; low-count bins are unreliable and should be interpreted cautiously.

- **Train/Test Leakage & NaN Checks (printed by `sanity_check.py`):**
  - **What:** Console outputs show counts of NaNs and overlaps between splits.
  - **Why it matters:** Leakage invalidates test metrics; NaNs break training and must be cleaned.
  - **Fixes:** Remove duplicate indices across splits, drop/fill NaNs, or regenerate splits.

Recommendations
- Always inspect `error_vs_age_*.csv` before trusting per-bin MAE — look at `count` to ensure statistical reliability.
- If ECG amplitudes or waveforms look suspicious, check data loading (`data/*.npy`) and `datasets.PTBXLDataset` indexing/transpose.
- For unstable training curves, try lowering the LR, reducing batch size, or enabling `cudnn.benchmark=False` to improve determinism.

If you want, I can:
- Aggregate all per-run `error_vs_age_*.csv` into a single summary table and a comparative plot.
- Add these interpretation notes into the main repo README.

File locations
- Plots and CSVs: [sanity_plots](sanity_plots/)
- Main scripts: [reproduce_paper/train.py](train.py), [reproduce_paper/sanity_check.py](sanity_check.py), [reproduce_paper/error_vs_age.py](error_vs_age.py)
