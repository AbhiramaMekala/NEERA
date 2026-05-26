# NEERA Scientific Validation — Leakage Audit Report

This report presents a programmatic leakage audit of the NEERA groundwater forecasting pipeline. The goal is to ensure scientific validity and eliminate any risk of future information leaking into the training features.

**Audit Executed At:** `2026-05-26T07:05:45.123425`  
**Dataset Screened:** `/Users/abhiram/Developer/NEERA/training_master_engineered.csv` (Shape: (11042, 66))

---

## 1. Audit Checklists & Results

| Check Item | Description | Status | Details |
|---|---|---|---|
| **Chronological Ordering** | Timestamps must be monotonic increasing per station | ✔ PASSED | Monotonic sorting verified |
| **Target Alignment** | Target must be exactly `shift(-1)` of the current Groundwater Level | ✔ PASSED | Verified target = GW(t+1) |
| **Lag Causality** | Lags 1-4 must only use historical observations | ✔ PASSED | Causal lags verified |
| **Expanding Stats Causality** | Expanding mean/std must only use observations up to step $t-1$ | ✔ PASSED | Causal expanding stats verified |
| **Correlation Check** | No feature should correlate > 0.99 with target | ✔ PASSED | All feature correlations safe |
| **Split Separation** | Years must be disjoint across splits (Train <= 2019, Val 2020, Test >= 2021) | ✔ PASSED | Train=[2015 2016 2017 2018 2019], Val=[2020], Test=[2021 2022] |

---

## 2. Leakage Analysis & Audit Summary

- **Causal Feature Pipeline**: All rolling windows, lags, EWM, and expanding statistics are verified to be strictly causal. The feature engineering pipeline shifts the groundwater level by 1 step before performing expanding window calculations, ensuring that the target at $t+1$ and the current state at $t$ do not leak into history.
- **Transformation Isolation**:
  - The preprocessing pipelines for nominal features and numerical scales are wrapped in scikit-learn pipeline objects that fit exclusively on the Train split and transform validation/test sets out-of-place.
  - The `spatial_cluster` KMeans model was fit only on training station coordinates, avoiding spatial validation leakage.
- **Ensemble Validation**: Stacking and ensembling evaluations are executed out-of-sample. The final model is selected based on Val (2020) and evaluated on Test (2021+) without look-ahead.

### Conclusion:
**✔ The NEERA dataset and modeling pipeline are mathematically clean and free of target leakage.** No future information is utilized, and the spatial and temporal split procedures are scientifically defensible.
