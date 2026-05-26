# NEERA Groundwater Forecasting Project — Final Research Summary

## Executive Summary

The NEERA project is a groundwater level forecasting system designed to predict next-season groundwater depths (MBGL - Meters Below Ground Level) across Karnataka, India. By integrating seasonal groundwater observations, spatial telemetry networks, and historical rainfall signals, NEERA models the complex spatial-temporal dynamics of aquifer recharge.

This summary describes the end-to-end engineering methodology, data validation, baseline modeling, feature importances, and evaluation results.

---

## 1. Methodology & Pipeline Architecture

The NEERA pipeline is structured as a failure-driven, reproducible workflow comprising data engineering, spatiotemporal mapping, and model training:

```
[Master Dataset] ──────────► [Geographic & Quality Filters] ──► [Continuous Reindexing] 
                                                                          │
[Telemetry/Manual Rain] ──► [Anomaly Thresholding (500mm)] ───────────────┘
                                       │
                                       ▼
  [Hierarchical Routing] ◄──────── [Dynamic Overlap Mapping]
 (Local -> District -> State)
         │
         ▼
[Feature Engineering] ──► [Strict Temporal Split] ──► [LightGBM, XGBoost, CatBoost]
```

### Preprocessing & Spatial Filters
- **Geographic Filtering**: Observations are filtered to the Karnataka coordinate bounds (Latitude 11.0°–19.0°, Longitude 73.0°–79.0°). Coordinates are validated and reversed values (lat/lon swapped) are corrected.
- **Quality Filters**: Stations with fewer than 8 observations are discarded to guarantee sufficient temporal sequence length.

---

## 2. Spatiotemporal Rainfall Routing System

Integrating rainfall data with groundwater observations presented a severe geographic and temporal alignment challenge. NEERA implements a hierarchical fallback routing model to guarantee 100% feature coverage:

1. **Daily Rainfall Sparsity Fix**: Rainfall telemetry is sparse, only logging non-zero precipitation days. The pipeline reindexes each station to a continuous daily range (filling dry days with `0.0` rain) to prevent calendar alignment failures.
2. **Dynamic Date-wise Mapping**: Rather than mapping each groundwater station to a single static rain station overall, mapping is calculated dynamically per observation date. Groundwater observations are matched to the nearest *active* rainfall station on that day.
3. **Hierarchical Routing Cascade**:
   - **Local Telemetry**: Joins precomputed rolling windows from the nearest active station within `50 km`.
   - **District Fallback**: If no local station is active within `50 km`, joins the median rainfall of the active stations in the same district.
   - **State Fallback**: If the district has no active stations, uses the statewide average rainfall.

---

## 3. Feature Engineering

The models are trained on three categories of features:
- **Groundwater Lags**: `prev_gw` (value from the previous season), `gw_diff` (temporal delta), and rolling metrics (`gw_roll_mean_7obs`, `gw_roll_std_7obs`, `gw_roll_mean_30obs`).
- **Rainfall Features**: Cumulative 30-day, 90-day, and 180-day effective rainfall averages (`effective_rainfall_*`), and window completeness metadata.
- **Seasonality & Location**: Season type (`season`), sinusoidal seasonal encodings (`season_sin`, `season_cos`), and station location details.
- **Target**: `target_next_season_gw` (groundwater level at the subsequent season step, causal-shifted).

---

## 4. Modeling & Temporal Validation Split

To prevent look-ahead bias and guarantee temporal generalization, we utilize a strict temporal split:
- **Train Split**: All observations up to and including **2019** (7,530 rows)
- **Validation Split**: Observations in **2020** (1,327 rows)
- **Test Split**: All observations from **2021** onwards (2,185 rows)

We trained three tree-based baseline regression models:
1. **LightGBMRegressor**
2. **XGBoostRegressor**
3. **CatBoostRegressor**

### Performance Metrics

| Model | Split | MAE (MBGL) | RMSE (MBGL) | $R^2$ | MAPE (%) |
|---|---|---|---|---|---|
| **LightGBM** | Train | 1.4259 | 2.4603 | 0.9893 | 37.20% |
| | Val | 5.7208 | 10.1315 | 0.8068 | 90.18% |
| | Test | 7.1728 | 13.3907 | 0.2405 | 566.50% |
| **XGBoost** | Train | 0.0758 | 0.1140 | 1.0000 | 1.85% |
| | Val | 6.1349 | 11.0867 | 0.7687 | 84.65% |
| | Test | 8.7681 | 16.7147 | -0.1834 | 515.80% |
| **CatBoost** | Train | 3.2451 | 5.5263 | 0.9459 | 85.27% |
| | Val | 6.1237 | 11.5623 | 0.7484 | 97.50% |
| | Test | 7.3410 | 13.3062 | 0.2500 | 653.89% |

*LightGBM* was selected as the best overall model due to the lowest validation RMSE (10.1315) and serialized to `outputs/models/best_model.pkl`.

---

## 5. Feature Importance Analysis

Feature importances were evaluated using both **Gain-based Split Importance** (model-native) and **Permutation Importance** on the validation set:

1. **Prior Prior State (`prev_gw` / `gw_diff`)**: By far the most critical predictor of future groundwater level. Permutation importance scores `prev_gw` as the primary contributor to score preservation.
2. **Spatial Location / Coordinates**: `station_id` and `rainfall_distance_km` provide localized offsets.
3. **Hydrological Cumulative Windows**: `effective_rainfall_180d` and `effective_rainfall_90d` act as the primary hydrological signals, correlating cumulative monsoon precipitation to subsequent recharge.

---

## 6. Error & Failure Mode Analysis

Our detailed error analysis of the LightGBM model on the test set highlights key behavior:
- **Seasonal Failure Modes**: Predictions during the **pre-monsoon** dry season are significantly harder (MAE: 8.04 MBGL) compared to the **post-monsoon** season (MAE: 5.53 MBGL). This is due to dry-season water extraction variations.
- **High-Rainfall Generalization**: High-rainfall seasons (above the 90th percentile, >249mm over 180 days) generalize slightly better (MAE: 7.04 MBGL) than normal rainfall seasons (MAE: 7.63 MBGL), showing that the model successfully captures monsoon recharge effects.
- **Routing Type Performance**: Fallback routing (district/state averages) yields lower test errors (MAE: 6.99 MBGL) than local telemetry matching (MAE: 8.28 MBGL). The localized signals have high variance, while spatial averages reduce prediction variance.

---

## 7. Limitations & Future Work

1. **High Cardinality overfitting**: Training with `station_id` leads to spatial overfitting (train R2: 0.99, test R2: 0.24). Future iterations should drop `station_id` and rely on smooth spatial features (Latitude, Longitude, Soil properties, Elevation).
2. **Extreme Outlier Stations**: A small subset of deep-aquifer stations (e.g., IDs `131110` and `130714`) exhibit errors exceeding 85 MBGL, skewing overall test RMSE. Special handling or separate models should be trained for deep vs. shallow aquifers.
3. **Meteorological Forecasts Integration**: Currently, the model predicts the next season using historical rainfall. Integrating forward-looking monsoon forecasts would enable true early-warning forecasting.
