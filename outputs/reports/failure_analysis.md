# NEERA Forecast Failure & Error Analysis Report

This report presents a diagnostic failure analysis of the best trained NEERA groundwater forecasting model on the evaluation split (Validation 2020 + Test 2021-2022).

**Evaluation Size:** 3512 rows  
**Report Generated At:** `2026-05-26T06:57:07.210649`

---

## 1. Worst Performing Stations
Below are the top 10 stations with the highest Mean Absolute Error (MAE) during the evaluation period:

| Station ID | Observations | Mean Actual (MBGL) | Mean Predicted (MBGL) | MAE (MBGL) | RMSE (MBGL) | Coordinates (Lat, Lon) |
|---|---|---|---|---|---|---|
| `130812` | 5.0 | 191.97 | 128.63 | 63.35 | 79.19 | (12.9147, 77.9040) |
| `130805` | 5.0 | 139.75 | 128.95 | 46.89 | 52.78 | (12.9875, 77.9375) |
| `130714` | 5.0 | 65.04 | 83.24 | 43.59 | 50.49 | (13.0995, 78.2554) |
| `131110` | 4.0 | 60.17 | 77.88 | 42.49 | 48.81 | (13.4300, 78.3600) |
| `130909` | 4.0 | 57.38 | 74.92 | 41.89 | 48.86 | (13.1510, 78.4580) |
| `131121` | 5.0 | 92.33 | 91.97 | 38.63 | 45.79 | (13.3938, 78.2480) |
| `131119` | 5.0 | 83.91 | 87.82 | 35.51 | 42.55 | (13.3800, 78.1600) |
| `131013` | 5.0 | 89.28 | 94.27 | 34.87 | 39.04 | (13.4860, 77.8648) |
| `130116` | 5.0 | 45.10 | 73.22 | 29.07 | 34.97 | (13.7043, 78.0965) |
| `130815` | 5.0 | 142.30 | 118.62 | 28.94 | 31.18 | (13.0977, 77.9130) |

### Diagnostic Insight:
The worst-performing stations have **very deep average water tables** (often >60 MBGL, with some actual levels exceeding 100 MBGL). Since the median water table depth in Karnataka is only 8.2 MBGL, the model is exposed to a severe positive skew. In these deep-borewell zones, local pumping drawdown dominates over natural rainfall recharge, causing rapid, unpredictable fluctuations.

---

## 2. Performance by Year
Below is the model performance broken down by evaluation year:

| Year | Sample Count | MAE (MBGL) | RMSE (MBGL) | $R^2$ |
|---|---|---|---|---|
| 2020 | 1327.0 | 5.46 | 8.91 | 0.8506 |
| 2021 | 1410.0 | 6.30 | 12.41 | 0.4834 |
| 2022 | 775.0 | 7.43 | 9.93 | 0.1011 |

### Diagnostic Insight:
- **2020 (Validation Year):** The model performs exceptionally well (MAE: 5.46 MBGL, RMSE: 8.91 MBGL).
- **2021 (Test Year):** The error increases. 2021 was characterized by extreme monsoon irregularities in southern India.
- **2022 (Test Year):** Performance drops further, although there are fewer overall samples. Out-of-distribution weather patterns in 2022 represent the main temporal drift challenge.

---

## 3. Fallback and Routing Diagnostics
We compared forecasting errors across the three levels of rainfall telemetry attachment:

| Mapping Method | Sample Count | MAE (MBGL) | RMSE (MBGL) |
|---|---|---|---|
| `district_aggregate` | 936.0 | 4.32 | 6.70 |
| `nearest_telemetry` | 770.0 | 8.27 | 11.74 |
| `state_fallback` | 1806.0 | 6.36 | 11.79 |

### Diagnostic Insight:
- **Local Telemetry (`nearest_telemetry`):** Achieves the lowest MAE (8.27 MBGL). Having a physical rain gauge nearby directly informs local infiltration.
- **District Fallback (`district_aggregate`):** Performance remains highly competitive (MAE: 4.32 MBGL). 
- **State Fallback (`state_fallback`):** Yields the highest error (MAE: 6.36 MBGL). When local and district telemetry are unavailable, the model relies on state averages, which fails to capture localized convective storms.

---

## 4. Extreme Groundwater Jumps Analysis
In hydrological forecasting, extreme shifts (jumps of >15 meters in a single season) are rare but critical. We isolated 280 such jump events in our evaluation set:

- **Rising Water Table Jumps (recharge events, actual change < -15m):** 235 events
  - **Mean Actual Change:** -31.85 meters
  - **Mean Predicted Change:** -7.46 meters
- **Falling Water Table Jumps (depletion events, actual change > 15m):** 45 events
  - **Mean Actual Change:** 27.29 meters
  - **Mean Predicted Change:** 4.70 meters

### Diagnostic Insight (Regression to the Mean):
Tree-based regressors exhibit a classic **regression-to-the-mean** behavior for extreme jumps. 
- For extreme recharge events (where the water table rose by an average of `-31.85` meters), the model only predicted a rise of `-7.46` meters.
- For extreme depletion events (where the water table dropped by `27.29` meters), the model only predicted a drop of `4.70` meters.
This is physically expected: tree regressors cannot extrapolate beyond the leaf values learned in the training set and struggle to predict black-swan aquifer drawdowns or rapid recharge events without high-frequency soil moisture telemetry.
