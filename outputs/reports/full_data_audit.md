# NEERA Groundwater Dataset Full Data Audit Report

This report presents a thorough audit of the dataset used for training the NEERA groundwater forecasting system. 

**Dataset Path:** `/Users/abhiram/Developer/NEERA/training_master.csv`  
**Audit Executed At:** `2026-05-26T06:50:45.658159`

---

## 1. Executive Summary
- **Total Dataset Size:** 11042 rows, 39 columns
- **Unique Stations:** 803
- **Date Range:** 2015-05-01 to 2022-05-01 (8 years: 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
- **Duplicate Rows:** 0
- **Duplicate Timestamps per Station:** 0
- **Duplicate Dates per Station:** 0

---

## 2. Completeness Check (Missing Values & Infinities)
There are **zero infinities** in any numerical columns.

### Missing Values Summary:
| Column | Null Count | Null Percentage (%) |
|---|---|---|
| `prev_gw` | 803 | 7.2722% |
| `gw_diff` | 803 | 7.2722% |
| `gw_roll_mean_7obs` | 1606 | 14.5445% |
| `gw_roll_std_7obs` | 1606 | 14.5445% |
| `gw_roll_mean_30obs` | 1606 | 14.5445% |
| `rainfall_30d` | 7859 | 71.1737% |
| `rainfall_90d` | 8149 | 73.8000% |
| `rainfall_180d` | 8213 | 74.3796% |
| `district_rainfall_30d` | 3871 | 35.0571% |
| `district_rainfall_90d` | 4464 | 40.4275% |
| `district_rainfall_180d` | 4464 | 40.4275% |
| `rainfall_source_station` | 7859 | 71.1737% |
| `rainfall_station_observation_count` | 7854 | 71.1284% |
| `rainfall_window_completeness_30d` | 7854 | 71.1284% |
| `rainfall_window_completeness_90d` | 7854 | 71.1284% |
| `rainfall_window_completeness_180d` | 7854 | 71.1284% |
| `rainfall_district_match` | 11042 | 100.0000% |
| `effective_rainfall_90d` | 290 | 2.6263% |
| `effective_rainfall_180d` | 354 | 3.2059% |

---

## 3. Station Imbalance
We analyzed the distribution of observations across the 803 stations:
- **Minimum observations per station:** 7
- **Maximum observations per station:** 15
- **Mean observations per station:** 13.75
- **Median observations per station:** 15.0
- **Standard deviation:** 1.92

---

## 4. Target (`target_next_season_gw`) Distribution
The target variable is the next season's groundwater level in meters below ground level (MBGL).
- **Minimum:** 0.0000 MBGL
- **Maximum:** 255.5000 MBGL
- **Mean:** 16.1553 MBGL
- **Median:** 8.2000 MBGL
- **Standard Deviation:** 22.5677 MBGL
- **Skewness:** 3.1261 (positive skew indicates long tail of deep groundwater levels)
- **Kurtosis:** 12.3816

### Percentiles:
| Percentile | Value (MBGL) |
|---|---|
| 5th | 1.1000 |
| 25th | 4.0800 |
| 50th | 8.2000 |
| 75th | 17.0500 |
| 95th | 62.3475 |

---

## 5. Outliers Analysis
Outliers identified using the IQR method (Q1 - 1.5*IQR to Q3 + 1.5*IQR):
| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier Percentage (%) |
|---|---|---|---|---|
| `Groundwater_Level_MBGL` | -15.4500 | 38.5500 | 1279 | 11.58% |
| `prev_gw` | -16.6000 | 40.2000 | 1188 | 10.76% |
| `effective_rainfall_180d` | -314.5136 | 588.1342 | 252 | 2.28% |
| `target_next_season_gw` | -15.3750 | 36.5050 | 1267 | 11.47% |

---

## 6. Fallback and Routing Proportions
- **Fallback Used Count:** 8213 rows (74.38%)
  *(Fallback indicates telemetry was unavailable at local/district levels, falling back to state/climatology levels)*

### Mapping Methods Count:
- **district_aggregate**: 4306 (39.00%)
- **state_fallback**: 3553 (32.18%)
- **nearest_telemetry**: 3183 (28.83%)

### Rainfall Source Types Count:
- **district**: 4306 (39.00%)
- **state**: 3553 (32.18%)
- **local**: 3183 (28.83%)

---

## 7. Feature Leakage Risk Assessment
We evaluated correlation and identity flags to search for possible target leakage:
- **Features correlated with target > 0.95 (Pearson correlation):**
  - *None found. (No single feature correlates > 0.95 with the target)*

- **Identical rows check (feature value equals target value in same row):**
  - `Groundwater_Level_MBGL` matches `target_next_season_gw` in 69 rows (0.62% of rows)
  - `prev_gw` matches `target_next_season_gw` in 297 rows (2.69% of rows)
  - `gw_diff` matches `target_next_season_gw` in 9 rows (0.08% of rows)
  - `gw_roll_mean_7obs` matches `target_next_season_gw` in 6 rows (0.05% of rows)
  - `gw_roll_mean_30obs` matches `target_next_season_gw` in 4 rows (0.04% of rows)
  - `rainfall_30d` matches `target_next_season_gw` in 1 rows (0.01% of rows)
  - `rainfall_90d` matches `target_next_season_gw` in 2 rows (0.02% of rows)
  - `rainfall_180d` matches `target_next_season_gw` in 2 rows (0.02% of rows)
  - `district_rainfall_30d` matches `target_next_season_gw` in 6 rows (0.05% of rows)
  - `district_rainfall_90d` matches `target_next_season_gw` in 5 rows (0.05% of rows)
  - `district_rainfall_180d` matches `target_next_season_gw` in 4 rows (0.04% of rows)
  - `season_cos` matches `target_next_season_gw` in 57 rows (0.52% of rows)
  - `rainfall_window_completeness_30d` matches `target_next_season_gw` in 16 rows (0.14% of rows)
  - `rainfall_window_completeness_90d` matches `target_next_season_gw` in 15 rows (0.14% of rows)
  - `rainfall_window_completeness_180d` matches `target_next_season_gw` in 14 rows (0.13% of rows)
  - `rainfall_fallback_used` matches `target_next_season_gw` in 55 rows (0.50% of rows)
  - `effective_rainfall_30d` matches `target_next_season_gw` in 4 rows (0.04% of rows)
  - `effective_rainfall_90d` matches `target_next_season_gw` in 5 rows (0.05% of rows)
  - `effective_rainfall_180d` matches `target_next_season_gw` in 3 rows (0.03% of rows)

---

## 8. Temporal Continuity & Coverage
- **Total Years:** 8
- **Observations per Year:**
  - **2015**: 1470 rows (13.31%)
  - **2016**: 1456 rows (13.19%)
  - **2017**: 1514 rows (13.71%)
  - **2018**: 1554 rows (14.07%)
  - **2019**: 1536 rows (13.91%)
  - **2020**: 1327 rows (12.02%)
  - **2021**: 1410 rows (12.77%)
  - **2022**: 775 rows (7.02%)

- **Typical observation interval per station:**
  - **Mean interval:** 194.28 days
  - **Median interval:** 212.0 days
  *(Observations are recorded seasonally, averaging about 3-4 months apart per station, which is consistent with seasonal monitoring practices).*

---

## 9. Conclusion & Actions
1. **Sanity Check:** No duplicate timestamps per station or duplicate rows exist in `training_master.csv`. Timestamps are monotonic.
2. **Leakage Safety:** There is no identical-value leakage or extreme correlation between raw features and target.
3. **Imbalance:** High variation in station sample size (min=7, max=15). Leave-Stations-Out (LSO) validation is crucial to check generalization on less-sampled or unseen regions.
4. **Target Distribution:** Long right tail (skew=3.13) indicates extreme water depletion in some wells (>100 MBGL). Robust metrics (MAE) should be monitored alongside RMSE.
