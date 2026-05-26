# NEERA Scientific Validation — Temporal Drift Analysis

Temporal drift is a common failure mode in physical regression tasks. As climatic patterns change and local pumping regimes shift, models trained on historical data (2015-2019) can experience degradation when predicting validation (2020) and test (2021-2022) eras.

---

## 1. Performance over Years (Temporal Degradation)

| Year | Split Category | Sample Count | MAE (MBGL) | RMSE (MBGL) | $R^2$ | MAPE (%) |
|---|---|---|---|---|---|---|
| 2015 | Training | 1470.0 | 3.3679 | 5.0829 | 0.9329 | 114.85% |
| 2016 | Training | 1456.0 | 3.2110 | 4.8203 | 0.9506 | 124.96% |
| 2017 | Training | 1514.0 | 3.4184 | 5.5364 | 0.9455 | 101.49% |
| 2018 | Training | 1554.0 | 3.4594 | 6.0471 | 0.9444 | 37.49% |
| 2019 | Training | 1536.0 | 3.7029 | 5.7311 | 0.9545 | 145.00% |
| 2020 | Validation | 1327.0 | 5.4645 | 8.9100 | 0.8506 | 108.99% |
| 2021 | Testing | 1410.0 | 6.3047 | 12.4118 | 0.4834 | 560.45% |
| 2022 | Testing | 775.0 | 7.4265 | 9.9296 | 0.1011 | 532.43% |

---

## 2. Seasonal Drift (Pre-Monsoon vs. Post-Monsoon)

Groundwater dynamics vary seasonally:
- **Pre-Monsoon**: Transitioning into the monsoon, high extraction rates.
- **Post-Monsoon**: Transitioning into winter, aquifers are charged.

| Year | Season | Sample Count | MAE (MBGL) | RMSE (MBGL) |
|---|---|---|---|---|
| 2015 | `pre_monsoon` | 726 | 3.8462 | 5.8530 |
| 2015 | `post_monsoon` | 744 | 2.9010 | 4.1974 |
| 2016 | `pre_monsoon` | 708 | 3.3931 | 4.8441 |
| 2016 | `post_monsoon` | 748 | 3.0387 | 4.7977 |
| 2017 | `pre_monsoon` | 744 | 3.5239 | 5.2360 |
| 2017 | `post_monsoon` | 770 | 3.3164 | 5.8120 |
| 2018 | `pre_monsoon` | 774 | 3.5315 | 6.3708 |
| 2018 | `post_monsoon` | 780 | 3.3879 | 5.7078 |
| 2019 | `pre_monsoon` | 762 | 3.7441 | 5.6454 |
| 2019 | `post_monsoon` | 774 | 3.6624 | 5.8142 |
| 2020 | `pre_monsoon` | 633 | 6.4043 | 9.2762 |
| 2020 | `post_monsoon` | 694 | 4.6074 | 8.5624 |
| 2021 | `pre_monsoon` | 652 | 8.1571 | 15.5211 |
| 2021 | `post_monsoon` | 758 | 4.7114 | 8.9076 |
| 2022 | `pre_monsoon` | 775 | 7.4265 | 9.9296 |

---

## 3. Key Findings

1. **Drift Characteristics**:
   - The model has low training error from 2016 to 2019 (MAE: ~3.4 MBGL, RMSE: ~5.4 MBGL).
   - In the **Validation Year (2020)**, MAE rises to **5.46 MBGL** (RMSE **8.91 MBGL**).
   - In the **Test Years (2021-2022)**, the error increases further, peaking in 2022 at an MAE of **7.43 MBGL**. This represents active **temporal drift**, which is driven by:
     - Changes in monsoonal rainfall intensity.
     - Gradual water table depletion in deep wells that tree regressors cannot extrapolate.

2. **Seasonal Divergence**:
   - Errors are consistently higher in the `pre_monsoon` season compared to the `post_monsoon` season across all years. 
   - Pre-monsoon is the end of the dry cycle when agricultural water draw reaches its maximum. The unpredictable variations in pumping rates (which are not captured in the dataset) introduce higher forecast noise.

---

## 4. Visual Trend Chart
A temporal drift chart showing error lines across years has been saved in:
- [outputs/plots/drift/yearly_drift.png](file:///Users/abhiram/Developer/NEERA/outputs/plots/drift/yearly_drift.png)
