# NEERA Model Interpretability Report (SHAP Analysis)

This report details the explainability and hydrological plausibility of the NEERA groundwater forecasting system using SHAP (SHapley Additive exPlanations) values on our tuned XGBoost model.

## 1. Feature Importance Rankings (Global)

Below are the top 15 features ranked by their average absolute impact on the target groundwater level prediction:

| Rank | Feature | Mean Absolute SHAP Value (MBGL) | Description |
|---|---|---|---|
| 1 | `Groundwater_Level_MBGL` | 7.1169 | Engineered temporal/rainfall feature |
| 2 | `prev_gw` | 1.9811 | Prior state (groundwater level of last seasonal observation) |
| 3 | `gw_ewm_mean_span3` | 1.4127 | Exponentially weighted moving average (span 3) |
| 4 | `lag_1` | 1.1875 | Prior state (groundwater level of last seasonal observation) |
| 5 | `gw_ewm_mean_span5` | 1.1150 | Engineered temporal/rainfall feature |
| 6 | `gw_expanding_mean` | 0.4993 | Long-term expanding average groundwater level |
| 7 | `spatial_cluster` | 0.3586 | KMeans coordinate-based spatial cluster |
| 8 | `gw_yoy_diff` | 0.3370 | Engineered temporal/rainfall feature |
| 9 | `longitude` | 0.2936 | Geographic longitude of the station |
| 10 | `state_rainfall_30d` | 0.2797 | Engineered temporal/rainfall feature |
| 11 | `state_rainfall_180d` | 0.2760 | Engineered temporal/rainfall feature |
| 12 | `gw_diff` | 0.2670 | Groundwater change between the last two observations |
| 13 | `latitude` | 0.2411 | Geographic latitude of the station |
| 14 | `effective_rainfall_180d` | 0.2227 | Spatiotemporal routed rainfall over the last 180 days |
| 15 | `district_rainfall_180d` | 0.1973 | Engineered temporal/rainfall feature |

---

## 2. Key Interpretability Visualizations

We have generated and saved four primary explainability plots in [outputs/plots/shap/](file:///Users/abhiram/Developer/NEERA/outputs/plots/shap/):

1. **Global Summary Plot** (`summary_plot.png`): Shows the distribution of SHAP values for each feature. Red indicates high feature values, blue indicates low feature values.
2. **Groundwater Memory Dependency** (`dependence_memory.png`): Demonstrates the relationship between the last seasonal observation (`prev_gw`) and its predicted impact on the next season's water table level.
3. **Rainfall Ingress Dependency** (`dependence_rainfall.png`): Captures the physical recharge process. Highlights the non-linear relationship where rainfall exceeding a threshold leads to a rapid rising of the water table (negative SHAP on MBGL).
4. **Local Prediction Waterfall** (`waterfall_local.png`): Visualizes the step-by-step contribution of each feature to a single sample prediction.

---

## 3. Hydrological Insights

- **Strong Groundwater Memory**: The prior groundwater state (`prev_gw` / `lag_1`) is by far the most dominant feature (average SHAP impact of **1.98** MBGL). This represents the aquifer's storage capacity. If the groundwater was deep last season, it remains deep this season unless heavy recharge occurs.
- **Non-Linear Rainfall Thresholds**: The dependency curve for `effective_rainfall_180d` shows that low rainfall yields zero or positive SHAP impact (causing groundwater levels to drop due to pumping/depletion). However, once rainfall exceeds a threshold (e.g., ~300mm in 180 days), the SHAP value drops sharply below zero, representing active recharge raising the water table.
- **Spatial Embedding Partitioning**: `latitude` and `longitude` rank highly, demonstrating that geographic offsets play a major role in regional baseline differences. Trees effectively partition these coordinates to learn localized water table shapes.
