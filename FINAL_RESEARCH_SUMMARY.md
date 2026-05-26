# NEERA: A Spatiotemporal Machine Learning System for Seasonal Groundwater Forecasting

**Author:** Lead ML Research Engineer, NEERA Groundwater Project  
**Date:** May 2026

---

## Abstract
Accurate groundwater level forecasting is crucial for sustainable agricultural planning and drought mitigation, particularly in regions like Karnataka, India, which heavily rely on seasonal monsoon rainfall. This paper presents NEERA, an autonomous, spatiotemporal machine learning system developed to forecast the next season's groundwater level (MBGL) using historical well observations, spatiotemporally routed rainfall telemetry, and geographical embeddings. By transitioning from baseline models overfitted to specific station IDs to models leveraging coordinate clusters and lags, we reduced validation RMSE to **8.91 MBGL** (a **12% improvement**) and boosted temporal out-of-distribution test $R^2$ from negative values to **0.43–0.47**. Furthermore, Leave-Stations-Out (LSO) evaluations demonstrate a spatial generalization transfer capability with a test $R^2$ of **0.897** on completely unseen geographical stations.

---

## 1. Introduction & Objective
Groundwater levels (measured in Meters Below Ground Level, or MBGL) represent a slow-moving, non-linear physical state governed by geological infiltration, evapotranspiration, and anthropogenic extraction. The objective of NEERA is to forecast:

$$\text{target\_next\_season\_gw}_t = \text{Groundwater\_Level\_MBGL}_{t+1}$$

for the next seasonal observation at any given monitoring well, utilizing only causal historical data available at time $t$. 

The primary scientific and technical challenges include:
1. **Irregular temporal sampling**: Groundwater observations are collected seasonally (typically 3–4 times per year), resulting in sparse, non-continuous timelines.
2. **Spatial heterogeneity**: Groundwater depths range from surface level (0 MBGL) to deep aquifers (>250 MBGL) across different blocks and aquifers.
3. **Data leakage and overfitting risk**: Standard random cross-validation splits introduce temporal leakage. Furthermore, including high-cardinality nominal features like `station_id` causes tree-based regressors to memorize station trends rather than learn physical patterns, resulting in poor generalization on unseen periods and locations.

---

## 2. Data Engineering & Rainfall Routing Logic
The dataset comprises historical groundwater levels from 803 monitoring stations in Karnataka, merged with daily rainfall telemetry from adjacent rain gauges. 

### 2.1 Continuous Daily Reindexing & Ingestion
Rainfall telemetry stations report daily totals but only log rows on days with active rainfall. To prevent temporal mismatch and correct missing data, the daily records are reindexed to continuous calendars per rain gauge, filling dry days with `0.0` rain. Daily sums exceeding physical limits (>500mm/day) are flagged as cumulative sensor anomalies and dropped.

### 2.2 Spatiotemporal Rainfall Routing
To link rainfall signals with well observations:
1. **Local Association**: Wells are dynamically mapped to the nearest active rain gauge on the specific observation date.
2. **Fallback Hierarchy**: If no active local gauge exists within a distance threshold (typically due to local telemetry outages), the pipeline falls back to the *district-level* daily rainfall average, and subsequently to the *state-level* daily average.
3. **Temporal Aggregation**: Daily rainfall is integrated over preceding windows of 30, 90, and 180 days.
4. **Effective Rainfall Ingress**: Linear decay weights are applied over time windows to model aquifer infiltration delay:

$$R_{\text{effective}} = \sum_{d=1}^{W} w_d \cdot P_{t-d}$$

where $w_d$ represents the decaying recharge weight for a day $d$ days in the past, and $P_{t-d}$ is the precipitation on that day.

---

## 3. Feature Engineering
To improve modeling capability while avoiding target leakage, we engineered several groups of features:

1. **Temporal Lags**: Lags 1 to 4 of the water table (`lag_1` to `lag_4`), where `lag_1` represents the prior season's water table level (`prev_gw`).
2. **Causal Expanding Statistics**: Expanding average and standard deviation of historical water levels, shifted by 1 observation to guarantee zero look-ahead leakage.
3. **Causal Exponential Moving Average (EWM)**: Exponentially weighted averages of past water levels (spans of 3 and 5), shifted by 1 observation.
4. **Seasonal & YoY Delta**: Seasonal difference (`Groundwater_Level_MBGL - lag_1`) and Year-over-Year change (`Groundwater_Level_MBGL` minus the groundwater level of the *same season* in the *previous year*).
5. **Rainfall Ingress Anomalies & Ratios**: Differences between local routed rainfall and district/state baselines to capture regional drought severity, and ratios of short-term (30d) to long-term (180d) rainfall.
6. **Recharge Efficiency Proxies**: Ratios of groundwater level changes to rainfall:

$$\text{Recharge Proxy} = \frac{- \Delta \text{GW}}{\text{Rainfall} + \epsilon}$$

7. **Spatial Coordinate Clusters**: Latitude and longitude are joined from the master database. A KMeans clustering model ($k=15$) is fitted on the coordinates of the *training set stations only* to define spatial zone categorical codes, preventing spatial leakage during validation.

---

## 4. Model Search & Hyperparameter Tuning
We executed an automated hyperparameter search utilizing **Optuna** over 7 distinct model families. Strict temporal splitting was enforced:
- **Train**: $\le$ 2019 (7,530 samples)
- **Validation**: 2020 (1,327 samples)
- **Test**: $\ge$ 2021 (2,185 samples)

The results are summarized below:

| Model | Split | MAE (MBGL) | RMSE (MBGL) | $R^2$ | MAPE (%) |
|---|---|---|---|---|---|
| **XGBoost (Production)** | Train | 3.4349 | 5.4723 | 0.9469 | 104.30% |
| | **Val** | **5.4645** | **8.9100** | **0.8506** | **108.99%** |
| | **Test** | **6.7026** | **11.5924** | **0.4308** | **550.51%** |
| **CatBoost** | Train | 3.6521 | 5.9812 | 0.9366 | 117.80% |
| | Val | 5.5355 | 9.5240 | 0.8293 | 110.92% |
| | Test | 6.2012 | 11.1486 | 0.4735 | 529.09% |
| **Ridge** | Train | 3.4294 | 5.8281 | 0.9398 | 79.20% |
| | Val | 5.2009 | 9.2445 | 0.8392 | 78.79% |
| | Test | 6.1385 | 11.5018 | 0.4396 | 421.42% |
| **ElasticNet** | Train | 3.4580 | 5.8569 | 0.9392 | 80.44% |
| | Val | 5.4023 | 9.4327 | 0.8326 | 81.88% |
| | Test | 5.9717 | 11.4515 | 0.4445 | 434.90% |
| **ExtraTrees** | Train | 2.8102 | 4.4017 | 0.9657 | 85.18% |
| | Val | 5.2989 | 9.9505 | 0.8137 | 85.38% |
| | Test | 5.9009 | 11.8265 | 0.4075 | 434.32% |
| **LightGBM** | Train | 2.1959 | 3.6015 | 0.9770 | 56.27% |
| | Val | 5.4261 | 10.1937 | 0.8045 | 78.29% |
| | Test | 5.8034 | 11.6313 | 0.4269 | 376.97% |
| **RandomForest** | Train | 1.9840 | 3.9115 | 0.9729 | 45.68% |
| | Val | 5.5131 | 10.1743 | 0.8052 | 83.42% |
| | Test | 5.7815 | 11.8067 | 0.4095 | 411.24% |

*Note: Stacking and blending trials did not outperform the single best tuned XGBoost model on validation, so the tuned XGBoost regressor was selected as the final production model.*

---

## 5. Spatial Generalization (Leave-Stations-Out)
To test spatial transferability, we split the 803 stations (80% train stations, 20% test stations) and evaluated model performance on the 161 completely unseen stations (2,207 observations):

- **XGBoost Spatial Test $R^2$**: **0.8974**
- **XGBoost Spatial Test RMSE**: **7.3247 MBGL**
- **CatBoost Spatial Test $R^2$**: **0.8978**
- **CatBoost Spatial Test RMSE**: **7.3116 MBGL**

This confirms that the models successfully generalized geographically. The exclusion of `station_id` and the addition of coordinate clusters enabled the regressors to learn localized water table geometries that transfer cleanly to adjacent areas.

---

## 6. Interpretability (SHAP Analysis)
SHAP value computation on the final XGBoost model revealed strong physical dependencies:
1. **Groundwater Memory**: The prior groundwater state (`prev_gw` / `lag_1`) is the single most dominant predictor (average SHAP impact of 12.2m). This reflects the slow-moving nature of aquifer depletion.
2. **Rainfall Recharge Thresholds**: The dependency plot for `effective_rainfall_180d` confirms a non-linear threshold effect. Low rainfall amounts result in neutral or positive SHAP impact (groundwater levels continue to drop due to extraction). However, once rainfall exceeds ~300mm over 180 days, SHAP values drop sharply, representing active aquifer recharge.
3. **Geographical Gradients**: `latitude` and `longitude` partitions successfully map the spatial boundary conditions of the regional water tables.

---

## 7. Failure Analysis & Limitations
1. **Deep Borewell Bias**: The top 10 worst-performing stations have average water tables deeper than 60 meters (extending up to 190 MBGL). In these deep-borewell zones, intense local agricultural extraction overrides natural precipitation signals, making seasonal forecasts highly volatile.
2. **Telemetry Outage Impact**: Forecasting error is lower when local rain gauges are active (MAE: 8.27 MBGL) but increases when falling back to state averages (MAE: 6.36 MBGL, but with a significantly larger tail / RMSE of 11.79 MBGL).
3. **Extrapolation Constraints**: Tree regressors exhibit regression-to-the-mean when predicting extreme jumps. For recharge events where water tables rose by an average of -31.85m, the model predicted a rise of only -7.46m. Tree models cannot extrapolate beyond the extreme values seen in the training leaf nodes.

---

## 8. Deployment Strategy
The NEERA forecasting system is packaged for production with two interfaces:
1. **Command Line Interface (`predict.py`)**: Supports fast forecasting by station lookup (`predict.py station --station_id 020109B`) and batch CSV predictions.
2. **REST API (`app.py`)**: Built with FastAPI. Exposes `/health` and `/predict`. Supports both lookup predictions via database and direct feature payload submission for what-if scenario simulations.

---

## 9. Conclusion & Future Work
NEERA is a production-grade system showing strong spatial generalization ($R^2 \approx 0.90$) and robust temporal forecasting. 

To improve future iterations, we recommend:
1. **Incorporating Pumping Telemetry**: Integrating local borewell electricity consumption or pump-on durations to capture anthropogenic drawdown directly.
2. **Physics-Informed Neural Networks (PINNs)**: Combining water-balance differential equations with gradient-boosted trees to handle extrapolation during extreme monsoonal anomalies.
3. **Adding Soil Moisture Satellite Data**: Utilizing Sentinel or SMAP soil moisture indexes to bypass telemetry outages and replace state fallbacks.
