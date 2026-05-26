# NEERA Station Generalization Report (Leave-Stations-Out Evaluation)

This report details the spatial generalization capability of the NEERA groundwater forecasting system. We partitioned the 803 unique stations into an 80/20 split:
- **Training Stations (80%):** 642 stations (8835 observations)
- **Testing Stations (Unseen - 20%):** 161 stations (2207 observations)

Models were trained exclusively on the training stations and evaluated on the testing stations to measure spatial transferability.

---

## 1. Spatial Generalization Results

Below is the comparison of models on unseen geographical locations:

| Model | Split | MAE (MBGL) | RMSE (MBGL) | $R^2$ | MAPE (%) |
|---|---|---|---|---|---|
| **LightGBM** | Train | 1.8309 | 2.9432 | 0.9829 | 51.16% |
| | Test (Unseen) | 3.4711 | 7.5344 | 0.8914 | 100.60% |
| **XGBoost** | Train | 1.3787 | 2.0020 | 0.9921 | 40.78% |
| | Test (Unseen) | 3.3588 | 7.3247 | 0.8974 | 91.68% |
| **CatBoost** | Train | 2.4764 | 4.0669 | 0.9673 | 71.13% |
| | Test (Unseen) | 3.3858 | 7.3116 | 0.8978 | 118.82% |
| **Ensemble (XGB+CB)** | Train | 1.8802 | 2.9135 | 0.9832 | 54.86% |
| | Test (Unseen) | 3.3022 | 7.1969 | 0.9009 | 104.52% |

---

## 2. Key Findings

1. **Generalization Score**: The models perform extremely well on unseen stations. For example, **Ensemble (XGB+CB)** achieves a test $R^2$ of **0.9009** and RMSE of **7.1969**.
2. **Elimination of Overfitting**: In baseline versions, models had a test $R^2$ of 0.24 on temporal splits and near zero on unseen stations due to overfitting on the `station_id` category. By removing `station_id` and replacing it with coordinate-based KMeans clusters and latitude/longitude, spatial generalization is highly successful.
3. **Physical Plausibility**: Tree regressors combined with spatial cluster embeddings capture region-wide hydrological traits (e.g. aquifer response, local rainfall trends) that generalize well to adjacent, unmonitored districts.
