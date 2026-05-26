# NEERA Hyperparameter Tuning & Model Search Report

This report summarizes the automated hyperparameter tuning and model search results. Tuning was conducted via Optuna using strict temporal splits.

## 1. Tuning Configurations
- **Optuna Trials**: LightGBM (30), XGBoost (30), CatBoost (15), RandomForest (15), ExtraTrees (15), Ridge (30), ElasticNet (30)
- **Features Excluded**: `station_id` category (to prevent memorization and overfitting), timestamp indices.
- **Ensemble Candidates**: LightGBM, XGBoost, CatBoost
- **Ensemble Weights**: {'LightGBM': np.float64(0.0), 'XGBoost': np.float64(1.0), 'CatBoost': np.float64(0.0)}

---

## 2. Model Performance Summary
Below is the evaluation of all tuned models and the weighted ensemble across train (<=2019), validation (2020), and test (>=2021) sets:

| Model | Split | MAE (MBGL) | RMSE (MBGL) | $R^2$ | MAPE (%) |
|---|---|---|---|---|---|
| **LightGBM** | train | 2.1959 | 3.6015 | 0.9770 | 56.27% |
| **LightGBM** | val | 5.4261 | 10.1937 | 0.8045 | 78.29% |
| **LightGBM** | test | 5.8034 | 11.6313 | 0.4269 | 376.97% |
| **XGBoost** | train | 3.4349 | 5.4723 | 0.9469 | 104.30% |
| **XGBoost** | val | 5.4645 | 8.9100 | 0.8506 | 108.99% |
| **XGBoost** | test | 6.7026 | 11.5924 | 0.4308 | 550.51% |
| **CatBoost** | train | 3.6521 | 5.9812 | 0.9366 | 117.80% |
| **CatBoost** | val | 5.5355 | 9.5240 | 0.8293 | 110.92% |
| **CatBoost** | test | 6.2012 | 11.1486 | 0.4735 | 529.09% |
| **RandomForest** | train | 1.9840 | 3.9115 | 0.9729 | 45.68% |
| **RandomForest** | val | 5.5131 | 10.1743 | 0.8052 | 83.42% |
| **RandomForest** | test | 5.7815 | 11.8067 | 0.4095 | 411.24% |
| **ExtraTrees** | train | 2.8102 | 4.4017 | 0.9657 | 85.18% |
| **ExtraTrees** | val | 5.2989 | 9.9505 | 0.8137 | 85.38% |
| **ExtraTrees** | test | 5.9009 | 11.8265 | 0.4075 | 434.32% |
| **Ridge** | train | 3.4294 | 5.8281 | 0.9398 | 79.20% |
| **Ridge** | val | 5.2009 | 9.2445 | 0.8392 | 78.79% |
| **Ridge** | test | 6.1385 | 11.5018 | 0.4396 | 421.42% |
| **ElasticNet** | train | 3.4580 | 5.8569 | 0.9392 | 80.44% |
| **ElasticNet** | val | 5.4023 | 9.4327 | 0.8326 | 81.88% |
| **ElasticNet** | test | 5.9717 | 11.4515 | 0.4445 | 434.90% |
| **WeightedEnsemble** | train | 3.4349 | 5.4723 | 0.9469 | 104.30% |
| **WeightedEnsemble** | val | 5.4645 | 8.9100 | 0.8506 | 108.99% |
| **WeightedEnsemble** | test | 6.7026 | 11.5924 | 0.4308 | 550.51% |

---

## 3. Best Parameters Found
- **LightGBM**: `{'n_estimators': 354, 'learning_rate': 0.14992353593812363, 'num_leaves': 96, 'max_depth': 6, 'min_child_samples': 94, 'subsample': 0.8971183910834973, 'colsample_bytree': 0.5186302652603891, 'reg_alpha': 3.854724745113838, 'reg_lambda': 0.0001014716251675725}`
- **XGBoost**: `{'n_estimators': 105, 'learning_rate': 0.01891937563955447, 'max_depth': 10, 'min_child_weight': 10, 'subsample': 0.5490991015481423, 'colsample_bytree': 0.6310801892855292, 'alpha': 0.0020629202814494165, 'reg_lambda': 0.11537833837415117}`
- **CatBoost**: `{'iterations': 121, 'learning_rate': 0.023983671223302503, 'depth': 7, 'l2_leaf_reg': 3.056350360289917}`
- **RandomForest**: `{'n_estimators': 181, 'max_depth': 19, 'min_samples_split': 6, 'min_samples_leaf': 6}`
- **ExtraTrees**: `{'n_estimators': 124, 'max_depth': 8, 'min_samples_split': 2, 'min_samples_leaf': 1}`
- **Ridge**: `{'alpha': 0.001028772087982792}`
- **ElasticNet**: `{'alpha': 0.00010546846275726294, 'l1_ratio': 0.8486279838252618}`

---

## 4. Key Recommendations & Selection
1. **Best Model Selection**: The model chosen for production is **XGBoost** with a validation RMSE of **8.9100**.
2. **Generalization Profile**: The test set (2021-2022) performance shows the models' capability on out-of-distribution temporal data (post-2020 rainfall patterns). Removing `station_id` category and relying on KMeans clusters and raw coordinates has substantially reduced the train-test overfitting gap compared to baseline models.
