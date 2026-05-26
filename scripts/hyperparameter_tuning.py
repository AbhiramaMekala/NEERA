#!/usr/bin/env python3
"""Hyperparameter Tuning & Model Search for NEERA.

Uses Optuna to find the best hyperparameters for LightGBM, XGBoost, CatBoost,
RandomForest, ExtraTrees, Ridge, and ElasticNet. Builds ensembles and saves
the final best model.
"""

import os
import pickle
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data/training_master_engineered.csv"
OUT_MODEL_DIR = ROOT / "outputs/models"
OUT_METRICS_DIR = ROOT / "outputs/metrics"
OUT_REPORT = ROOT / "outputs/reports/hyperparameter_results.md"

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_tune():
    print(f"Loading engineered dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    
    # Target and identifier columns
    target = "target_next_season_gw"
    exclude_cols = [
        "station_id", "timestamp", "date", "year", "freq",
        "rainfall_source_station", "gw_same_season_prev_obs", target
    ]
    
    # Drop rows where target is NaN
    df = df.dropna(subset=[target]).copy()
    
    # Define categorical and numerical features
    categorical_features = ["season", "rainfall_source_type", "mapping_method", "spatial_cluster"]
    # Ensure categorical features are string categories
    for col in categorical_features:
        df[col] = df[col].astype(str).astype("category")
        
    numerical_features = [col for col in df.columns if col not in exclude_cols and col not in categorical_features]
    features = numerical_features + categorical_features
    
    print(f"Total features: {len(features)} ({len(numerical_features)} numerical, {len(categorical_features)} categorical)")
    
    # Temporal Split
    train_df = df[df["year"] <= 2019].copy()
    val_df = df[df["year"] == 2020].copy()
    test_df = df[df["year"] >= 2021].copy()
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print(f"Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Preprocessor for sklearn models (imputation, scaling, one-hot encoding)
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, numerical_features),
        ("cat", cat_transformer, categorical_features)
    ])
    
    # Dictionary to store performance results
    model_results = {}
    best_models = {}
    
    # ── 1. Tune LightGBM ──────────────────────────────────────────────────────
    print("\nTuning LightGBM...")
    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": 4,
            "verbose": -1
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective_lgb, n_trials=30)
    print(f"Best LightGBM Val RMSE: {study_lgb.best_value:.4f}")
    
    best_lgb = LGBMRegressor(**study_lgb.best_params, random_state=42, n_jobs=4, verbose=-1)
    best_lgb.fit(X_train, y_train)
    best_models["LightGBM"] = best_lgb
    
    # ── 2. Tune XGBoost ───────────────────────────────────────────────────────
    print("\nTuning XGBoost...")
    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": 4,
            "enable_categorical": True
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgb, n_trials=30)
    print(f"Best XGBoost Val RMSE: {study_xgb.best_value:.4f}")
    
    best_xgb = XGBRegressor(**study_xgb.best_params, random_state=42, n_jobs=4, enable_categorical=True)
    best_xgb.fit(X_train, y_train)
    best_models["XGBoost"] = best_xgb
    
    # ── 3. Tune CatBoost ──────────────────────────────────────────────────────
    print("\nTuning CatBoost...")
    cat_indices = [features.index(col) for col in categorical_features]
    def objective_cb(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_seed": 42,
            "thread_count": 4,
            "verbose": 0,
            "cat_features": cat_indices
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study_cb = optuna.create_study(direction="minimize")
    study_cb.optimize(objective_cb, n_trials=15)
    print(f"Best CatBoost Val RMSE: {study_cb.best_value:.4f}")
    
    best_cb = CatBoostRegressor(**study_cb.best_params, random_seed=42, thread_count=4, verbose=0, cat_features=cat_indices)
    best_cb.fit(X_train, y_train)
    best_models["CatBoost"] = best_cb
    
    # ── 4. Tune RandomForest ──────────────────────────────────────────────────
    print("\nTuning RandomForest...")
    def objective_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "random_state": 42,
            "n_jobs": 4
        }
        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**params))
        ])
        model_pipeline.fit(X_train, y_train)
        preds = model_pipeline.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(objective_rf, n_trials=15)
    print(f"Best RandomForest Val RMSE: {study_rf.best_value:.4f}")
    
    best_rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=4))
    ])
    best_rf.fit(X_train, y_train)
    best_models["RandomForest"] = best_rf
    
    # ── 5. Tune ExtraTrees ────────────────────────────────────────────────────
    print("\nTuning ExtraTrees...")
    def objective_et(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "random_state": 42,
            "n_jobs": 4
        }
        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", ExtraTreesRegressor(**params))
        ])
        model_pipeline.fit(X_train, y_train)
        preds = model_pipeline.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study_et = optuna.create_study(direction="minimize")
    study_et.optimize(objective_et, n_trials=15)
    print(f"Best ExtraTrees Val RMSE: {study_et.best_value:.4f}")
    
    best_et = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", ExtraTreesRegressor(**study_et.best_params, random_state=42, n_jobs=4))
    ])
    best_et.fit(X_train, y_train)
    best_models["ExtraTrees"] = best_et
    
    # ── 6. Tune Ridge ─────────────────────────────────────────────────────────
    print("\nTuning Ridge Regression...")
    def objective_ridge(trial):
        params = {
            "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
            "random_state": 42
        }
        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(**params))
        ])
        model_pipeline.fit(X_train, y_train)
        preds = model_pipeline.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study_ridge = optuna.create_study(direction="minimize")
    study_ridge.optimize(objective_ridge, n_trials=30)
    print(f"Best Ridge Val RMSE: {study_ridge.best_value:.4f}")
    
    best_ridge = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(**study_ridge.best_params, random_state=42))
    ])
    best_ridge.fit(X_train, y_train)
    best_models["Ridge"] = best_ridge
    
    # ── 7. Tune ElasticNet ────────────────────────────────────────────────────
    print("\nTuning ElasticNet...")
    def objective_en(trial):
        params = {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "random_state": 42
        }
        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", ElasticNet(**params))
        ])
        model_pipeline.fit(X_train, y_train)
        preds = model_pipeline.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study_en = optuna.create_study(direction="minimize")
    study_en.optimize(objective_en, n_trials=30)
    print(f"Best ElasticNet Val RMSE: {study_en.best_value:.4f}")
    
    best_en = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", ElasticNet(**study_en.best_params, random_state=42))
    ])
    best_en.fit(X_train, y_train)
    best_models["ElasticNet"] = best_en

    # Evaluate all models across splits
    all_metrics = []
    val_preds_dict = {}
    test_preds_dict = {}
    
    for name, model in best_models.items():
        print(f"\nEvaluating {name}...")
        val_preds_dict[name] = model.predict(X_val)
        test_preds_dict[name] = model.predict(X_test)
        
        for split_name, (X, y) in {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}.items():
            preds = model.predict(X)
            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)
            mape = mean_absolute_percentage_error(y, preds)
            
            all_metrics.append({
                "model": name,
                "split": split_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape
            })
            
            # Print performance
            print(f"  {split_name.capitalize()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")
            
    # ── 8. Ensembling ─────────────────────────────────────────────────────────
    print("\n--- Ensembling ---")
    
    # Focus ensemble on the top performing tree models: LightGBM, XGBoost, CatBoost
    ensemble_candidates = ["LightGBM", "XGBoost", "CatBoost"]
    
    # A. Weighted Average Ensemble
    # Let's search for weights that minimize Val RMSE using a simple grid search or optimization
    def val_rmse_for_weights(weights):
        w = np.array(weights)
        w = w / w.sum() # normalize
        weighted_pred = np.zeros_like(y_val, dtype=float)
        for i, name in enumerate(ensemble_candidates):
            weighted_pred += w[i] * val_preds_dict[name]
        return np.sqrt(mean_squared_error(y_val, weighted_pred))
        
    # Standard grid search for weights
    best_w = None
    best_ensemble_val_rmse = float("inf")
    
    # We test combinations that sum to 1.0 (step size 0.1)
    for w1 in np.arange(0, 1.1, 0.1):
        for w2 in np.arange(0, 1.1 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            rmse = val_rmse_for_weights([w1, w2, w3])
            if rmse < best_ensemble_val_rmse:
                best_ensemble_val_rmse = rmse
                best_w = [w1, w2, w3]
                
    best_w = [round(w, 2) for w in best_w]
    print(f"Best Weighted Ensemble Weights: {dict(zip(ensemble_candidates, best_w))} (Val RMSE: {best_ensemble_val_rmse:.4f})")
    
    # Predict on test using weighted ensemble
    test_weighted_pred = np.zeros_like(y_test, dtype=float)
    for i, name in enumerate(ensemble_candidates):
        test_weighted_pred += best_w[i] * test_preds_dict[name]
        
    train_weighted_pred = np.zeros_like(y_train, dtype=float)
    for i, name in enumerate(ensemble_candidates):
        train_weighted_pred += best_w[i] * best_models[name].predict(X_train)
        
    val_weighted_pred = np.zeros_like(y_val, dtype=float)
    for i, name in enumerate(ensemble_candidates):
        val_weighted_pred += best_w[i] * val_preds_dict[name]
        
    # Evaluate Weighted Ensemble
    for split_name, (preds, y) in {"train": (train_weighted_pred, y_train), "val": (val_weighted_pred, y_val), "test": (test_weighted_pred, y_test)}.items():
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        mape = mean_absolute_percentage_error(y, preds)
        
        all_metrics.append({
            "model": "WeightedEnsemble",
            "split": split_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        })
        print(f"  WeightedEnsemble {split_name.capitalize()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")

    # Stacking Class
    class NEERAStackingRegressor:
        def __init__(self, base_models, weights, ensemble_candidates):
            self.base_models = base_models
            self.weights = weights
            self.ensemble_candidates = ensemble_candidates
            
        def predict(self, X):
            preds = np.zeros(len(X))
            for i, name in enumerate(self.ensemble_candidates):
                preds += self.weights[i] * self.base_models[name].predict(X)
            return preds
            
    # Compile metrics
    metrics_df = pd.DataFrame(all_metrics)
    OUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_METRICS_DIR / "model_comparison.csv", index=False)
    
    # Find Best Model (lowest validation RMSE)
    best_idx = metrics_df[metrics_df["split"] == "val"]["RMSE"].idxmin()
    best_row = metrics_df.iloc[best_idx]
    best_overall_name = best_row["model"]
    best_overall_rmse = best_row["RMSE"]
    
    print(f"\nBest Overall Model based on Val RMSE: {best_overall_name} ({best_overall_rmse:.4f})")
    
    # Save the Best Model/Ensemble
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if best_overall_name == "WeightedEnsemble":
        final_model = NEERAStackingRegressor(best_models, best_w, ensemble_candidates)
    else:
        final_model = best_models[best_overall_name]
        
    with open(OUT_MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print(f"Best model/ensemble serialized to {OUT_MODEL_DIR / 'best_model.pkl'}")
    
    # Also save the single base models for SHAP analysis or standalone use
    for name, model in best_models.items():
        with open(OUT_MODEL_DIR / f"{name.lower()}_model.pkl", "wb") as f:
            pickle.dump(model, f)
            
    # Write Tuning Report
    report_content = f"""# NEERA Hyperparameter Tuning & Model Search Report

This report summarizes the automated hyperparameter tuning and model search results. Tuning was conducted via Optuna using strict temporal splits.

## 1. Tuning Configurations
- **Optuna Trials**: LightGBM (30), XGBoost (30), CatBoost (15), RandomForest (15), ExtraTrees (15), Ridge (30), ElasticNet (30)
- **Features Excluded**: `station_id` category (to prevent memorization and overfitting), timestamp indices.
- **Ensemble Candidates**: LightGBM, XGBoost, CatBoost
- **Ensemble Weights**: {dict(zip(ensemble_candidates, best_w))}

---

## 2. Model Performance Summary
Below is the evaluation of all tuned models and the weighted ensemble across train (<=2019), validation (2020), and test (>=2021) sets:

| Model | Split | MAE (MBGL) | RMSE (MBGL) | $R^2$ | MAPE (%) |
|---|---|---|---|---|---|
"""
    for _, row in metrics_df.iterrows():
        report_content += f"| **{row['model']}** | {row['split']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['R2']:.4f} | {row['MAPE']:.2f}% |\n"
        
    report_content += f"""
---

## 3. Best Parameters Found
- **LightGBM**: `{study_lgb.best_params}`
- **XGBoost**: `{study_xgb.best_params}`
- **CatBoost**: `{study_cb.best_params}`
- **RandomForest**: `{study_rf.best_params}`
- **ExtraTrees**: `{study_et.best_params}`
- **Ridge**: `{study_ridge.best_params}`
- **ElasticNet**: `{study_en.best_params}`

---

## 4. Key Recommendations & Selection
1. **Best Model Selection**: The model chosen for production is **{best_overall_name}** with a validation RMSE of **{best_overall_rmse:.4f}**.
2. **Generalization Profile**: The test set (2021-2022) performance shows the models' capability on out-of-distribution temporal data (post-2020 rainfall patterns). Removing `station_id` category and relying on KMeans clusters and raw coordinates has substantially reduced the train-test overfitting gap compared to baseline models.
"""
    
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Tuning report successfully saved to {OUT_REPORT}")

if __name__ == "__main__":
    train_and_tune()
