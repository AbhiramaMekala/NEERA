#!/usr/bin/env python3
"""Baseline Model Training script for NEERA.

Trains CatBoost, LightGBM, and XGBoost regressors on the preprocessed dataset
using a strict temporal split. Evaluates performance metrics and serializes the
best model.
"""

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "training_master.csv"
OUT_MODEL_DIR = ROOT / "outputs/models"
OUT_METRICS_DIR = ROOT / "outputs/metrics"
OUT_PRED_DIR = ROOT / "outputs/predictions"
OUT_PLOTS_DIR = ROOT / "outputs/plots"

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_evaluate():
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # 1. Define Features & Target
    numeric_features = [
        "prev_gw", "gw_diff", "gw_roll_mean_7obs", "gw_roll_std_7obs", "gw_roll_mean_30obs",
        "effective_rainfall_30d", "effective_rainfall_90d", "effective_rainfall_180d",
        "rainfall_window_completeness_30d", "rainfall_window_completeness_90d", "rainfall_window_completeness_180d",
        "rainfall_distance_km", "season_sin", "season_cos"
    ]
    categorical_features = ["season", "rainfall_source_type", "mapping_method", "station_id"]
    target = "target_next_season_gw"

    # Convert binary indicator to numeric
    if "rainfall_fallback_used" in df.columns:
        df["rainfall_fallback_used"] = df["rainfall_fallback_used"].astype(float)
        numeric_features.append("rainfall_fallback_used")

    # Clean missing targets
    df = df.dropna(subset=[target]).copy()

    # Pre-process categories
    for col in categorical_features:
        df[col] = df[col].astype(str).astype("category")

    features = numeric_features + categorical_features
    print(f"Training features: {features}")
    print(f"Target: {target}")

    # 2. Strict Temporal Split
    train_df = df[df["year"] <= 2019].copy()
    val_df = df[df["year"] == 2020].copy()
    test_df = df[df["year"] >= 2021].copy()

    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # Save splits for downstream usage
    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_PRED_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUT_PRED_DIR / "val_split.csv", index=False)
    test_df.to_csv(OUT_PRED_DIR / "test_split.csv", index=False)

    # 3. Fit Models
    models = {}
    metrics_list = []

    # A. LightGBM
    print("\n--- Training LightGBM ---")
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=4,
        verbose=-1
    )
    lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[] # We can add early stopping round via parameters if needed or let it run fully as baseline
    )
    models["LightGBM"] = lgb

    # B. XGBoost
    print("\n--- Training XGBoost ---")
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=4,
        enable_categorical=True
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    models["XGBoost"] = xgb

    # C. CatBoost
    print("\n--- Training CatBoost ---")
    # For CatBoost, convert category dtypes back to string or list index of categories
    cat_indices = [features.index(col) for col in categorical_features]
    cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=100,
        thread_count=4,
        cat_features=cat_indices
    )
    cb.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100
    )
    models["CatBoost"] = cb

    # 4. Evaluate Models
    best_model_name = None
    best_val_rmse = float("inf")

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        for split_name, (X, y) in {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}.items():
            # Handle predictions
            preds = model.predict(X)
            
            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)
            mape = mean_absolute_percentage_error(y, preds)

            metrics_list.append({
                "model": name,
                "split": split_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape
            })

            print(f"  {split_name.capitalize()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")

            if split_name == "val" and rmse < best_val_rmse:
                best_val_rmse = rmse
                best_model_name = name

    # 5. Export Metrics
    metrics_df = pd.DataFrame(metrics_list)
    OUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_METRICS_DIR / "model_metrics.csv", index=False)
    print(f"\nModel metrics saved to {OUT_METRICS_DIR / 'model_metrics.csv'}")

    # 6. Save Best Model
    best_model = models[best_model_name]
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model ({best_model_name}) serialized to {OUT_MODEL_DIR / 'best_model.pkl'}")

    # Write a quick text model comparison report
    with open(ROOT / "outputs/reports/model_comparison.txt", "w", encoding="utf-8") as f:
        f.write("=========================================================\n")
        f.write("NEERA MODEL COMPARISON SUMMARY\n")
        f.write(f"Generated at: {pd.Timestamp.now().isoformat()}\n")
        f.write("=========================================================\n\n")
        f.write(f"Best Model Selected: {best_model_name} (Val RMSE: {best_val_rmse:.4f})\n\n")
        f.write(metrics_df.to_string(index=False) + "\n")

if __name__ == "__main__":
    train_and_evaluate()
