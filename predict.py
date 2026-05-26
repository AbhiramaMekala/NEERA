#!/usr/bin/env python3
"""NEERA Groundwater Forecasting Hardened Inference Pipeline.

Performs schema validation, missing feature imputation, physical plausibility checks,
uncertainty interval estimation (P10/P50/P90), and logs predictions.
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs/models/best_model_validated.pkl"
Q10_PATH = ROOT / "outputs/models/model_q10.pkl"
Q50_PATH = ROOT / "outputs/models/model_q50.pkl"
Q90_PATH = ROOT / "outputs/models/model_q90.pkl"
DATA_PATH = ROOT / "data/training_master_engineered.csv"
LOG_PATH = ROOT / "outputs/predictions/inference_log.csv"

MODEL_VERSION = "2.0.0 (CatBoost Validated)"

def load_models():
    models = {}
    for name, path in [("cb", MODEL_PATH), ("q10", Q10_PATH), ("q50", Q50_PATH), ("q90", Q90_PATH)]:
        if not path.exists():
            print(f"Error: Required model file not found at {path}")
            sys.exit(1)
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models

def get_features_list(model):
    if hasattr(model, "feature_names_"):
        return list(model.feature_names_)
    return None

def impute_missing_features(df, features, database_df=None):
    df = df.copy()
    
    # 1. Determine baseline impute values
    impute_values = {}
    cat_cols = ["season", "rainfall_source_type", "mapping_method", "spatial_cluster"]
    
    if database_df is not None:
        if "year" not in database_df.columns and "date" in database_df.columns:
            database_df["year"] = pd.to_datetime(database_df["date"]).dt.year
        train_df = database_df[database_df["year"] <= 2019]
        for f in features:
            if f in cat_cols:
                impute_values[f] = train_df[f].mode().iloc[0] if not train_df[f].mode().empty else "unknown"
            else:
                col_numeric = pd.to_numeric(train_df[f], errors="coerce")
                impute_values[f] = col_numeric.median() if not col_numeric.isna().all() else 0.0
    else:
        impute_values = {
            "prev_gw": 8.2, "gw_diff": 0.0, "effective_rainfall_180d": 150.0,
            "season": "post_monsoon", "rainfall_source_type": "district",
            "mapping_method": "district_aggregate", "spatial_cluster": "0"
        }
        for f in features:
            if f not in impute_values:
                impute_values[f] = 0.0 if f not in cat_cols else "unknown"

    # 2. Impute and type-cast each feature
    for f in features:
        if f not in df.columns:
            df[f] = impute_values[f]
        else:
            if f not in cat_cols:
                df[f] = pd.to_numeric(df[f], errors="coerce")
            df[f] = df[f].fillna(impute_values[f])
            
    # 3. Final categorical type casting
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype("category")
            
    return df[features]

def run_physical_plausibility_checks(row, prediction):
    prev_gw = row.get("prev_gw", np.nan)
    rain_180 = row.get("effective_rainfall_180d", np.nan)
    
    warnings = []
    if pd.isna(prev_gw) or pd.isna(rain_180):
        return warnings
        
    # Rule A: Suspicious Recharge
    if (prediction < prev_gw - 2.0) and (rain_180 < 50.0):
        warnings.append("SUSPICIOUS_RECHARGE: Model predicts water table rise >2m despite extremely low rain (<50mm).")
    # Rule B: Suspicious Depletion
    if (prediction > prev_gw + 5.0) and (rain_180 > 500.0):
        warnings.append("SUSPICIOUS_DEPLETION: Model predicts water table drop >5m despite extremely high rain (>500mm).")
        
    return warnings

def log_prediction_history(station_id, forecast_date, p50, p10, p90, warnings):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    log_row = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_version": MODEL_VERSION,
        "station_id": station_id,
        "forecast_date": str(forecast_date),
        "prediction_p50": p50,
        "prediction_p10": p10,
        "prediction_p90": p90,
        "warnings": "; ".join(warnings) if warnings else "None"
    }
    
    log_df = pd.DataFrame([log_row])
    
    if LOG_PATH.exists():
        log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_PATH, index=False)

def predict_single_station(station_id, date_str=None):
    models = load_models()
    
    if not DATA_PATH.exists():
        print(f"Error: Database {DATA_PATH} not found. Cannot look up station history.")
        sys.exit(1)
        
    print(f"Loading station history from database...")
    db_df = pd.read_csv(DATA_PATH)
    db_df["date"] = pd.to_datetime(db_df["date"])
    
    # Filter station
    station_df = db_df[db_df["station_id"] == station_id]
    if len(station_df) == 0:
        print(f"Error: Station ID '{station_id}' not found in the database.")
        sys.exit(1)
        
    if date_str:
        try:
            target_date = pd.to_datetime(date_str)
        except Exception:
            print(f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD.")
            sys.exit(1)
        station_df["date_diff"] = (station_df["date"] - target_date).abs()
        row = station_df.sort_values(by="date_diff").iloc[0]
        actual_date = row["date"]
    else:
        row = station_df.sort_values(by="date", ascending=False).iloc[0]
        actual_date = row["date"]
        
    print(f"\n--- Station Metadata & Current State (Date: {actual_date.strftime('%Y-%m-%d')}) ---")
    print(f"Station ID:          {row['station_id']}")
    print(f"Coordinates:         ({row['latitude']:.4f}, {row['longitude']:.4f})")
    print(f"Spatial Cluster:     {row['spatial_cluster']}")
    print(f"Current GW Level:    {row['Groundwater_Level_MBGL']:.2f} MBGL")
    print(f"Prior GW Level:      {row['prev_gw']:.2f} MBGL" if not pd.isna(row['prev_gw']) else "Prior GW Level:      N/A")
    print(f"Rainfall (180d):     {row['effective_rainfall_180d']:.1f} mm")
    
    features = get_features_list(models["cb"])
    input_row = pd.DataFrame([row])
    X = impute_missing_features(input_row, features, db_df)
    
    # Predict
    pred_val = float(models["cb"].predict(X)[0])
    
    # Predict Uncertainty Intervals (P10, P90)
    p10 = float(models["q10"].predict(X)[0])
    p90 = float(models["q90"].predict(X)[0])
    p10 = min(p10, pred_val)
    p90 = max(p90, pred_val)
    
    # Run physical sanity checks
    warnings_list = run_physical_plausibility_checks(row, pred_val)
    
    # Log prediction
    log_prediction_history(row["station_id"], actual_date.strftime("%Y-%m-%d"), pred_val, p10, p90, warnings_list)
    
    # Output to console
    print(f"\n=========================================================")
    print(f"NEERA UNCERTAINTY-AWARE FORECAST (Version {MODEL_VERSION})")
    print(f"=========================================================")
    print(f"Predicted Level (P50):    {pred_val:.2f} MBGL")
    print(f"80% Confidence Interval:  [{p10:.2f} MBGL to {p90:.2f} MBGL]")
    print(f"Interval Uncertainty:     {p90 - p10:.2f} meters")
    
    actual_next = row.get("target_next_season_gw", np.nan)
    if not pd.isna(actual_next):
        print(f"Actual Next Level:        {actual_next:.2f} MBGL")
        print(f"Forecast Error (P50):     {pred_val - actual_next:.2f} meters")
        
    print(f"=========================================================")
    if warnings_list:
        print("ALERT: Hydrological Plausibility Warnings:")
        for w in warnings_list:
            print(f"  - [WARNING] {w}")
        print(f"=========================================================")
    print()

def predict_batch(batch_csv, output_csv):
    models = load_models()
    
    print(f"Loading batch data from {batch_csv}...")
    try:
        df = pd.read_csv(batch_csv)
    except Exception as e:
        print(f"Error reading batch CSV: {e}")
        sys.exit(1)
        
    # Load database statistics for imputation if present
    db_df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    
    features = get_features_list(models["cb"])
    X = impute_missing_features(df, features, db_df)
    
    print("Running batch predictions and interval estimations...")
    preds = models["cb"].predict(X)
    
    # Quantiles
    p10_arr = models["q10"].predict(X)
    p90_arr = models["q90"].predict(X)
    p10_arr = np.minimum(p10_arr, preds)
    p90_arr = np.maximum(p90_arr, preds)
    
    # Assemble outputs
    output_df = df.copy()
    output_df["predicted_p50_mbgl"] = preds
    output_df["predicted_p10_mbgl"] = p10_arr
    output_df["predicted_p90_mbgl"] = p90_arr
    
    # Run physical sanity alerts row-by-row
    plausibility_warnings = []
    for idx, row in df.iterrows():
        warns = run_physical_plausibility_checks(row, preds[idx])
        plausibility_warnings.append("; ".join(warns) if warns else "OK")
    output_df["physical_plausibility_warnings"] = plausibility_warnings
    
    # Export
    output_df.to_csv(output_csv, index=False)
    print(f"Success! Batch predictions exported to {output_csv}")
    
    # Log batch run (summary)
    log_prediction_history(
        "BATCH_INFERENCE",
        f"Batch of {len(df)} samples",
        float(np.mean(preds)),
        float(np.mean(p10_arr)),
        float(np.mean(p90_arr)),
        ["Batch run logged successfully"]
    )

def main():
    parser = argparse.ArgumentParser(
        description="NEERA Hardened Groundwater Level Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Station Mode
    station_parser = subparsers.add_parser("station", help="Forecast next season for a specific station")
    station_parser.add_argument("--station_id", required=True, type=str, help="Station ID (e.g. 020109B)")
    station_parser.add_argument("--date", type=str, default=None, help="Specific observation date to forecast from (YYYY-MM-DD)")
    
    # Batch Mode
    batch_parser = subparsers.add_parser("batch", help="Batch forecast on an input CSV file")
    batch_parser.add_argument("--input_csv", required=True, type=str, help="Path to input CSV containing features")
    batch_parser.add_argument("--output_csv", required=True, type=str, help="Path to write the CSV with predictions")
    
    args = parser.parse_args()
    
    if args.command == "station":
        predict_single_station(args.station_id, args.date)
    elif args.command == "batch":
        predict_batch(args.input_csv, args.output_csv)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
