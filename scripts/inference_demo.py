#!/usr/bin/env python3
"""Interactive Inference Demo script for NEERA.

Loads the best serialized model and runs forecasting on specified groundwater stations
or randomly selected stations from the test set, displaying feature values and predictions.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PKL = ROOT / "outputs/models/best_model.pkl"
TEST_CSV = ROOT / "outputs/predictions/test_split.csv"

def run_demo(station_id=None):
    # Load model
    print(f"Loading best model from {MODEL_PKL}...")
    if not MODEL_PKL.exists():
        print(f"Error: Model not found at {MODEL_PKL}. Please run train_baseline.py first.")
        return
        
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    # Load test split
    print(f"Loading test split data from {TEST_CSV}...")
    if not TEST_CSV.exists():
        print(f"Error: Test data not found at {TEST_CSV}. Please run train_baseline.py first.")
        return
        
    df_test = pd.read_csv(TEST_CSV)

    # Identify features
    if hasattr(model, "feature_name_"):
        features = model.feature_name_
    elif hasattr(model, "feature_names_"):
        features = model.feature_names_
    else:
        raise ValueError("Model does not have feature_name_ attribute.")

    target = "target_next_season_gw"

    # Select station
    if station_id:
        station_df = df_test[df_test["station_id"].astype(str) == str(station_id)]
        if station_df.empty:
            print(f"Station {station_id} not found in the test set. Available sample station IDs:")
            print(df_test["station_id"].unique()[:10])
            return
    else:
        # Pick a random station from test set
        station_id = np.random.choice(df_test["station_id"].unique())
        station_df = df_test[df_test["station_id"] == station_id]
        print(f"No station ID specified. Randomly selected station: {station_id}")

    # Prepare features
    station_df = station_df.sort_values("timestamp")
    
    # Cast categories
    for col in features:
        if station_df[col].dtype == object or col in ["season", "rainfall_source_type", "mapping_method", "station_id"]:
            station_df[col] = station_df[col].astype(str).astype("category")

    X_station = station_df[features]
    y_actual = station_df[target]

    # Predict
    preds = model.predict(X_station)

    # Print results
    print("\n" + "=" * 60)
    print(f"NEERA GROUNDWATER INFERENCE RUN FOR STATION: {station_id}")
    print("=" * 60)
    
    # Let's print each observation date, actual, and predicted
    for idx, (index, row) in enumerate(station_df.iterrows()):
        date_str = row["date"]
        actual_val = y_actual.iloc[idx]
        pred_val = preds[idx]
        err = actual_val - pred_val
        fallback = "Yes" if row["rainfall_fallback_used"] == 1.0 else "No (Local telemetry)"
        dist = f"{row['rainfall_distance_km']:.1f}km" if pd.notna(row["rainfall_distance_km"]) else "N/A"
        
        print(f"\nDate: {date_str} (Season: {row['season']})")
        print(f"  - Inputs:")
        print(f"    - Current Groundwater: {row['Groundwater_Level_MBGL']:.2f} MBGL")
        print(f"    - Prev Season Groundwater: {row['prev_gw']:.2f} MBGL")
        print(f"    - Cumulative 180d effective rain: {row['effective_rainfall_180d']:.2f} mm")
        print(f"    - Rain station distance: {dist} (Fallback used: {fallback})")
        print(f"  - Outputs:")
        print(f"    - Actual Next Season Groundwater: {actual_val:.2f} MBGL")
        print(f"    - Predicted Next Season Groundwater: {pred_val:.2f} MBGL")
        print(f"    - Prediction Error: {err:.2f} MBGL")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEERA Groundwater Forecasting Inference Demo")
    parser.add_argument("--station_id", type=str, help="Groundwater station ID to run inference for.")
    args = parser.parse_args()
    
    run_demo(args.station_id)
