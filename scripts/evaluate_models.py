#!/usr/bin/env python3
"""Model Evaluation and Error Analysis script for NEERA.

Evaluates the best model on the test split, exports predictions, and performs
detailed error analysis across stations, seasons, years, rainfall levels,
and fallback routing types.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "outputs/predictions/test_split.csv"
MODEL_PKL = ROOT / "outputs/models/best_model.pkl"
OUT_PRED_CSV = ROOT / "outputs/predictions/test_predictions.csv"
OUT_REPORT = ROOT / "outputs/reports/error_analysis.txt"

def run_evaluation_and_analysis():
    print("Loading best model and test data...")
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    df_test = pd.read_csv(TEST_CSV)
    
    # Extract features matching the model
    if hasattr(model, "feature_name_"):
        features = model.feature_name_
    elif hasattr(model, "feature_names_"):
        features = model.feature_names_
    else:
        raise ValueError("Model does not have feature_name_ attribute.")

    target = "target_next_season_gw"

    # Cast categoricals to match model
    for col in features:
        if df_test[col].dtype == object or col in ["season", "rainfall_source_type", "mapping_method", "station_id"]:
            df_test[col] = df_test[col].astype(str).astype("category")

    X_test = df_test[features]
    y_test = df_test[target]

    print("Generating predictions on test set...")
    preds = model.predict(X_test)
    
    # Build predictions dataframe
    preds_df = pd.DataFrame({
        "station_id": df_test["station_id"],
        "timestamp": df_test["timestamp"],
        "date": df_test["date"],
        "actual": y_test,
        "predicted": preds,
        "error": y_test - preds,
        "abs_error": np.abs(y_test - preds),
        "squared_error": (y_test - preds) ** 2,
        "season": df_test["season"],
        "rainfall_fallback_used": df_test["rainfall_fallback_used"],
        "effective_rainfall_180d": df_test["effective_rainfall_180d"]
    })

    # Save predictions
    OUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(OUT_PRED_CSV, index=False)
    print(f"Predictions saved to {OUT_PRED_CSV}")

    # Perform Error Analysis
    report = []
    report.append("=========================================================")
    report.append("NEERA DETAILED ERROR ANALYSIS REPORT")
    report.append(f"Model: {type(model).__name__}")
    report.append(f"Generated at: {pd.Timestamp.now().isoformat()}")
    report.append("=========================================================\n")

    # 1. Overall Test Metrics
    mae = preds_df["abs_error"].mean()
    rmse = np.sqrt(preds_df["squared_error"].mean())
    mape = (preds_df["abs_error"] / preds_df["actual"].replace(0, np.nan)).dropna().mean() * 100
    report.append("Overall Test Set Metrics:")
    report.append(f"  - Mean Absolute Error (MAE): {mae:.4f} MBGL")
    report.append(f"  - Root Mean Squared Error (RMSE): {rmse:.4f} MBGL")
    report.append(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n")

    # 2. Worst Stations (Top 10 by MAE)
    station_errors = preds_df.groupby("station_id", as_index=False).agg(
        obs_count=("actual", "count"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(x.mean()))
    ).sort_values("mae", ascending=False)

    report.append("Top 10 Worst Performing Groundwater Stations by MAE:")
    report.append(station_errors.head(10).to_string(index=False))
    report.append("")

    # 3. Error by Year
    preds_df["year"] = pd.to_datetime(preds_df["date"]).dt.year
    year_errors = preds_df.groupby("year").agg(
        obs_count=("actual", "count"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(x.mean()))
    ).sort_values("year")

    report.append("Error Metrics by Year:")
    report.append(year_errors.to_string())
    report.append("")

    # 4. Seasonal Failure Modes
    season_errors = preds_df.groupby("season", observed=True).agg(
        obs_count=("actual", "count"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(x.mean()))
    ).sort_values("mae", ascending=False)

    report.append("Error Metrics by Season:")
    report.append(season_errors.to_string())
    report.append("")

    # 5. High Rainfall vs Normal Rainfall Failure Modes
    rain_p90 = preds_df["effective_rainfall_180d"].quantile(0.90)
    high_rain_df = preds_df[preds_df["effective_rainfall_180d"] >= rain_p90]
    normal_rain_df = preds_df[preds_df["effective_rainfall_180d"] < rain_p90]

    report.append("Rainfall Failure Mode Analysis:")
    report.append(f"  - High Rainfall Threshold (90th percentile): {rain_p90:.2f} mm")
    report.append(f"  - High Rainfall MAE (>= p90): {high_rain_df['abs_error'].mean():.4f} MBGL (obs count: {len(high_rain_df)})")
    report.append(f"  - Normal Rainfall MAE (< p90): {normal_rain_df['abs_error'].mean():.4f} MBGL (obs count: {len(normal_rain_df)})\n")

    # 6. Fallback Routing System Errors
    fallback_errors = preds_df.groupby("rainfall_fallback_used").agg(
        obs_count=("actual", "count"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(x.mean()))
    )

    report.append("Routing and Fallback Analysis:")
    report.append("  (rainfall_fallback_used: 1 = Fallback to district/state, 0 = Local telemetry used)")
    report.append(fallback_errors.to_string())

    # Write report
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"Error analysis report written to {OUT_REPORT}")

if __name__ == "__main__":
    run_evaluation_and_analysis()
