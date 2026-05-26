#!/usr/bin/env python3
"""Visualization script for NEERA.

Generates actual vs predicted scatter plots, residual distribution plots, and
station-wise prediction curves over time for a sample of good, average, and
worst-performing stations.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PREDS_CSV = ROOT / "outputs/predictions/test_predictions.csv"
OUT_PLOT_DIR = ROOT / "outputs/plots"

def create_visualizations():
    print("Loading test predictions...")
    df = pd.read_csv(PREDS_CSV)
    df["date"] = pd.to_datetime(df["date"])

    OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Actual vs Predicted Scatter Plot
    print("Generating actual vs predicted scatter plot...")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x="actual", y="predicted", alpha=0.6, color="teal")
    # Draw identity line
    min_val = min(df["actual"].min(), df["predicted"].min())
    max_val = max(df["actual"].max(), df["predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Identity Line")
    plt.title("Actual vs Predicted Groundwater Level (MBGL) on Test Set")
    plt.xlabel("Actual MBGL")
    plt.ylabel("Predicted MBGL")
    plt.legend()
    plt.tight_layout()
    scatter_path = OUT_PLOT_DIR / "actual_vs_predicted.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")

    # 2. Residual Distribution Plot
    print("Generating residual distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["error"], kde=True, color="purple", bins=50)
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    plt.title("Residual Distribution on Test Set (Actual - Predicted)")
    plt.xlabel("Residual (MBGL)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    residual_path = OUT_PLOT_DIR / "residual_distribution.png"
    plt.savefig(residual_path, dpi=150)
    plt.close()
    print(f"Saved residual plot to {residual_path}")

    # 3. Station-wise prediction curves over time
    print("Selecting sample stations for temporal curves...")
    # Find stations with test data
    station_mae = df.groupby("station_id")["abs_error"].mean().reset_index()
    # Sort them
    station_mae = station_mae.sort_values("abs_error").reset_index(drop=True)
    
    if len(station_mae) >= 3:
        best_station = station_mae.iloc[0]["station_id"]
        median_station = station_mae.iloc[len(station_mae)//2]["station_id"]
        worst_station = station_mae.iloc[-1]["station_id"]

        samples = {
            "best": (best_station, "Best Prediction (Lowest MAE)"),
            "median": (median_station, "Average Prediction (Median MAE)"),
            "worst": (worst_station, "Worst Prediction (Highest MAE)")
        }

        # Load train and val data for these stations to show full curves
        train_df = pd.read_csv(ROOT / "outputs/predictions/train_split.csv")
        val_df = pd.read_csv(ROOT / "outputs/predictions/val_split.csv")
        test_df = pd.read_csv(ROOT / "outputs/predictions/test_split.csv")
        
        train_df["date"] = pd.to_datetime(train_df["date"])
        val_df["date"] = pd.to_datetime(val_df["date"])
        test_df["date"] = pd.to_datetime(test_df["date"])

        for category, (stn, title) in samples.items():
            print(f"Plotting curve for {category} station: {stn}...")
            
            # Combine history
            st_train = train_df[train_df["station_id"] == stn][["date", "target_next_season_gw"]].rename(columns={"target_next_season_gw": "actual"})
            st_val = val_df[val_df["station_id"] == stn][["date", "target_next_season_gw"]].rename(columns={"target_next_season_gw": "actual"})
            st_test = df[df["station_id"] == stn][["date", "actual", "predicted"]]
            
            # Label splits
            st_train["split"] = "Train"
            st_val["split"] = "Val"
            st_test["split"] = "Test"
            
            # Combine
            full_history = pd.concat([st_train, st_val, st_test], ignore_index=True).sort_values("date")
            
            plt.figure(figsize=(12, 6))
            # Plot actual historical curve
            plt.plot(full_history["date"], full_history["actual"], marker="o", label="Actual MBGL", color="navy", linewidth=2)
            # Plot predicted points for test split
            st_test_sorted = st_test.sort_values("date")
            plt.plot(st_test_sorted["date"], st_test_sorted["predicted"], marker="x", linestyle="--", label="Predicted MBGL (Test)", color="orange", linewidth=2)
            
            # Add split transition lines
            if not st_train.empty:
                max_train_date = st_train["date"].max()
                plt.axvline(x=max_train_date, color="grey", linestyle=":", alpha=0.7)
                plt.text(max_train_date, plt.ylim()[0] + 2, "Val Split", rotation=90, verticalalignment="bottom", color="grey")
            
            plt.title(f"Groundwater Forecasting Curve for Station {stn} — {title}")
            plt.xlabel("Date")
            plt.ylabel("Next Season Groundwater Level (MBGL)")
            plt.legend()
            plt.tight_layout()
            curve_path = OUT_PLOT_DIR / f"station_curve_{category}.png"
            plt.savefig(curve_path, dpi=150)
            plt.close()
            print(f"Saved station curve to {curve_path}")

if __name__ == "__main__":
    create_visualizations()
