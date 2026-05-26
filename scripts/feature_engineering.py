#!/usr/bin/env python3
"""Feature Engineering script for NEERA.

Autonomously engineers temporal, rainfall, spatial, and quality features,
verifies that no target leakage occurs, and outputs the engineered dataset.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "training_master.csv"
MASTER_CSV = ROOT / "master_dataset.csv"
OUTPUT_CSV = ROOT / "data/training_master_engineered.csv"

def engineer_features():
    print(f"Loading datasets...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Sort by station and timestamp
    df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    
    # Load coordinates from master_dataset.csv
    print("Extracting coordinates from master_dataset.csv...")
    df_master = pd.read_csv(MASTER_CSV, low_memory=False)
    coords = df_master.dropna(subset=["station_id", "latitude", "longitude"]).groupby("station_id")[["latitude", "longitude"]].median().reset_index()
    
    # Merge coordinates
    df = df.merge(coords, on="station_id", how="left")
    
    # Impute missing coordinates (if any, though audit showed 0 missing)
    if df["latitude"].isna().any():
        print("Warning: some stations are missing coordinates, imputing with state medians...")
        df["latitude"] = df["latitude"].fillna(12.97) # Bangalore Lat
        df["longitude"] = df["longitude"].fillna(77.59) # Bangalore Lon
        
    print(f"Coordinates merged. Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    
    # ── 1. Temporal features ─────────────────────────────────────────────────
    print("Generating temporal features...")
    
    # Lags (1 is already prev_gw, but let's make sure it matches lag_1)
    df["lag_1"] = df.groupby("station_id")["Groundwater_Level_MBGL"].shift(1)
    df["lag_2"] = df.groupby("station_id")["Groundwater_Level_MBGL"].shift(2)
    df["lag_3"] = df.groupby("station_id")["Groundwater_Level_MBGL"].shift(3)
    df["lag_4"] = df.groupby("station_id")["Groundwater_Level_MBGL"].shift(4)
    
    # Verify that lag_1 matches prev_gw (ignoring first row of each station which might be NaN)
    diff_count = ((df["lag_1"] - df["prev_gw"]).abs() > 1e-4).sum()
    print(f"  - lag_1 mismatch count with prev_gw: {diff_count}")
    
    # Expanding stats: shift(1) first to avoid leakage!
    # df.groupby('station_id')['Groundwater_Level_MBGL'].shift(1) returns the causal series of observations
    def compute_causal_expanding(series_grp):
        # Shift to be causal, then expand
        shifted = series_grp.shift(1)
        # Pandas expanding requires min_periods=1
        exp_mean = shifted.expanding(min_periods=1).mean()
        exp_std = shifted.expanding(min_periods=2).std() # Std needs at least 2 points
        return pd.DataFrame({"exp_mean": exp_mean, "exp_std": exp_std}, index=series_grp.index)
        
    expanding_df = df.groupby("station_id")["Groundwater_Level_MBGL"].apply(compute_causal_expanding).reset_index(level=0, drop=True)
    df["gw_expanding_mean"] = expanding_df["exp_mean"]
    df["gw_expanding_std"] = expanding_df["exp_std"]
    
    # Exponentially weighted averages: shift(1) first to avoid leakage!
    def compute_causal_ewma(series_grp):
        shifted = series_grp.shift(1)
        ewm_3 = shifted.ewm(span=3, min_periods=1).mean()
        ewm_5 = shifted.ewm(span=5, min_periods=1).mean()
        return pd.DataFrame({"ewm_3": ewm_3, "ewm_5": ewm_5}, index=series_grp.index)
        
    ewma_df = df.groupby("station_id")["Groundwater_Level_MBGL"].apply(compute_causal_ewma).reset_index(level=0, drop=True)
    df["gw_ewm_mean_span3"] = ewma_df["ewm_3"]
    df["gw_ewm_mean_span5"] = ewma_df["ewm_5"]
    
    # Seasonal delta
    df["gw_seasonal_diff_2"] = df["lag_1"] - df["lag_2"]
    
    # Year-over-Year (YoY) Change (same season, previous year)
    # Group by station_id and season, sort by timestamp, then shift by 1
    df = df.sort_values(by=["station_id", "season", "timestamp"])
    df["gw_same_season_prev_obs"] = df.groupby(["station_id", "season"])["Groundwater_Level_MBGL"].shift(1)
    df["gw_yoy_diff"] = df["Groundwater_Level_MBGL"] - df["gw_same_season_prev_obs"]
    
    # Resort back to station_id and timestamp
    df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    
    # ── 2. Rainfall features ─────────────────────────────────────────────────
    print("Generating rainfall features...")
    # Rainfall intensity
    df["rainfall_intensity_30d"] = df["effective_rainfall_30d"] / 30.0
    df["rainfall_intensity_90d"] = df["effective_rainfall_90d"] / 90.0
    df["rainfall_intensity_180d"] = df["effective_rainfall_180d"] / 180.0
    
    # Rainfall anomalies: difference from district / state rainfall
    for w in [30, 90, 180]:
        eff_rain = f"effective_rainfall_{w}d"
        dist_rain = f"district_rainfall_{w}d"
        state_rain = f"state_rainfall_{w}d"
        
        # Fill missing district with state
        baseline_rain = df[dist_rain].fillna(df[state_rain])
        df[f"rainfall_anomaly_{w}d"] = df[eff_rain] - baseline_rain
        
    # Rainfall ratios
    df["rainfall_ratio_30d_180d"] = df["effective_rainfall_30d"] / (df["effective_rainfall_180d"] + 1e-5)
    df["rainfall_ratio_90d_180d"] = df["effective_rainfall_90d"] / (df["effective_rainfall_180d"] + 1e-5)
    
    # Drought indicators: simple threshold of 180d rainfall being low
    # E.g. bottom 15th percentile of 180d effective rainfall in training set is around 100mm. Let's make it continuous or binary
    r180_threshold = df.loc[df["date"].dt.year <= 2019, "effective_rainfall_180d"].quantile(0.15)
    if pd.isna(r180_threshold):
        r180_threshold = 100.0
    df["drought_indicator_180d"] = (df["effective_rainfall_180d"] < r180_threshold).astype(float)
    
    # Recharge efficiency proxies (change in GW level divided by rainfall in the season)
    # If water table rose (MBGL decreased), gw_diff is negative.
    # Recharge efficiency proxy = -gw_diff / (effective_rainfall_180d + 1e-5)
    df["recharge_efficiency_proxy_90d"] = -df["gw_diff"] / (df["effective_rainfall_90d"] + 1e-5)
    df["recharge_efficiency_proxy_180d"] = -df["gw_diff"] / (df["effective_rainfall_180d"] + 1e-5)
    
    # Rolling trend slopes (difference in rainfall over windows)
    df["rainfall_trend_slope"] = (df["effective_rainfall_30d"] - (df["effective_rainfall_90d"] - df["effective_rainfall_30d"])/2.0)
    
    # ── 3. Spatial features ──────────────────────────────────────────────────
    print("Generating spatial features...")
    # Fit KMeans cluster on training set coordinates only to prevent leakage
    train_coords = df.loc[df["date"].dt.year <= 2019, ["latitude", "longitude"]].drop_duplicates()
    n_clusters = 15
    print(f"  - Fitting KMeans with {n_clusters} clusters on train coordinates (size={len(train_coords)})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(train_coords)
    
    # Predict clusters for all stations
    df["spatial_cluster"] = kmeans.predict(df[["latitude", "longitude"]])
    # Convert cluster to string category so trees treat it correctly
    df["spatial_cluster"] = df["spatial_cluster"].astype(str).astype("category")
    
    # ── 4. Quality Features ──────────────────────────────────────────────────
    print("Generating quality features...")
    # Source reliability score
    reliability_map = {"local": 3.0, "district": 2.0, "state": 1.0}
    df["source_reliability_score"] = df["rainfall_source_type"].map(reliability_map).fillna(0.0)
    
    # ── 5. Zero Target Leakage Verification ──────────────────────────────────
    print("Verifying leakage constraints...")
    # Target should only be target_next_season_gw
    target_col = "target_next_season_gw"
    
    # Verify that target is indeed shift(-1) of Groundwater_Level_MBGL per station
    leakage_failures = 0
    for st, g in df.groupby("station_id"):
        g = g.sort_values("timestamp")
        if len(g) < 2:
            continue
        targets = g[target_col].iloc[:-1].values
        next_gws = g["Groundwater_Level_MBGL"].iloc[1:].values
        if not np.allclose(targets, next_gws, equal_nan=True):
            leakage_failures += 1
            
    if leakage_failures > 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: target_next_season_gw mismatch in {leakage_failures} stations!")
    print("  ✔ Shift validation: target_next_season_gw is exactly shift(-1) of Groundwater_Level_MBGL.")
    
    # Check if any new feature correlates exactly 1.0 with the target (which is leakage)
    engineered_cols = [c for c in df.columns if c not in ["station_id", "timestamp", "date", "season", "freq", "rainfall_source_type", "rainfall_source_station", "mapping_method", target_col, "spatial_cluster", "gw_same_season_prev_obs"]]
    num_df = df[engineered_cols].select_dtypes(include=[np.number])
    corrs = num_df.corrwith(df[target_col]).abs()
    perfect_corrs = corrs[corrs > 0.999]
    if len(perfect_corrs) > 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: Features correlate perfectly (>0.999) with target: {perfect_corrs.index.tolist()}")
    print("  ✔ Correlation validation: No engineered feature has a Pearson correlation > 0.999 with the target.")
    
    # Save the engineered dataset
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Engineered dataset successfully saved to {OUTPUT_CSV} (shape: {df.shape})")

if __name__ == "__main__":
    engineer_features()
