#!/usr/bin/env python3
"""Data Validation script for NEERA.

Validates the preprocessing pipeline outputs for leakage, temporal consistency,
ordering, duplicate keys, and out-of-bound variables.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "training_master.csv"
OUT_REPORT = ROOT / "outputs/reports/data_validation_report.txt"

def validate_dataset():
    print(f"Loading dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = pd.to_datetime(df["date"])

    issues = []
    checks = {}

    # 1. Row count & Columns
    checks["total_rows"] = len(df)
    checks["unique_stations"] = df["station_id"].nunique()
    print(f"Total Rows: {len(df)}, Unique Stations: {df['station_id'].nunique()}")

    # 2. Check for duplicate timestamps per station
    duplicates = df.duplicated(subset=["station_id", "timestamp"]).sum()
    checks["duplicate_timestamps"] = int(duplicates)
    if duplicates > 0:
        issues.append(f"CRITICAL: Found {duplicates} duplicate station-timestamp rows.")

    # 3. Check for broken temporal ordering
    non_monotonic = 0
    for st, g in df.groupby("station_id"):
        if not g["timestamp"].is_monotonic_increasing:
            non_monotonic += 1
    checks["non_monotonic_stations"] = non_monotonic
    if non_monotonic > 0:
        issues.append(f"CRITICAL: Found {non_monotonic} stations with unsorted/non-monotonic timestamps.")

    # 4. Check impossible groundwater values (0 to 150 MBGL)
    gw_col = "Groundwater_Level_MBGL"
    invalid_gw = df[(df[gw_col] < 0) | (df[gw_col] > 150)]
    checks["invalid_gw_values_count"] = len(invalid_gw)
    if len(invalid_gw) > 0:
        issues.append(f"WARNING: Found {len(invalid_gw)} rows with suspicious groundwater level (outside 0-150 MBGL).")

    # 5. Validate target shift & Leakage (target_next_season_gw should be next season's GW)
    leakage_failures = 0
    target_col = "target_next_season_gw"
    for st, g in df.groupby("station_id"):
        g = g.sort_values("timestamp")
        if len(g) < 2:
            continue
        targets = g[target_col].iloc[:-1].values
        next_gws = g[gw_col].iloc[1:].values
        if not np.allclose(targets, next_gws, equal_nan=True):
            leakage_failures += 1
    checks["target_leakage_failures"] = leakage_failures
    if leakage_failures > 0:
        issues.append(f"CRITICAL: Found target alignment mismatch in {leakage_failures} stations. Possible future leakage or wrong shifts!")

    # 6. Check rolling rainfall values (must be non-negative and < 5000mm)
    rain_features = ["effective_rainfall_30d", "effective_rainfall_90d", "effective_rainfall_180d"]
    invalid_rain_count = 0
    for col in rain_features:
        if col in df.columns:
            invalid_r = df[(df[col] < 0) | (df[col] > 5000)]
            invalid_rain_count += len(invalid_r)
    checks["invalid_rain_values_count"] = invalid_rain_count
    if invalid_rain_count > 0:
        issues.append(f"WARNING: Found {invalid_rain_count} rainfall rows with negative values or > 5000mm.")

    # 7. Check for duplicate dates per station
    duplicate_dates = df.duplicated(subset=["station_id", "date"]).sum()
    checks["duplicate_dates"] = int(duplicate_dates)
    if duplicate_dates > 0:
        issues.append(f"WARNING: Found {duplicate_dates} duplicate station-date rows.")

    # Create report string
    report = []
    report.append("=========================================================")
    report.append("NEERA DATA VALIDATION REPORT")
    report.append(f"Generated at: {pd.Timestamp.now().isoformat()}")
    report.append("=========================================================\n")
    report.append("Summary Statistics:")
    for k, v in checks.items():
        report.append(f"  - {k}: {v}")
    report.append("\nIssues & Warnings:")
    if not issues:
        report.append("  ✔ No data quality or future leakage issues detected.")
    else:
        for issue in issues:
            report.append(f"  - {issue}")

    # Write report
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"Data validation report written to {OUT_REPORT}")

    if any("CRITICAL" in issue for issue in issues):
        print("Data validation failed with CRITICAL issues!")
        sys.exit(1)
    else:
        print("Data validation passed successfully.")

if __name__ == "__main__":
    validate_dataset()
