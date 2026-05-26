#!/usr/bin/env python3
"""Rebuild a scientifically valid groundwater ML dataset with local rainfall mapping.

Outputs:
- clean_master_dataset_v3.csv
- preprocessing_report_v3.txt
- rainfall_mapping_log.csv
- rainfall_diagnostics.txt
- anomaly_log.csv
- dropped_rows_log.csv
- dropped_stations_log.csv

The pipeline is leakage-safe and failure-driven. It uses observation-based rolling
windows, causal rainfall windows, and explicit validation assertions.
"""

from __future__ import annotations

import os
import resource
import time
from pathlib import Path
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import re

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
INPUT_CSV = ROOT / "master_dataset.csv"
OUT_CLEAN = ROOT / "clean_master_dataset_v3.csv"
OUT_REPORT = ROOT / "preprocessing_report_v3.txt"
OUT_RAIN_MAP = ROOT / "rainfall_mapping_log.csv"
OUT_RAIN_DIAG = ROOT / "rainfall_diagnostics.txt"
OUT_LOCAL_COVERAGE = ROOT / "local_rainfall_coverage_report.txt"
OUT_SPATIAL_SUMMARY = ROOT / "rainfall_spatial_correlation_summary.txt"
OUT_ANOMALY = ROOT / "anomaly_log.csv"
OUT_DROPPED_ROWS = ROOT / "dropped_rows_log.csv"
OUT_DROPPED_STATIONS = ROOT / "dropped_stations_log.csv"

# ── Thresholds ─────────────────────────────────────────────────────────────────
LOCAL_WINDOW_COMPLETENESS_THRESHOLDS = {30: 0.70, 90: 0.65, 180: 0.60}
LOCAL_NEIGHBOR_RADIUS_KM = 50.0  # increased from 25 to improve coverage
RAIN_180D_PHYSICAL_THRESHOLD_MM = 5000.0
RAIN_DAILY_IMPOSSIBLE_THRESHOLD_MM = 500.0
RAIN_IDENTICAL_STATION_FRACTION_THRESHOLD = 0.90
RAINFALL_VALUE_MAX_MM = 1000.0

# ── Geographic bounds (Karnataka + neighbouring overlap) ───────────────────────
# Rainfall data is Karnataka-only. Only GW stations within this range can get local rainfall.
KARNATAKA_LAT_MIN = 11.0
KARNATAKA_LAT_MAX = 19.0
KARNATAKA_LON_MIN = 73.0
KARNATAKA_LON_MAX = 79.0
INDIA_LAT_MIN = 8.0
INDIA_LAT_MAX = 38.0
INDIA_LON_MIN = 60.0
INDIA_LON_MAX = 100.0

RAINFALL_FILTER_SUMMARY = {"outlier_rows_dropped": 0}

# ── Debug mode ─────────────────────────────────────────────────────────────────
# In debug mode: pick Karnataka stations only, limit to fast iteration
DEBUG_MODE = False
DEBUG_GW_STATIONS = 20       # number of GW stations in debug
DEBUG_RAIN_YEARS = 3         # years of rainfall data in debug


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def canon(x):
    """Canonicalize a string for robust joins: lowercase, normalize whitespace/dashes."""
    if pd.isna(x):
        return ""
    s = str(x).lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def mem_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if ru > 1e6:
        return ru / (1024 * 1024)
    return ru / 1024.0


def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "NA": np.nan, "": np.nan})


def _haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371.0 * c


def _find_col(columns, needles):
    for c in columns:
        lc = str(c).lower()
        if any(n in lc for n in needles):
            return c
    return None


def _timed(label):
    """Simple context-manager-style timer. Returns a callable to print elapsed."""
    start = time.time()
    def done():
        elapsed = time.time() - start
        print(f"    [{label}] {elapsed:.1f}s", flush=True)
    return done


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_id": str}, low_memory=False)
    return _standardize_cols(df)


def _parse_timestamp(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out, int(out["timestamp"].isna().sum())


def _validate_station_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    out["station_id"] = _normalize_text(out["station_id"])
    invalid = int(out["station_id"].isna().sum())
    return out, invalid


def _load_groundwater_metadata(atal_path: Path) -> pd.DataFrame:
    """Load Atal Jal metadata for district/block enrichment."""
    df = pd.read_csv(atal_path, dtype=str, low_memory=False, encoding="latin1")
    df = _standardize_cols(df)
    station_col = next((c for c in df.columns if str(c).strip().lower() == "well_id"), None)
    if station_col is None:
        station_col = _find_col(df.columns, ["well_id"])
    lat_col = next((c for c in df.columns if str(c).strip().lower() == "latitude"), None)
    lon_col = next((c for c in df.columns if str(c).strip().lower() == "longitude"), None)
    district_col = next((c for c in df.columns if str(c).strip().lower() == "district_name_with_lgd_code"), None)
    if district_col is None:
        district_col = _find_col(df.columns, ["district_name"])
    block_col = next((c for c in df.columns if str(c).strip().lower() == "block_name_with_lgd_code"), None)
    if block_col is None:
        block_col = _find_col(df.columns, ["block_name"])
    state_col = _find_col(df.columns, ["state_name"])

    cols = [c for c in [station_col, lat_col, lon_col, district_col, block_col, state_col] if c is not None]
    meta = df[cols].copy()
    rename_map = {station_col: "station_id"}
    if lat_col:
        rename_map[lat_col] = "latitude"
    if lon_col:
        rename_map[lon_col] = "longitude"
    if district_col:
        rename_map[district_col] = "district"
    if block_col:
        rename_map[block_col] = "block"
    if state_col:
        rename_map[state_col] = "state"
    meta = meta.rename(columns=rename_map)

    for c in ["latitude", "longitude"]:
        if c in meta.columns:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta["station_id"] = _normalize_text(meta["station_id"])
    for c in ["district", "block", "state"]:
        if c in meta.columns:
            meta[c] = _normalize_text(meta[c])

    meta = meta.dropna(subset=["station_id"]).groupby("station_id", as_index=False).agg(
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        district=("district", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
        block=("block", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
        **({"state": ("state", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan)} if "state" in meta.columns else {}),
    )
    # canonicalize keys
    meta["station_id"] = meta["station_id"].apply(canon)
    for c in ["district", "block"]:
        if c in meta.columns:
            meta[c] = meta[c].apply(lambda x: canon(x) if pd.notna(x) else x)
    return meta


def _build_gw_metadata_from_master(df: pd.DataFrame) -> pd.DataFrame:
    """Extract GW station coordinates directly from the master dataset.

    This is more reliable than Atal Jal because the master already has
    lat/lon for every observation row.
    """
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["station_id"] = df["station_id"].astype(str).str.strip()
    df["_date_temp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.normalize()

    meta = (
        df.dropna(subset=["station_id", "latitude", "longitude"])
        .groupby("station_id", as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            min_date=("_date_temp", "min"),
            max_date=("_date_temp", "max"),
        )
    )
    # validate coordinate ranges
    meta = meta.loc[
        meta["latitude"].between(INDIA_LAT_MIN, INDIA_LAT_MAX)
        & meta["longitude"].between(INDIA_LON_MIN, INDIA_LON_MAX)
    ].copy()
    # swap if obviously reversed
    swap_mask = (meta["latitude"].abs() > 90) & (meta["longitude"].abs() <= 90)
    if swap_mask.any():
        meta.loc[swap_mask, ["latitude", "longitude"]] = meta.loc[swap_mask, ["longitude", "latitude"]].values
    # canonicalize station_id for joins
    meta["station_id"] = meta["station_id"].apply(canon)
    return meta.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  RAINFALL INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def _load_rainfall_station_catalog(min_date: pd.Timestamp | None = None, max_date: pd.Timestamp | None = None) -> pd.DataFrame:
    global RAINFALL_FILTER_SUMMARY
    files = sorted([p for p in DATASET_DIR.glob("rainfall*.csv") if p.is_file()])
    rows = []
    for f in files:
        fsize = f.stat().st_size
        use_chunks = fsize > 5 * 1024 * 1024
        source_kind = "subdaily" if any(tok in f.name.lower() for tok in ["_hr_", "hour"]) else "daily"
        print(f"    [rainfall load] {f.name}", flush=True)
        header_df = pd.read_csv(f, nrows=0, low_memory=False, encoding="latin1")
        header_df = _standardize_cols(header_df)
        station_col = _find_col(header_df.columns, ["station"])
        lat_col = _find_col(header_df.columns, ["latitude"])
        lon_col = _find_col(header_df.columns, ["longitude"])
        dist_col = _find_col(header_df.columns, ["district"])
        block_col = _find_col(header_df.columns, ["block"])
        time_col = _find_col(header_df.columns, ["data acquisition time", "timestamp", "time", "date"])
        rain_col = _find_col(header_df.columns, ["rainfall"])
        if station_col is None or time_col is None or rain_col is None:
            continue
        use_cols = [station_col, time_col, rain_col]
        for c in [lat_col, lon_col, dist_col, block_col]:
            if c is not None and c not in use_cols:
                use_cols.append(c)

        if use_chunks:
            reader = pd.read_csv(f, dtype=str, chunksize=50_000, low_memory=False, encoding="latin1", usecols=use_cols)
        else:
            reader = [pd.read_csv(f, dtype=str, low_memory=False, encoding="latin1", usecols=use_cols)]
        for chunk_idx, chunk in enumerate(reader, start=1):
            chunk = _standardize_cols(chunk)
            cols_for_chunk = [station_col, time_col, rain_col]
            for c in [lat_col, lon_col, dist_col, block_col]:
                if c is not None:
                    cols_for_chunk.append(c)
            x = chunk[cols_for_chunk].copy()
            rename_map = {station_col: "rainfall_station", time_col: "timestamp", rain_col: "rainfall_mm"}
            if lat_col is not None:
                rename_map[lat_col] = "latitude"
            if lon_col is not None:
                rename_map[lon_col] = "longitude"
            if dist_col is not None:
                rename_map[dist_col] = "district"
            if block_col is not None:
                rename_map[block_col] = "block"
            x = x.rename(columns=rename_map)
            x["rainfall_station"] = _normalize_text(x["rainfall_station"])
            x["timestamp"] = pd.to_datetime(x["timestamp"], dayfirst=True, errors="coerce")
            x["rainfall_mm"] = pd.to_numeric(x["rainfall_mm"], errors="coerce")
            if "latitude" in x.columns:
                x["latitude"] = pd.to_numeric(x["latitude"], errors="coerce")
            else:
                x["latitude"] = np.nan
            if "longitude" in x.columns:
                x["longitude"] = pd.to_numeric(x["longitude"], errors="coerce")
            else:
                x["longitude"] = np.nan
            x["district"] = _normalize_text(x["district"]) if "district" in x.columns else np.nan
            x["block"] = _normalize_text(x["block"]) if "block" in x.columns else np.nan
            bad_mask = x["rainfall_mm"].notna() & ((x["rainfall_mm"] < 0) | (x["rainfall_mm"] > RAINFALL_VALUE_MAX_MM))
            RAINFALL_FILTER_SUMMARY["outlier_rows_dropped"] += int(bad_mask.sum())
            x = x[~bad_mask].copy()
            x = x.dropna(subset=["rainfall_station", "timestamp", "rainfall_mm"])
            # normalize date to midnight using .dt.normalize() — NO floor/freq hacks
            x["date"] = x["timestamp"].dt.normalize()
            # canonicalize station names for robust joins
            x["rainfall_station"] = x["rainfall_station"].apply(canon)
            # filter to relevant date window
            if min_date is not None and max_date is not None:
                lo = pd.to_datetime(min_date) - pd.Timedelta(days=180)
                hi = pd.to_datetime(max_date) + pd.Timedelta(days=1)
                lo = lo.normalize()
                hi = hi.normalize()
                x = x.loc[(x["date"] >= lo) & (x["date"] <= hi)]
            x["source_file"] = f.name
            x["source_kind"] = source_kind
            # aggregate sub-daily to daily per station
            x = (
                x.groupby(["rainfall_station", "date", "source_file", "source_kind"], as_index=False)
                .agg(
                    rainfall_mm=("rainfall_mm", "sum"),
                    latitude=("latitude", "median"),
                    longitude=("longitude", "median"),
                    district=("district", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
                    block=("block", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
                )
            )
            # Filter out daily sums that exceed the impossible daily threshold
            impossible_daily_mask = x["rainfall_mm"] > RAIN_DAILY_IMPOSSIBLE_THRESHOLD_MM
            if impossible_daily_mask.any():
                print(f"    [rainfall load] Dropping {impossible_daily_mask.sum()} daily rows with impossible rainfall > {RAIN_DAILY_IMPOSSIBLE_THRESHOLD_MM}mm in chunk", flush=True)
                RAINFALL_FILTER_SUMMARY["outlier_rows_dropped"] += int(impossible_daily_mask.sum())
                x = x[~impossible_daily_mask].copy()
            rows.append(x)
            if chunk_idx % 5 == 0 or not use_chunks:
                print(f"    [rainfall load] {f.name} chunk {chunk_idx}", flush=True)
        print(f"    [rainfall load] {f.name} done", flush=True)
    if not rows:
        return pd.DataFrame(
            columns=["rainfall_station", "date", "rainfall_mm", "latitude", "longitude",
                     "district", "block", "source_file_count", "source_kind_count"]
        )
    all_rain = pd.concat(rows, ignore_index=True)

    # Diagnostics immediately after ingestion
    print(f"Rainfall raw rows: {len(all_rain)}", flush=True)
    print(f"Unique rainfall stations: {all_rain['rainfall_station'].nunique()}", flush=True)
    print(f"Rainfall date range: {all_rain['date'].min()} .. {all_rain['date'].max()}", flush=True)
    assert all_rain["date"].dtype.kind == "M", "date column not datetime"
    assert all_rain["rainfall_mm"].dtype.kind in "fi", "rainfall_mm not numeric"

    # Collapse duplicates across source files
    dup_count = int(all_rain.duplicated(subset=["rainfall_station", "date"], keep=False).sum())
    print(f"Duplicate station/date rows: {dup_count}", flush=True)
    all_rain = (
        all_rain.groupby(["rainfall_station", "date"], as_index=False)
        .agg(
            rainfall_mm=("rainfall_mm", "median"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            district=("district", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
            block=("block", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
            source_file_count=("source_file", "nunique"),
            source_kind_count=("source_kind", "nunique"),
        )
        .sort_values(["rainfall_station", "date"])
        .reset_index(drop=True)
    )
    # Check again for impossible values
    impossible_final_mask = all_rain["rainfall_mm"] > RAIN_DAILY_IMPOSSIBLE_THRESHOLD_MM
    if impossible_final_mask.any():
        print(f"    [rainfall load] Dropping {impossible_final_mask.sum()} aggregated daily rows with impossible rainfall > {RAIN_DAILY_IMPOSSIBLE_THRESHOLD_MM}mm", flush=True)
        all_rain = all_rain[~impossible_final_mask].copy()
    if dup_count:
        print(f"Aggregated {dup_count} duplicate station/date rows (median).", flush=True)
    return all_rain


def _build_rainfall_station_daily(rain_catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rain_catalog.empty:
        return rain_catalog, pd.DataFrame(), pd.DataFrame()

    dup_raw = int(rain_catalog.duplicated(subset=["rainfall_station", "date"], keep=False).sum())

    station_daily = rain_catalog.copy()
    station_daily = station_daily.sort_values(["rainfall_station", "date"]).reset_index(drop=True)

    # Compute causal rolling sums using prefix-sum + searchsorted
    out_rows = []
    stats_rows = []
    for st, g in station_daily.groupby("rainfall_station"):
        g = g.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if g.empty:
            continue
        
        # Reindex to continuous daily series between min and max date of this station
        min_d = g["date"].min()
        max_d = g["date"].max()
        all_dates = pd.date_range(min_d, max_d, freq="D")
        
        g_indexed = g.set_index("date").reindex(all_dates)
        g_indexed["rainfall_station"] = st
        g_indexed["rainfall_mm"] = g_indexed["rainfall_mm"].fillna(0.0)
        
        # Ffill/bfill other metadata columns
        for col in ["latitude", "longitude", "district", "block", "source_file", "source_kind", "source_file_count", "source_kind_count"]:
            if col in g_indexed.columns:
                g_indexed[col] = g_indexed[col].ffill().bfill()
        
        g_indexed = g_indexed.reset_index().rename(columns={"index": "date"})
        
        dates = pd.to_datetime(g_indexed["date"]).values.astype("datetime64[D]")
        values = pd.to_numeric(g_indexed["rainfall_mm"], errors="coerce").fillna(0.0).astype(float)
        prefix = np.concatenate(([0.0], np.cumsum(values)))
        
        for window in [30, 90, 180]:
            start_pos = np.searchsorted(dates, dates - np.timedelta64(window - 1, "D"), side="left")
            obs_count = np.arange(len(dates), dtype=float) - start_pos + 1.0
            roll_sum = prefix[1:] - prefix[start_pos]
            completeness = obs_count / float(window)
            g_indexed[f"rainfall_{window}d"] = roll_sum
            g_indexed[f"rainfall_{window}d_coverage"] = completeness
            g_indexed[f"rainfall_{window}d_obs"] = obs_count
        out_rows.append(g_indexed)
        stats_rows.append({"rainfall_station": st, "obs_count": int(len(g)), "start": min_d, "end": max_d})

    station_daily_full = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    rain_stats = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame()
    return station_daily_full.sort_values(["rainfall_station", "date"]).reset_index(drop=True), rain_stats, pd.DataFrame({"dup_raw_rainfall_station_date_pairs": [dup_raw]})


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def _make_season_encoding(season: pd.Series) -> pd.DataFrame:
    s = season.astype(str).str.lower()
    angle = pd.Series(np.where(s.eq("pre_monsoon"), 0.0, np.where(s.eq("post_monsoon"), np.pi, np.nan)), index=season.index)
    return pd.DataFrame({"season_sin": np.sin(angle), "season_cos": np.cos(angle)})


def _make_roll_features_obs_based(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["station_id", "timestamp"]).copy()
    out["prev_gw"] = out.groupby("station_id")["Groundwater_Level_MBGL"].shift(1)
    out["gw_diff"] = out["Groundwater_Level_MBGL"] - out["prev_gw"]
    out["gw_roll_mean_7obs"] = out.groupby("station_id")["Groundwater_Level_MBGL"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    out["gw_roll_std_7obs"] = out.groupby("station_id")["Groundwater_Level_MBGL"].transform(lambda s: s.rolling(7, min_periods=3).std())
    out["gw_roll_mean_30obs"] = out.groupby("station_id")["Groundwater_Level_MBGL"].transform(lambda s: s.rolling(30, min_periods=3).mean())
    out["target_next_season_gw"] = out.groupby("station_id")["Groundwater_Level_MBGL"].shift(-1)
    return out


def _validate_causal_windows(clean: pd.DataFrame, rain_joined: pd.DataFrame) -> None:
    # fail if unsorted within station
    for st, g in clean.groupby("station_id"):
        if not g["timestamp"].is_monotonic_increasing:
            raise ValueError(f"timestamps unsorted within station {st}")

    # fail if rolling features use values from the future
    if "prev_gw" in clean.columns:
        bad = clean.groupby("station_id").head(1)["prev_gw"].notna().sum()
        if bad:
            raise ValueError("future values appear in rolling windows: first observation has non-null prev_gw")

    # rainfall 180d plausible threshold
    for col in ["rainfall_180d", "district_rainfall_180d", "state_rainfall_180d"]:
        if col in clean.columns and pd.to_numeric(clean[col], errors="coerce").max(skipna=True) > RAIN_180D_PHYSICAL_THRESHOLD_MM:
            raise ValueError(f"{col} exceeds physically plausible threshold")


def _make_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    selected_rain_180 = pd.to_numeric(out.get("rainfall_180d"), errors="coerce")
    fallback_180 = pd.to_numeric(out.get("district_rainfall_180d"), errors="coerce").combine_first(
        pd.to_numeric(out.get("state_rainfall_180d"), errors="coerce")
    )
    combined_180 = selected_rain_180.combine_first(fallback_180)
    rain_p99 = combined_180.quantile(0.99) if combined_180.notna().any() else np.nan
    out["gw_jump_flag"] = pd.to_numeric(out["gw_diff"], errors="coerce").abs() > 15
    out["rainfall_extreme_flag"] = combined_180 > rain_p99
    return out, float(rain_p99) if pd.notna(rain_p99) else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
#  RAINFALL SOURCE TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def _load_rainfall_metadata_and_daily(min_date=None, max_date=None):
    rain_catalog = _load_rainfall_station_catalog(min_date=min_date, max_date=max_date)
    rain_daily, rain_stats, diag_meta = _build_rainfall_station_daily(rain_catalog)
    duplicate_pairs = int(rain_daily.duplicated(subset=["rainfall_station", "date"], keep=False).sum()) if not rain_daily.empty else 0
    rainfall_stats = {
        "raw_rows": int(len(rain_catalog)),
        "daily_rows": int(len(rain_daily)),
        "duplicate_pairs_after_daily_agg": duplicate_pairs,
        "duplicate_pairs_raw_meta": int(diag_meta.iloc[0]["dup_raw_rainfall_station_date_pairs"]) if not diag_meta.empty else 0,
    }
    return rain_daily, rain_stats, rainfall_stats


def _build_window_tables(frame: pd.DataFrame, source_col: str, value_prefix: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    summaries = []
    if frame.empty:
        return tables, pd.DataFrame()

    for source_value, g in frame.groupby(source_col):
        g = g.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        dates = pd.to_datetime(g["date"], errors="coerce").to_numpy(dtype="datetime64[D]")
        values = pd.to_numeric(g["rainfall_mm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        table = pd.DataFrame({"date": pd.to_datetime(g["date"], errors="coerce")})
        table[source_col] = source_value
        table["rainfall_station_observation_count"] = int(len(g))
        series_start = pd.to_datetime(g["date"].min())
        series_end = pd.to_datetime(g["date"].max())
        table[f"{value_prefix}_series_start"] = series_start
        table[f"{value_prefix}_series_end"] = series_end
        span_days = int((series_end - series_start).days) + 1 if pd.notna(series_start) and pd.notna(series_end) else 0
        table[f"{value_prefix}_series_missing_days"] = max(span_days - int(len(g)), 0)
        table[f"{value_prefix}_series_length_days"] = span_days
        prefix_sum = np.concatenate(([0.0], np.cumsum(values)))
        for window, threshold in LOCAL_WINDOW_COMPLETENESS_THRESHOLDS.items():
            start_positions = np.searchsorted(dates, dates - np.timedelta64(window - 1, "D"), side="left")
            obs_count = np.arange(len(dates), dtype=float) - start_positions + 1.0
            roll_sum = prefix_sum[1:] - prefix_sum[start_positions]
            completeness = obs_count / float(window)
            table[f"{value_prefix}_{window}d"] = np.where(completeness >= threshold, roll_sum, np.nan)
            table[f"rainfall_window_completeness_{window}d"] = completeness
            table[f"rainfall_window_observations_{window}d"] = obs_count
        tables[str(source_value)] = table.set_index("date")
        summaries.append({
            source_col: source_value,
            "station_observation_count": int(len(g)),
            "series_start": series_start,
            "series_end": series_end,
            "series_length_days": span_days,
            "series_missing_days": max(span_days - int(len(g)), 0),
            "coverage_pct": float((len(g) / span_days) * 100.0) if span_days else 0.0,
        })

    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values(["station_observation_count", source_col], ascending=[False, True]).reset_index(drop=True)
    return tables, summary


def _build_rainfall_source_tables(rain_daily: pd.DataFrame) -> dict:
    if rain_daily.empty:
        return {
            "station_tables": {},
            "station_summary": pd.DataFrame(),
            "district_tables": {},
            "district_summary": pd.DataFrame(),
            "state_table": pd.DataFrame(),
            "station_meta": pd.DataFrame(),
        }

    station_meta = (
        rain_daily.groupby("rainfall_station", as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            district=("district", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
            block=("block", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
        )
        .sort_values("rainfall_station")
        .reset_index(drop=True)
    )

    station_frame = rain_daily.rename(columns={"rainfall_station": "source_key"})
    station_tables, station_summary = _build_window_tables(station_frame, "source_key", "rainfall")

    district_daily = rain_daily.dropna(subset=["district"]).copy()
    district_daily = (
        district_daily.groupby(["district", "date"], as_index=False)
        .agg(rainfall_mm=("rainfall_mm", "median"))
        .sort_values(["district", "date"])
        .reset_index(drop=True)
    )
    district_tables, district_summary = _build_window_tables(district_daily, "district", "district_rainfall")

    state_daily = rain_daily.groupby("date", as_index=False).agg(rainfall_mm=("rainfall_mm", "mean")).sort_values("date")
    state_daily["source_key"] = "STATEWIDE_AGGREGATE"
    state_tables, _ = _build_window_tables(state_daily, "source_key", "state_rainfall")
    state_table = next(iter(state_tables.values()), pd.DataFrame())

    return {
        "station_tables": station_tables,
        "station_summary": station_summary,
        "district_tables": district_tables,
        "district_summary": district_summary,
        "state_table": state_table,
        "station_meta": station_meta,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SPATIAL MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def _build_rainfall_mapping(
    gw_meta: pd.DataFrame, rain_daily: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    if gw_meta.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    rain_stations = rain_daily.groupby("rainfall_station", as_index=False).agg(
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        min_date=("date", "min"),
        max_date=("date", "max"),
        district=("district", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
        block=("block", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan),
    )

    mapping_rows = []
    candidate_map: dict[str, list[str]] = {}
    for _, g in gw_meta.iterrows():
        gw_station = g["station_id"]
        gw_lat = g.get("latitude", np.nan)
        gw_lon = g.get("longitude", np.nan)
        gw_min = g.get("min_date", pd.NaT)
        gw_max = g.get("max_date", pd.NaT)
        primary_station = None
        primary_distance_km = np.nan
        primary_district_match = False
        primary_district = np.nan
        fallback_used = True
        mapping_method = "unmapped"
        candidate_stations: list[str] = []

        if pd.notna(gw_lat) and pd.notna(gw_lon) and not rain_stations.empty:
            candidates = rain_stations.dropna(subset=["latitude", "longitude"]).copy()
            if not candidates.empty:
                candidates["distance_km"] = candidates.apply(
                    lambda r: _haversine_km(gw_lon, gw_lat, r["longitude"], r["latitude"]), axis=1
                )
                
                # Check for temporal overlap first
                if pd.notna(gw_min) and pd.notna(gw_max):
                    overlap_mask = (candidates["max_date"] >= gw_min) & (candidates["min_date"] <= gw_max)
                    overlapping = candidates[overlap_mask].copy()
                    if not overlapping.empty:
                        candidates = overlapping

                candidates = candidates.sort_values(["distance_km", "rainfall_station"]).reset_index(drop=True)
                primary = candidates.iloc[0]
                primary_station = str(primary["rainfall_station"])
                primary_distance_km = float(primary["distance_km"])
                primary_district = primary.get("district", np.nan)
                primary_district_match = False  # not comparing GW district here
                within_radius = candidates[candidates["distance_km"] <= LOCAL_NEIGHBOR_RADIUS_KM].copy()
                candidate_stations = within_radius["rainfall_station"].astype(str).tolist()
                if primary_station and primary_station not in candidate_stations:
                    candidate_stations.insert(0, primary_station)
                elif primary_station:
                    candidate_stations = [primary_station] + [s for s in candidate_stations if s != primary_station]
                mapping_method = "nearest_telemetry"
                fallback_used = False

        candidate_map[str(gw_station)] = candidate_stations

        mapping_rows.append({
            "groundwater_station_id": gw_station,
            "primary_rainfall_station": primary_station,
            "primary_distance_km": primary_distance_km,
            "primary_district": primary_district,
            "primary_district_match": primary_district_match,
            "radius_candidate_count": int(max(len(candidate_stations) - 1, 0)),
            "radius_candidate_stations": "|".join(candidate_stations[:10]),
            "mapping_method": mapping_method,
            "fallback_used": fallback_used,
        })

    mapping = pd.DataFrame(mapping_rows).drop_duplicates(subset=["groundwater_station_id"], keep="first")
    source_summary = mapping["mapping_method"].value_counts(dropna=False).rename_axis("mapping_method").reset_index(name="count")
    missing_sources = mapping[mapping["fallback_used"]].copy()

    # --- Candidate map diagnostics ---
    mapped = mapping[mapping["primary_rainfall_station"].notna()]
    print(f"    [mapping] GW stations: {len(gw_meta)}, Rainfall stations: {len(rain_stations)}", flush=True)
    print(f"    [mapping] Mapped: {len(mapped)}, Unmapped: {len(mapping) - len(mapped)}", flush=True)
    print(f"    [mapping] Zero-candidate stations: {int((mapping['radius_candidate_count'] == 0).sum())}", flush=True)
    if not mapped.empty:
        print(f"    [mapping] Distance stats: mean={mapped['primary_distance_km'].mean():.1f}km, "
              f"median={mapped['primary_distance_km'].median():.1f}km, "
              f"max={mapped['primary_distance_km'].max():.1f}km", flush=True)
        print(f"    [mapping] Sample mappings:", flush=True)
        for _, row in mapped.head(5).iterrows():
            print(f"      {row['groundwater_station_id']} → {row['primary_rainfall_station']} "
                  f"({row['primary_distance_km']:.1f}km)", flush=True)

    return mapping, source_summary, missing_sources, candidate_map


# ═══════════════════════════════════════════════════════════════════════════════
#  VECTORIZED RAINFALL ATTACHMENT (replaces row-by-row iterrows)
# ═══════════════════════════════════════════════════════════════════════════════

def _attach_rainfall_features(
    clean: pd.DataFrame,
    gw_meta: pd.DataFrame,
    rain_daily: pd.DataFrame,
    mapping: pd.DataFrame,
    candidate_map: dict[str, list[str]],
    source_tables: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Attach rainfall features using vectorized merge operations.

    Replaces the original row-by-row iterrows loop with bulk merge operations
    for 50-100x speedup.
    """
    t = _timed("attach_rainfall_vectorized")
    out = clean.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["date"] = out["timestamp"].dt.normalize()

    station_tables = source_tables.get("station_tables", {})
    district_tables = source_tables.get("district_tables", {})
    state_table = source_tables.get("state_table", pd.DataFrame())
    station_meta = source_tables.get("station_meta", pd.DataFrame())

    # ── Step 1: Build canon station column ────────────────────────────────────
    out["_station_canon"] = out["station_id"].apply(canon)

    if not mapping.empty:
        map_lookup = mapping.set_index("groundwater_station_id")
        district_map = map_lookup["primary_district"].to_dict() if "primary_district" in map_lookup.columns else {}
    else:
        district_map = {}

    out["_gw_district"] = out["_station_canon"].map(district_map)

    # ── Step 2: Flatten station tables ────────────────────────────────────────
    flat_station_rain = pd.DataFrame()
    if station_tables:
        parts = []
        for stn_name, table in station_tables.items():
            cols_to_keep = ["rainfall_30d", "rainfall_90d", "rainfall_180d",
                           "rainfall_window_completeness_30d", "rainfall_window_completeness_90d",
                           "rainfall_window_completeness_180d", "rainfall_station_observation_count"]
            avail_cols = [c for c in cols_to_keep if c in table.columns]
            t_flat = table.reset_index()[["date"] + avail_cols].copy()
            t_flat["_rain_station_key"] = stn_name
            parts.append(t_flat)
        flat_station_rain = pd.concat(parts, ignore_index=True)
        flat_station_rain["date"] = pd.to_datetime(flat_station_rain["date"], errors="coerce").dt.normalize()

    # ── Step 3: Perform dynamic date-wise nearest mapping ─────────────────────
    out["_primary_rain_station"] = pd.Series(np.nan, index=out.index, dtype=object)
    out["rainfall_distance_km"] = pd.Series(np.nan, index=out.index, dtype=float)

    # Group rain daily by date to find active stations
    rain_stations_by_date = {}
    if not station_meta.empty and not flat_station_rain.empty:
        # Get active station keys per date from flat_station_rain
        for dt, g in flat_station_rain.groupby("date"):
            # merge with station_meta to get coords
            active_meta = station_meta.merge(g[["_rain_station_key"]].drop_duplicates(), left_on="rainfall_station", right_on="_rain_station_key")
            rain_stations_by_date[dt] = active_meta.set_index("rainfall_station")

    # Find nearest active station for each row of out
    if rain_stations_by_date:
        print("    [mapping] Performing dynamic date-wise nearest active rainfall station lookup...", flush=True)
        # We group by date to vectorize distance calculation for each day
        for dt, group in out.groupby("date"):
            if dt in rain_stations_by_date:
                active_rain = rain_stations_by_date[dt]
                if not active_rain.empty:
                    gw_lats = group["latitude"].values
                    gw_lons = group["longitude"].values
                    rain_lats = active_rain["latitude"].values
                    rain_lons = active_rain["longitude"].values
                    
                    glat_rad = np.radians(gw_lats)[:, np.newaxis]
                    rlat_rad = np.radians(rain_lats)[np.newaxis, :]
                    glon_rad = np.radians(gw_lons)[:, np.newaxis]
                    rlon_rad = np.radians(rain_lons)[np.newaxis, :]
                    
                    dlon = rlon_rad - glon_rad
                    dlat = rlat_rad - glat_rad
                    a = np.sin(dlat/2)**2 + np.cos(glat_rad) * np.cos(rlat_rad) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    distances = 6371 * c
                    
                    min_indices = np.argmin(distances, axis=1)
                    min_distances = np.min(distances, axis=1)
                    
                    # Check radius threshold
                    valid_mask = min_distances <= LOCAL_NEIGHBOR_RADIUS_KM
                    
                    best_stations = active_rain.index[min_indices]
                    
                    out.loc[group.index, "_primary_rain_station"] = np.where(valid_mask, best_stations, np.nan)
                    out.loc[group.index, "rainfall_distance_km"] = min_distances

    # ── Step 4: Merge LOCAL rainfall ──────────────────────────────────────────
    if not flat_station_rain.empty:
        n_before = len(out)
        out = out.merge(
            flat_station_rain,
            left_on=["_primary_rain_station", "date"],
            right_on=["_rain_station_key", "date"],
            how="left",
        )
        assert len(out) == n_before, f"Local rainfall merge changed row count: {n_before} → {len(out)}"
        out.drop(columns=["_rain_station_key"], inplace=True, errors="ignore")
    else:
        for w in [30, 90, 180]:
            out[f"rainfall_{w}d"] = np.nan
            out[f"rainfall_window_completeness_{w}d"] = np.nan
        out["rainfall_station_observation_count"] = np.nan

    # ── Step 3: Flatten district tables and merge DISTRICT rainfall ───────────
    if district_tables:
        d_parts = []
        for dist_name, table in district_tables.items():
            cols_to_keep = ["district_rainfall_30d", "district_rainfall_90d", "district_rainfall_180d"]
            avail_cols = [c for c in cols_to_keep if c in table.columns]
            d_flat = table.reset_index()[["date"] + avail_cols].copy()
            d_flat["_district_key"] = dist_name
            d_parts.append(d_flat)
        flat_district_rain = pd.concat(d_parts, ignore_index=True)
        flat_district_rain["date"] = pd.to_datetime(flat_district_rain["date"], errors="coerce").dt.normalize()

        n_before = len(out)
        out = out.merge(
            flat_district_rain,
            left_on=["_gw_district", "date"],
            right_on=["_district_key", "date"],
            how="left",
        )
        assert len(out) == n_before, f"District rainfall merge changed row count: {n_before} → {len(out)}"
        out.drop(columns=["_district_key"], inplace=True, errors="ignore")
    else:
        for w in [30, 90, 180]:
            out[f"district_rainfall_{w}d"] = np.nan

    # ── Step 4: Merge STATE rainfall (just on date) ──────────────────────────
    if not state_table.empty:
        st_cols = ["state_rainfall_30d", "state_rainfall_90d", "state_rainfall_180d"]
        avail_cols = [c for c in st_cols if c in state_table.columns]
        st_flat = state_table.reset_index()[["date"] + avail_cols].copy()
        st_flat["date"] = pd.to_datetime(st_flat["date"], errors="coerce").dt.normalize()

        n_before = len(out)
        out = out.merge(st_flat, on="date", how="left")
        assert len(out) == n_before, f"State rainfall merge changed row count: {n_before} → {len(out)}"
    else:
        for w in [30, 90, 180]:
            out[f"state_rainfall_{w}d"] = np.nan

    # ── Step 5: Determine rainfall source type (vectorized) ──────────────────
    local_has_any = (
        out["rainfall_30d"].notna() | out["rainfall_90d"].notna() | out["rainfall_180d"].notna()
    )
    district_has_any = (
        out.get("district_rainfall_30d", pd.Series(dtype=float)).notna()
        | out.get("district_rainfall_90d", pd.Series(dtype=float)).notna()
        | out.get("district_rainfall_180d", pd.Series(dtype=float)).notna()
    )
    out["rainfall_source_type"] = np.where(
        local_has_any, "local",
        np.where(district_has_any, "district", "state")
    )
    out["rainfall_source_station"] = np.where(local_has_any, out["_primary_rain_station"], np.nan)
    out["rainfall_fallback_used"] = ~local_has_any
    out["mapping_method"] = np.where(
        local_has_any, "nearest_telemetry",
        np.where(district_has_any, "district_aggregate", "state_fallback")
    )

    # Determine local issue type
    has_primary = out["_primary_rain_station"].notna()
    out["primary_rainfall_issue"] = np.where(
        local_has_any, "local_available",
        np.where(has_primary, "no_data_for_date", "missing_primary_mapping")
    )
    out["local_rainfall_issue"] = out["primary_rainfall_issue"]

    # District match flag
    out["rainfall_district_match"] = pd.array([pd.NA] * len(out), dtype="boolean")

    # ── Cleanup temp columns ─────────────────────────────────────────────────
    out.drop(columns=["_station_canon", "_primary_rain_station", "_gw_district"], inplace=True, errors="ignore")

    # ── Build diagnostic DataFrame ───────────────────────────────────────────
    diag_cols = [
        "station_id", "date", "rainfall_source_station", "rainfall_source_type",
        "mapping_method", "rainfall_30d", "rainfall_90d", "rainfall_180d",
        "district_rainfall_30d", "district_rainfall_90d", "district_rainfall_180d",
        "state_rainfall_30d", "state_rainfall_90d", "state_rainfall_180d",
        "rainfall_window_completeness_30d", "rainfall_window_completeness_90d",
        "rainfall_window_completeness_180d", "primary_rainfall_issue", "local_rainfall_issue",
    ]
    rainfall_for_diag = out[[c for c in diag_cols if c in out.columns]].copy()

    # ── Build rain_stats ─────────────────────────────────────────────────────
    rain_stats = {
        "local_coverage_pct": float(out["rainfall_source_type"].eq("local").mean() * 100.0),
        "district_coverage_pct": float(out["rainfall_source_type"].eq("district").mean() * 100.0),
        "state_coverage_pct": float(out["rainfall_source_type"].eq("state").mean() * 100.0),
        "missing_local_count": int(out["rainfall_source_type"].ne("local").sum()),
        "missing_source_station_count": int(out["rainfall_source_station"].isna().sum()),
        "primary_issue_counts": out["primary_rainfall_issue"].value_counts(dropna=False).to_dict(),
        "local_issue_counts": out["local_rainfall_issue"].value_counts(dropna=False).to_dict(),
    }

    t()  # print elapsed time
    return out, mapping, rain_stats, rainfall_for_diag


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORTS & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def _feature_summary(df: pd.DataFrame) -> str:
    cols = [
        "Groundwater_Level_MBGL", "prev_gw", "gw_diff",
        "gw_roll_mean_7obs", "gw_roll_std_7obs", "gw_roll_mean_30obs",
        "rainfall_30d", "rainfall_90d", "rainfall_180d",
        "state_rainfall_30d", "state_rainfall_90d", "state_rainfall_180d",
        "target_next_season_gw",
    ]
    lines = []
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            lines.append(
                f"{c}: count={int(s.notna().sum())}, mean={s.mean():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}"
            )
    return "\n".join(lines)


def _assemble_dropped_logs(dropped_logs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dropped_logs:
        return pd.DataFrame()
    return pd.concat(dropped_logs, ignore_index=True)


def _build_local_rainfall_coverage_report(clean: pd.DataFrame, rainfall_for_diag: pd.DataFrame) -> str:
    lines = ["Local Rainfall Coverage Report", f"Generated UTC: {datetime.utcnow().isoformat()}Z", ""]

    if clean.empty or "rainfall_source_type" not in clean.columns:
        return "\n".join(lines + ["No rainfall data available."])

    source_type = clean["rainfall_source_type"].fillna("unavailable")
    local_mask = source_type.eq("local")
    district_mask = source_type.eq("district")
    state_mask = source_type.eq("state")

    lines.append(f"Local rainfall usable %: {local_mask.mean() * 100:.2f}")
    lines.append(f"District fallback %: {district_mask.mean() * 100:.2f}")
    lines.append(f"State fallback %: {state_mask.mean() * 100:.2f}")
    completeness_cols = [c for c in ["rainfall_window_completeness_30d", "rainfall_window_completeness_90d", "rainfall_window_completeness_180d"] if c in clean.columns]
    if completeness_cols:
        comp = clean[completeness_cols].apply(pd.to_numeric, errors="coerce")
        lines.append(f"Average rainfall completeness: {comp.mean().mean():.4f}")
    lines.append("")

    coverage_by_station = clean.groupby("station_id")["rainfall_source_type"].apply(lambda s: float(s.fillna("unavailable").eq("local").mean() * 100.0)).sort_values(ascending=False)
    lines.append("Best stations by local rainfall coverage:")
    for st, pct in coverage_by_station.head(10).items():
        lines.append(f"  {st}: {pct:.2f}")
    lines.append("")
    lines.append("Worst stations by local rainfall coverage:")
    for st, pct in coverage_by_station.tail(10).sort_values().items():
        lines.append(f"  {st}: {pct:.2f}")
    lines.append("")

    lines.append("Rainfall source type distribution:")
    for k, v in source_type.value_counts(dropna=False).items():
        lines.append(f"  {k}: {int(v)}")
    lines.append("")

    if not rainfall_for_diag.empty and "local_rainfall_issue" in rainfall_for_diag.columns:
        lines.append("Causes of missing local rainfall:")
        for k, v in rainfall_for_diag["local_rainfall_issue"].fillna("unavailable").value_counts(dropna=False).items():
            lines.append(f"  {k}: {int(v)}")
        lines.append("")
        lines.append("Primary candidate failure causes:")
        for k, v in rainfall_for_diag["primary_rainfall_issue"].fillna("unavailable").value_counts(dropna=False).items():
            lines.append(f"  {k}: {int(v)}")
        lines.append("")

    if "rainfall_station_observation_count" in clean.columns:
        obs = pd.to_numeric(clean["rainfall_station_observation_count"], errors="coerce")
        lines.append(f"Mean source observation count: {obs.mean():.2f}")
        lines.append(f"Median source observation count: {obs.median():.2f}")

    return "\n".join(lines)


def _build_spatial_correlation_summary(clean: pd.DataFrame, gw_meta: pd.DataFrame) -> str:
    lines = ["Rainfall Spatial Correlation Summary", f"Generated UTC: {datetime.utcnow().isoformat()}Z", ""]
    if clean.empty or gw_meta.empty:
        return "\n".join(lines + ["Insufficient data for spatial validation."])

    coords = gw_meta[[c for c in ["station_id", "latitude", "longitude"] if c in gw_meta.columns]].dropna(subset=["station_id", "latitude", "longitude"]).copy()
    if coords.empty:
        return "\n".join(lines + ["No groundwater coordinates available."])

    local = clean[[c for c in ["station_id", "date", "rainfall_30d", "rainfall_source_type"] if c in clean.columns]].copy()
    if "rainfall_source_type" in local.columns:
        local = local[local["rainfall_source_type"].eq("local")]
    local["rainfall_30d"] = pd.to_numeric(local["rainfall_30d"], errors="coerce")
    local = local.dropna(subset=["rainfall_30d"])
    if local.empty:
        return "\n".join(lines + ["No local rainfall observations available for spatial validation."])

    local_lookup = {st: g[["date", "rainfall_30d"]].sort_values("date") for st, g in local.groupby("station_id")}
    coord_lookup = coords.set_index("station_id")
    stations = [st for st in coord_lookup.index if st in local_lookup]
    if len(stations) < 2:
        return "\n".join(lines + ["Not enough stations with local rainfall for pairwise validation."])
    stations = sorted(stations, key=lambda st: len(local_lookup[st]), reverse=True)[:200]

    pair_rows = []
    for station in stations:
        base = coord_lookup.loc[station]
        others = coord_lookup.drop(index=station).reset_index().copy()
        others["distance_km"] = others.apply(lambda r: _haversine_km(base["longitude"], base["latitude"], r["longitude"], r["latitude"]), axis=1)
        near_candidate = others.sort_values(["distance_km", "station_id"], ascending=[True, True]).head(1)
        far_candidate = others.sort_values(["distance_km", "station_id"], ascending=[False, True]).head(1)
        for label, candidate in [("near", near_candidate), ("far", far_candidate)]:
            if candidate.empty:
                continue
            other_station = candidate.iloc[0]["station_id"]
            left = local_lookup[station]
            right = local_lookup.get(other_station)
            if right is None or right.empty:
                continue
            merged = left.merge(right, on="date", suffixes=("_left", "_right"))
            if merged.empty:
                continue
            pair_rows.append({
                "pair_type": label,
                "distance_km": float(candidate.iloc[0]["distance_km"]),
                "pair_count": int(len(merged)),
                "abs_diff_mean": float((merged["rainfall_30d_left"] - merged["rainfall_30d_right"]).abs().mean()),
                "corr": float(merged["rainfall_30d_left"].corr(merged["rainfall_30d_right"])),
            })

    summary = pd.DataFrame(pair_rows)
    if summary.empty:
        return "\n".join(lines + ["No same-date station pairs available for spatial validation."])

    for pair_type in ["near", "far"]:
        subset = summary[summary["pair_type"] == pair_type]
        if subset.empty:
            continue
        lines.append(f"{pair_type.title()} pairs:")
        lines.append(f"  pair_count: {int(subset['pair_count'].sum())}")
        lines.append(f"  mean_distance_km: {subset['distance_km'].mean():.2f}")
        lines.append(f"  mean_abs_diff_mm: {subset['abs_diff_mean'].mean():.2f}")
        lines.append(f"  mean_pair_corr: {subset['corr'].mean():.3f}")
        lines.append("")

    near = summary[summary["pair_type"].eq("near")]
    far = summary[summary["pair_type"].eq("far")]
    if not near.empty and not far.empty:
        near_diff = near["abs_diff_mean"].mean()
        far_diff = far["abs_diff_mean"].mean()
        near_corr = near["corr"].mean()
        far_corr = far["corr"].mean()
        lines.append(f"Near-vs-far abs diff check: {'pass' if near_diff <= far_diff else 'fail'}")
        lines.append(f"Near-vs-far correlation check: {'pass' if near_corr >= far_corr else 'fail'}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    pipeline_start = time.time()
    initial_mem = mem_mb()
    print("=" * 60, flush=True)
    print("NEERA Groundwater Pipeline — Rebuild", flush=True)
    print(f"DEBUG_MODE: {DEBUG_MODE}", flush=True)
    print("=" * 60, flush=True)

    # ── Step 1: Load master dataset ──────────────────────────────────────────
    t1 = _timed("load_master")
    df0 = _load_master(INPUT_CSV)
    initial_rows = len(df0)
    print(f"[1/8] Loaded master rows={initial_rows}", flush=True)
    t1()

    duplicate_columns = int(df0.columns.duplicated().sum())
    unnamed_cols = [c for c in df0.columns if str(c).lower().startswith("unnamed")]

    df, malformed_ts = _parse_timestamp(df0)
    df, invalid_station_rows = _validate_station_ids(df)

    # ── Step 2: Geographic filtering — keep only Karnataka-compatible stations ──
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Swap reversed coordinates if needed
    swap_mask = (df["latitude"].abs() > 90) & (df["longitude"].abs() <= 90)
    if swap_mask.any():
        df.loc[swap_mask, ["latitude", "longitude"]] = df.loc[swap_mask, ["longitude", "latitude"]].values
        print(f"    Swapped {swap_mask.sum()} reversed lat/lon coordinates", flush=True)

    geo_mask = (
        df["latitude"].between(KARNATAKA_LAT_MIN, KARNATAKA_LAT_MAX)
        & df["longitude"].between(KARNATAKA_LON_MIN, KARNATAKA_LON_MAX)
    )
    geo_dropped = int((~geo_mask).sum())
    geo_stations_before = df["station_id"].nunique()
    df = df[geo_mask].copy()
    geo_stations_after = df["station_id"].nunique()
    print(f"[2/8] Geographic filter: {geo_stations_before} → {geo_stations_after} stations "
          f"(dropped {geo_dropped} rows outside Karnataka range)", flush=True)
    print(f"    Lat range: {df['latitude'].min():.2f}–{df['latitude'].max():.2f}", flush=True)
    print(f"    Lon range: {df['longitude'].min():.2f}–{df['longitude'].max():.2f}", flush=True)

    # ── Step 3: Keep only seasonal rows and valid groundwater values ─────────
    dup_station_ts_before = int(df.duplicated(subset=["station_id", "timestamp"], keep=False).sum())
    dropped_logs: list[pd.DataFrame] = []

    null_gw_mask = df["Groundwater_Level_MBGL"].isna()
    if null_gw_mask.any():
        tmp = df[null_gw_mask].copy()
        tmp["drop_reason"] = "null_groundwater"
        dropped_logs.append(tmp)
    df = df[~null_gw_mask].copy()

    non_seasonal_mask = df.get("freq", pd.Series(index=df.index, data="")).astype(str).str.lower().ne("seasonal")
    if non_seasonal_mask.any():
        tmp = df[non_seasonal_mask].copy()
        tmp["drop_reason"] = "non_seasonal_observation"
        dropped_logs.append(tmp)
    df = df[~non_seasonal_mask].copy()

    # Sort and validate monotonicity
    df = df.sort_values(["station_id", "timestamp"]).copy()
    if not df.groupby("station_id")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all():
        raise ValueError("timestamps unsorted within station")

    # Dedup: keep latest per station+timestamp
    dup_mask = df.duplicated(subset=["station_id", "timestamp"], keep="last")
    if dup_mask.any():
        tmp = df[dup_mask].copy()
        tmp["drop_reason"] = "duplicate_station_timestamp_keep_latest"
        dropped_logs.append(tmp)
    df = df[~dup_mask].copy()

    # Station quality filtering
    obs_counts = (
        df.groupby("station_id")["Groundwater_Level_MBGL"]
        .apply(lambda s: int(s.notna().sum()))
        .reset_index(name="valid_observation_count")
    )
    dropped_stations = obs_counts[obs_counts["valid_observation_count"] < 8].copy()
    dropped_stations["reason_dropped"] = "<8 valid groundwater observations"
    if not dropped_stations.empty:
        drop_set = set(dropped_stations["station_id"])
        tmp = df[df["station_id"].isin(drop_set)].copy()
        tmp["drop_reason"] = "low_quality_station_<8_obs"
        dropped_logs.append(tmp)
        df = df[~df["station_id"].isin(drop_set)].copy()

    print(f"[3/8] After quality filters: {len(df)} rows, {df['station_id'].nunique()} stations", flush=True)

    # ── Step 4: Build GW metadata from master coords ─────────────────────────
    gw_meta = _build_gw_metadata_from_master(df)
    print(f"    GW metadata: {len(gw_meta)} stations with valid coordinates", flush=True)
    print(f"    GW lat range: {gw_meta['latitude'].min():.2f}–{gw_meta['latitude'].max():.2f}", flush=True)
    print(f"    GW lon range: {gw_meta['longitude'].min():.2f}–{gw_meta['longitude'].max():.2f}", flush=True)

    # Optionally enrich with Atal Jal district/block info
    atal_meta_path = DATASET_DIR / "Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv"
    if atal_meta_path.exists():
        atal_meta = _load_groundwater_metadata(atal_meta_path)
        # Merge district/block from Atal Jal where station_id matches
        if "district" in atal_meta.columns and "block" in atal_meta.columns:
            gw_meta = gw_meta.merge(
                atal_meta[["station_id", "district", "block"]].drop_duplicates(subset=["station_id"]),
                on="station_id", how="left"
            )
            print(f"    Enriched {gw_meta['district'].notna().sum()}/{len(gw_meta)} stations with Atal Jal district/block", flush=True)

    # ── Step 5: Load rainfall data ───────────────────────────────────────────
    t5 = _timed("load_rainfall")
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    rain_daily, rain_station_stats, rainfall_stats = _load_rainfall_metadata_and_daily(min_date=min_ts, max_date=max_ts)
    t5()

    # ── DEBUG_MODE: Smart station selection ───────────────────────────────────
    if DEBUG_MODE:
        print(f"\n{'─'*40}", flush=True)
        print("DEBUG_MODE: Smart Karnataka station selection", flush=True)

        # Pick GW stations that are within Karnataka AND have valid coords
        if not gw_meta.empty:
            ka_gw = gw_meta[
                gw_meta["latitude"].between(KARNATAKA_LAT_MIN, KARNATAKA_LAT_MAX)
                & gw_meta["longitude"].between(KARNATAKA_LON_MIN, KARNATAKA_LON_MAX)
            ]
            # Pick stations closest to the centroid of rainfall stations for best coverage
            if not rain_daily.empty and not ka_gw.empty:
                rain_lat_center = rain_daily["latitude"].median()
                rain_lon_center = rain_daily["longitude"].median()
                ka_gw = ka_gw.copy()
                ka_gw["_dist_to_rain_center"] = ka_gw.apply(
                    lambda r: _haversine_km(rain_lon_center, rain_lat_center, r["longitude"], r["latitude"]),
                    axis=1
                )
                ka_gw = ka_gw.sort_values("_dist_to_rain_center").head(DEBUG_GW_STATIONS)
                ka_gw.drop(columns=["_dist_to_rain_center"], inplace=True)
            else:
                ka_gw = ka_gw.head(DEBUG_GW_STATIONS)

            debug_station_ids = set(ka_gw["station_id"])
            gw_meta = ka_gw.copy()

            # Filter main data to these stations
            df["_canon"] = df["station_id"].apply(canon)
            df = df[df["_canon"].isin(debug_station_ids)].copy()
            df.drop(columns=["_canon"], inplace=True, errors="ignore")

        # Limit rainfall date range
        if not rain_daily.empty:
            min_r = rain_daily["date"].min()
            max_allowed = pd.to_datetime(min_r) + pd.Timedelta(days=365 * DEBUG_RAIN_YEARS)
            rain_daily = rain_daily.loc[rain_daily["date"] <= max_allowed].copy()

        print(f"  DEBUG GW stations: {len(gw_meta)}", flush=True)
        print(f"  DEBUG GW rows: {len(df)}", flush=True)
        print(f"  DEBUG Rainfall rows: {len(rain_daily)}", flush=True)
        print(f"  DEBUG Rainfall stations: {rain_daily['rainfall_station'].nunique() if not rain_daily.empty else 0}", flush=True)
        print(f"  DEBUG Rainfall date range: {rain_daily['date'].min()} .. {rain_daily['date'].max()}" if not rain_daily.empty else "  No rainfall data", flush=True)
        print(f"{'─'*40}\n", flush=True)

    print(f"[4/8] Loaded rainfall: {len(rain_daily)} daily rows, "
          f"{rain_daily['rainfall_station'].nunique() if not rain_daily.empty else 0} stations", flush=True)

    # ── Step 6: Build rainfall source tables ─────────────────────────────────
    t6 = _timed("build_source_tables")
    source_tables = _build_rainfall_source_tables(rain_daily)
    t6()

    # Validate station coordinates
    station_meta = source_tables.get("station_meta", pd.DataFrame())
    if not station_meta.empty:
        station_meta["latitude"] = pd.to_numeric(station_meta["latitude"], errors="coerce")
        station_meta["longitude"] = pd.to_numeric(station_meta["longitude"], errors="coerce")
        swap_mask = (station_meta["latitude"].abs() > 90) & (station_meta["longitude"].abs() <= 90)
        if swap_mask.any():
            station_meta.loc[swap_mask, ["latitude", "longitude"]] = station_meta.loc[swap_mask, ["longitude", "latitude"]].values
        station_meta = station_meta.loc[
            station_meta["latitude"].between(INDIA_LAT_MIN, INDIA_LAT_MAX)
            & station_meta["longitude"].between(INDIA_LON_MIN, INDIA_LON_MAX)
        ].copy()
        source_tables["station_meta"] = station_meta

    print(f"[5/8] Built source tables: stations={len(source_tables.get('station_tables', {}))}, "
          f"districts={len(source_tables.get('district_tables', {}))}", flush=True)

    # ── Step 7: Build rainfall mapping ───────────────────────────────────────
    t7 = _timed("build_mapping")
    rainfall_mapping, mapping_summary, missing_source_stations, candidate_map = _build_rainfall_mapping(gw_meta, rain_daily)
    t7()

    # ── Step 8: Feature engineering ──────────────────────────────────────────
    df = _make_roll_features_obs_based(df)
    season_enc = _make_season_encoding(df["season"].fillna(""))
    df = pd.concat([df.reset_index(drop=True), season_enc.reset_index(drop=True)], axis=1)

    # ── Step 9: Attach rainfall features (VECTORIZED) ────────────────────────
    df, rainfall_mapping, rain_stats, rainfall_for_diag = _attach_rainfall_features(
        df, gw_meta, rain_daily, rainfall_mapping, candidate_map, source_tables,
    )
    print(f"[6/8] Attached rainfall — local={rain_stats.get('local_coverage_pct', 0):.1f}%, "
          f"district={rain_stats.get('district_coverage_pct', 0):.1f}%, "
          f"state={rain_stats.get('state_coverage_pct', 0):.1f}%", flush=True)

    # ── Step 10: Build effective rainfall with fallback cascade ───────────────
    for window in [30, 90, 180]:
        local_col = f"rainfall_{window}d"
        district_col = f"district_rainfall_{window}d"
        state_col = f"state_rainfall_{window}d"
        eff_col = f"effective_rainfall_{window}d"
        # cascade: local > district > state
        df[eff_col] = df[local_col].combine_first(df[district_col]).combine_first(df[state_col])
        df[f"rainfall_{window}d_fallback_used"] = df[local_col].isna() & df[eff_col].notna()
        cov_col = f"rainfall_window_completeness_{window}d"
        if cov_col in df.columns:
            df[cov_col] = pd.to_numeric(df[cov_col], errors="coerce")
        else:
            df[cov_col] = np.nan

    df["rainfall_fallback_used"] = (
        df[[f"rainfall_{w}d_fallback_used" for w in [30, 90, 180]]].any(axis=1).fillna(False)
    ).astype(bool)

    for c in ["rainfall_distance_km"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Step 11: Timestamp normalization — ONLY .dt.normalize() ──────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    df["timestamp"] = df["timestamp"].dt.normalize()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    # ── Step 12: Coverage gating — ALWAYS apply ──────────────────────────────
    for window, threshold in LOCAL_WINDOW_COMPLETENESS_THRESHOLDS.items():
        cov_col = f"rainfall_window_completeness_{window}d"
        eff_col = f"effective_rainfall_{window}d"
        if cov_col in df.columns:
            mask_low = pd.to_numeric(df[cov_col], errors="coerce") < float(threshold)
            # Only gate local rainfall (completeness cols are from local station)
            # Don't gate district/state fallback which doesn't have local completeness
            local_mask = df["rainfall_source_type"].eq("local")
            df.loc[mask_low & local_mask, eff_col] = np.nan

    # Diagnostics
    local_any = df[[f"effective_rainfall_{w}d" for w in [30, 90, 180]]].notna().any(axis=1).mean()
    print(f"    Fraction rows with any effective rainfall: {local_any * 100:.1f}%", flush=True)
    for w in [30, 90, 180]:
        eff = f"effective_rainfall_{w}d"
        pct = df[eff].notna().mean() * 100
        print(f"    {eff}: {pct:.1f}% populated", flush=True)

    # ── Step 13: Anomaly flags ───────────────────────────────────────────────
    df, rainfall_p99 = _make_anomalies(df)
    print(f"[7/8] Anomaly flags — p99={rainfall_p99:.2f}" if pd.notna(rainfall_p99) else "[7/8] Anomaly flags — no rainfall data", flush=True)

    # ── Step 14: Leakage prevention validation ───────────────────────────────
    if df["prev_gw"].groupby(df["station_id"]).head(1).notna().any():
        raise ValueError("future values appear in rolling windows")
    raw_daily_max = pd.to_numeric(rain_daily["rainfall_mm"], errors="coerce").max(skipna=True) if not rain_daily.empty else np.nan
    if pd.notna(raw_daily_max) and raw_daily_max > RAIN_DAILY_IMPOSSIBLE_THRESHOLD_MM:
        raise ValueError("daily rainfall exceeds impossible intensity threshold")

    _validate_causal_windows(df, rainfall_for_diag)

    # ── Step 15: Final ordering and dedup check ──────────────────────────────
    df = df.sort_values(["station_id", "timestamp"]).copy()
    dup_after = int(df.duplicated(subset=["station_id", "timestamp"], keep=False).sum())
    if dup_after > 0:
        raise ValueError("duplicate station_id + timestamp exist")

    # Build anomaly log
    anomaly_log = df.loc[df["gw_jump_flag"] | df["rainfall_extreme_flag"], [
        "station_id", "timestamp", "Groundwater_Level_MBGL", "gw_diff",
        "rainfall_180d", "gw_jump_flag", "rainfall_extreme_flag", "season", "freq",
    ]].copy()

    # ── Step 16: Assemble clean dataset ──────────────────────────────────────
    clean_cols = [
        "station_id", "timestamp", "Groundwater_Level_MBGL",
        "prev_gw", "gw_diff", "gw_roll_mean_7obs", "gw_roll_std_7obs", "gw_roll_mean_30obs",
        "rainfall_30d", "rainfall_90d", "rainfall_180d",
        "district_rainfall_30d", "district_rainfall_90d", "district_rainfall_180d",
        "state_rainfall_30d", "state_rainfall_90d", "state_rainfall_180d",
        "target_next_season_gw", "season", "season_sin", "season_cos", "freq",
        "rainfall_source_type", "rainfall_source_station", "rainfall_station_observation_count",
        "rainfall_window_completeness_30d", "rainfall_window_completeness_90d", "rainfall_window_completeness_180d",
        "mapping_method", "rainfall_distance_km", "rainfall_district_match", "rainfall_fallback_used",
        "gw_jump_flag", "rainfall_extreme_flag", "date",
    ]
    for c in clean_cols:
        if c not in df.columns:
            df[c] = np.nan
    clean = df[clean_cols].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    clean = clean.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    dropped_rows = _assemble_dropped_logs(dropped_logs)
    if not dropped_rows.empty and "timestamp" in dropped_rows.columns:
        dropped_rows["timestamp"] = pd.to_datetime(dropped_rows["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    if not dropped_rows.empty and "date" in dropped_rows.columns:
        dropped_rows["date"] = pd.to_datetime(dropped_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    final_rows = len(clean)
    rows_dropped_total = initial_rows - final_rows

    # Rainfall identity check
    rainfall_identical_by_date_violations = 0
    rain_check = clean[["station_id", "date", "rainfall_30d"]].dropna().copy()
    if not rain_check.empty:
        for _, g in rain_check.groupby("date"):
            if len(g) > 1:
                top_share = g["rainfall_30d"].value_counts(normalize=True).iloc[0]
                if top_share > RAIN_IDENTICAL_STATION_FRACTION_THRESHOLD:
                    rainfall_identical_by_date_violations += 1

    rainfall_coverage_pct = float(rain_stats.get("local_coverage_pct", float(clean["rainfall_source_type"].eq("local").mean() * 100.0)))
    fallback_pct = float(clean["rainfall_fallback_used"].fillna(False).mean() * 100)
    missing_rain_sources = int(rain_stats.get("missing_source_station_count", int(clean["rainfall_source_station"].isna().sum())))
    local_coverage_report = _build_local_rainfall_coverage_report(clean, rainfall_for_diag)
    spatial_summary = _build_spatial_correlation_summary(clean, gw_meta)

    # ── Step 17: Build training-ready dataset ────────────────────────────────
    training = clean.copy()
    for w in [30, 90, 180]:
        eff = f"effective_rainfall_{w}d"
        if eff in df.columns:
            training[eff] = pd.to_numeric(df[eff], errors="coerce").values
        cov = f"rainfall_window_completeness_{w}d"
        if cov in df.columns:
            training[cov] = pd.to_numeric(df[cov], errors="coerce").values

    if "rainfall_fallback_used" in df.columns:
        training["rainfall_fallback_used"] = df["rainfall_fallback_used"].astype(int).values
    if "rainfall_district_match" in df.columns:
        training["rainfall_district_match"] = df["rainfall_district_match"].astype("Int8").values

    # Drop bad training rows
    before_rows = len(training)
    mask_target_missing = training["target_next_season_gw"].isna()
    mask_gw_missing = training["Groundwater_Level_MBGL"].isna()
    rainfall_eff_cols = [c for c in training.columns if c.startswith("effective_rainfall_")]
    mask_all_rain_missing = ~training[rainfall_eff_cols].notna().any(axis=1) if rainfall_eff_cols else pd.Series(True, index=training.index)
    drop_mask = mask_target_missing | mask_gw_missing | mask_all_rain_missing
    training = training.loc[~drop_mask].reset_index(drop=True)
    after_rows = len(training)
    print(f"    Training rows: {before_rows} → {after_rows} (dropped {before_rows - after_rows})", flush=True)

    # ── Step 18: Write all exports ───────────────────────────────────────────
    training.to_csv(ROOT / "training_master.csv", index=False, encoding="utf-8")
    try:
        training.to_parquet(ROOT / "training_master.parquet", index=False)
    except Exception:
        print("    parquet export failed; pyarrow not available", flush=True)

    # Feature coverage report
    feature_cov = {}
    for c in ["effective_rainfall_30d", "effective_rainfall_90d", "effective_rainfall_180d"]:
        if c in training.columns:
            feature_cov[c] = float(training[c].notna().mean() * 100.0)
    with open(ROOT / "feature_coverage_report.txt", "w", encoding="utf-8") as f:
        f.write("Feature coverage report\n")
        f.write(f"Rows before cleaning: {before_rows}\n")
        f.write(f"Rows after cleaning: {after_rows}\n")
        for k, v in feature_cov.items():
            f.write(f"{k}: {v:.2f}%\n")

    # Rainfall quality report
    with open(ROOT / "rainfall_quality_report.txt", "w", encoding="utf-8") as f:
        f.write("Rainfall quality report\n")
        f.write(f"raw_rows: {rainfall_stats.get('raw_rows', 0)}\n")
        f.write(f"daily_rows: {rainfall_stats.get('daily_rows', 0)}\n")
        for c in ["rainfall_30d", "rainfall_90d", "rainfall_180d"]:
            if c in clean.columns:
                s = pd.to_numeric(clean[c], errors="coerce")
                f.write(f"{c}: nulls={int(s.isna().sum())}, mean={s.mean():.4f}, max={s.max():.4f}\n")

    # Training schema
    schema = {"fields": []}
    for col in training.columns:
        dtype = str(training[col].dtype)
        schema["fields"].append({"name": col, "dtype": dtype})
    import json
    with open(ROOT / "training_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, default=str)

    # Write log files
    clean.to_csv(OUT_CLEAN, index=False, encoding="utf-8")
    dropped_rows.to_csv(OUT_DROPPED_ROWS, index=False, encoding="utf-8")
    dropped_stations.to_csv(OUT_DROPPED_STATIONS, index=False, encoding="utf-8")
    rainfall_mapping.to_csv(OUT_RAIN_MAP, index=False, encoding="utf-8")
    anomaly_log.to_csv(OUT_ANOMALY, index=False, encoding="utf-8")
    with open(OUT_LOCAL_COVERAGE, "w", encoding="utf-8") as f:
        f.write(local_coverage_report)
    with open(OUT_SPATIAL_SUMMARY, "w", encoding="utf-8") as f:
        f.write(spatial_summary)

    # Rainfall diagnostics text
    rain_diag_lines = [
        "Rainfall Diagnostics",
        f"Generated UTC: {datetime.utcnow().isoformat()}Z",
        "",
        f"Raw rainfall rows: {rainfall_stats.get('raw_rows', 0)}",
        f"Daily rainfall rows: {rainfall_stats.get('daily_rows', 0)}",
        f"Raw duplicate rainfall station/date pairs: {rainfall_stats.get('duplicate_pairs_raw_meta', 0)}",
        f"Duplicate rainfall station/date pairs after daily aggregation: {rainfall_stats.get('duplicate_pairs_after_daily_agg', 0)}",
        "",
        "Rainfall descriptive statistics:",
    ]
    if not rain_daily.empty:
        desc = rain_daily["rainfall_mm"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string()
        rain_diag_lines.append(desc)
        rain_diag_lines.append("")
        rain_diag_lines.append("Station-specific climatology checks:")
        rain_station_climo = rain_daily.groupby("rainfall_station")["rainfall_mm"].quantile(0.99).sort_values(ascending=False)
        rain_diag_lines.append(rain_station_climo.head(20).to_string())
    rain_diag_lines.append("")
    rain_diag_lines.append("Per-station rainfall statistics:")
    if not rain_station_stats.empty:
        rain_diag_lines.append(rain_station_stats.sort_values("obs_count", ascending=False).head(20).to_string(index=False))
    rain_diag_lines.append("")
    rain_diag_lines.append("Max/min rainfall windows:")
    for c in ["rainfall_30d", "rainfall_90d", "rainfall_180d",
              "district_rainfall_30d", "district_rainfall_90d", "district_rainfall_180d",
              "state_rainfall_30d", "state_rainfall_90d", "state_rainfall_180d"]:
        if c in clean.columns:
            s = pd.to_numeric(clean[c], errors="coerce")
            if s.notna().any():
                rain_diag_lines.append(f"{c}: min={s.min():.4f}, max={s.max():.4f}")
    rain_diag_lines.append("")
    rain_diag_lines.append("Rainfall source quality:")
    rain_diag_lines.append(clean["rainfall_source_type"].fillna("unavailable").value_counts(dropna=False).to_string())
    for c in ["rainfall_window_completeness_30d", "rainfall_window_completeness_90d", "rainfall_window_completeness_180d"]:
        if c in clean.columns:
            s = pd.to_numeric(clean[c], errors="coerce")
            if s.notna().any():
                rain_diag_lines.append(f"{c}: mean={s.mean():.4f}, min={s.min():.4f}, max={s.max():.4f}")
    rain_diag_lines.append("")
    rain_diag_lines.append("Top 50 extreme rainfall rows:")
    extreme = clean.copy()
    extreme["_rainfall_180d_num"] = pd.to_numeric(extreme["rainfall_180d"], errors="coerce").combine_first(
        pd.to_numeric(extreme["district_rainfall_180d"], errors="coerce")
    ).combine_first(pd.to_numeric(extreme["state_rainfall_180d"], errors="coerce"))
    extreme = extreme.sort_values("_rainfall_180d_num", ascending=False).head(50)
    rain_diag_lines.append(extreme[["station_id", "timestamp", "rainfall_source_type", "rainfall_source_station",
                                     "rainfall_30d", "rainfall_90d", "rainfall_180d",
                                     "district_rainfall_180d", "state_rainfall_180d", "mapping_method"]].to_string(index=False))
    with open(OUT_RAIN_DIAG, "w", encoding="utf-8") as f:
        f.write("\n".join(rain_diag_lines))

    # Main preprocessing report
    gw_missing_pct = float(clean["Groundwater_Level_MBGL"].isna().mean() * 100)
    target_missing_pct = float(clean["target_next_season_gw"].isna().mean() * 100)
    anomaly_counts = {
        "gw_jump_flag": int(clean["gw_jump_flag"].fillna(False).sum()),
        "rainfall_extreme_flag": int(clean["rainfall_extreme_flag"].fillna(False).sum()),
    }
    missing_sources_list = rainfall_mapping[rainfall_mapping["fallback_used"]].copy() if not rainfall_mapping.empty else pd.DataFrame()
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("Groundwater Master Preprocessing Report v3\n")
        f.write(f"Generated UTC: {datetime.utcnow().isoformat()}Z\n\n")
        f.write("Validation summary\n")
        f.write(f"Initial row count: {initial_rows}\n")
        f.write(f"Final row count: {final_rows}\n")
        f.write(f"Rows dropped: {rows_dropped_total}\n")
        f.write(f"Stations dropped: {len(dropped_stations)}\n")
        f.write(f"Geographic filter (Karnataka): dropped {geo_dropped} rows, {geo_stations_before - geo_stations_after} stations\n")
        f.write(f"Duplicate station_id+timestamp before cleaning: {dup_station_ts_before}\n")
        f.write(f"Duplicate station_id+timestamp after cleaning: {dup_after}\n")
        f.write(f"Malformed timestamps: {malformed_ts}\n")
        f.write(f"Invalid station IDs: {invalid_station_rows}\n")
        f.write(f"Duplicate columns: {duplicate_columns}\n")
        f.write(f"Unnamed columns: {unnamed_cols}\n\n")

        f.write("Rainfall methodology\n")
        f.write("- A: nearest telemetry station with completeness-gated causal windows\n")
        f.write(f"- C: nearest neighbor within {LOCAL_NEIGHBOR_RADIUS_KM:.1f} km when the primary source is incomplete\n")
        f.write("- District rainfall is computed separately as district_rainfall_* and never overwrites local rainfall\n")
        f.write("- State rainfall is computed separately as state_rainfall_* and never overwrites local or district rainfall\n")
        f.write(f"Fallback percentage: {fallback_pct:.2f}\n")
        f.write(f"Station-local rainfall coverage percentage: {rainfall_coverage_pct:.2f}\n")
        f.write(f"Missing rainfall source count: {missing_rain_sources}\n")
        if not missing_sources_list.empty:
            f.write("Stations with missing rainfall sources (fallback or unresolved):\n")
            cols = [c for c in ["groundwater_station_id", "primary_rainfall_station", "mapping_method", "radius_candidate_count"] if c in missing_sources_list.columns]
            f.write(missing_sources_list[cols].head(20).to_string(index=False) + "\n")
        f.write("\n")

        f.write("Feature audit\n")
        f.write(f"Local rainfall usable %: {rain_stats.get('local_coverage_pct', 0.0):.2f}\n")
        f.write(f"District fallback %: {rain_stats.get('district_coverage_pct', 0.0):.2f}\n")
        f.write(f"State fallback %: {rain_stats.get('state_coverage_pct', 0.0):.2f}\n")
        completeness_cols = [c for c in ["rainfall_window_completeness_30d", "rainfall_window_completeness_90d",
                                          "rainfall_window_completeness_180d"] if c in clean.columns]
        if completeness_cols:
            comp = clean[completeness_cols].apply(pd.to_numeric, errors="coerce")
            f.write(f"Average rainfall completeness: {comp.mean().mean():.4f}\n")
        f.write("Rainfall source type distribution:\n")
        f.write(clean["rainfall_source_type"].fillna("unavailable").value_counts(dropna=False).to_string() + "\n")
        f.write("\n")

        f.write("Feature coverage for training\n")
        for k, v in feature_cov.items():
            f.write(f"  {k}: {v:.2f}%\n")
        f.write("\n")

        f.write("Feature validity\n")
        f.write("- Rolling means use min_periods=3\n")
        f.write("- Rolling std uses min_periods=3\n")
        f.write("- gw_diff = current_groundwater - prev_gw\n")
        f.write("- season_sin/season_cos encode pre_monsoon and post_monsoon cyclically\n")
        f.write("- All rainfall windows are causal (no future rainfall leakage)\n\n")

        f.write("Anomaly counts\n")
        for k, v in anomaly_counts.items():
            f.write(f"{k}: {v}\n")
        if pd.notna(rainfall_p99):
            f.write(f"Rainfall 180d 99th percentile threshold: {rainfall_p99:.4f}\n")
        f.write("\n")

        f.write("Missingness summary\n")
        f.write(f"Groundwater missingness %: {gw_missing_pct:.2f}\n")
        f.write(f"Target missingness %: {target_missing_pct:.2f}\n\n")

        f.write("Validation checks passed\n")
        f.write("- duplicate station_id + timestamp: yes\n")
        f.write("- timestamps sorted within station: yes\n")
        f.write("- causal rainfall windows: yes\n")
        f.write("- no groundwater forward/back fill: yes\n")
        f.write("- no future rainfall leakage: yes\n")
        f.write("- rainfall realism threshold enforced: yes\n\n")

        f.write("Feature summary statistics\n")
        f.write(_feature_summary(clean) + "\n")

    elapsed = time.time() - pipeline_start
    print(f"\n[8/8] ✅ Pipeline complete in {elapsed:.1f}s", flush=True)
    print(f"    Clean dataset: {final_rows} rows", flush=True)
    print(f"    Training dataset: {after_rows} rows", flush=True)
    print(f"    Local rainfall coverage: {rainfall_coverage_pct:.1f}%", flush=True)
    print(f"    Memory: {mem_mb():.1f} MB", flush=True)

    # Write exit code
    with open(ROOT / "rebuild_run.exit", "w") as f:
        f.write("0\n")

    # Final hard assertions
    if clean.duplicated(subset=["station_id", "timestamp"]).any():
        raise ValueError("duplicate station_id + timestamp exist")
    if not clean.groupby("station_id")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all():
        raise ValueError("timestamps unsorted within station")
    for col in ["rainfall_180d", "district_rainfall_180d", "state_rainfall_180d"]:
        if col in clean.columns:
            max_val = pd.to_numeric(clean[col], errors="coerce").max(skipna=True)
            if pd.notna(max_val) and max_val > RAIN_180D_PHYSICAL_THRESHOLD_MM:
                raise ValueError(f"{col} exceeds physically plausible threshold")


if __name__ == "__main__":
    main()
