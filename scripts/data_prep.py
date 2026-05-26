#!/usr/bin/env python3
"""
Chunk-aware data preparation pipeline for NEERA workspace.

Implements memory-efficient, chunked processing for large telemetry CSVs,
intermediate chunk writing, progress and memory logging, validation checks,
and generation of `master_dataset.csv` and `preprocessing_report.txt`.
"""

import os
import sys
import time
import logging
import resource
from pathlib import Path
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd

try:
    import psutil

    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Dataset"
OUT_CSV = ROOT / "master_dataset.csv"
REPORT_TXT = ROOT / "preprocessing_report.txt"
LOG_FILE = ROOT / "scripts" / "preprocessing.log"
TMP_DIR = ROOT / "scripts" / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_FILESIZE_THRESHOLD = 50 * 1024 * 1024  # 50MB
TELEMETRY_CHUNK_ROWS = 200_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger("data_prep")


def mem_usage_mb():
    if PSUTIL_AVAILABLE:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if ru > 1e6:
        return ru / (1024 * 1024)
    return ru / 1024.0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371000 * c


def standardize_colname(c):
    c = str(c).strip()
    c = c.replace(" ", "_").replace("(", "").replace(")", "")
    c = c.replace(".", "").replace("-", "_")
    return c


def read_csv_with_fallback(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        kw = dict(kwargs)
        kw["encoding"] = "latin1"
        return pd.read_csv(path, **kw)


def load_atal_jal(path):
    df = read_csv_with_fallback(path, dtype=str)
    df.columns = [standardize_colname(c) for c in df.columns]

    meta_cols = [
        c
        for c in df.columns
        if c.lower()
        in {
            "well_id",
            "site_name",
            "latitude",
            "longitude",
            "aquifer",
            "well_depth_meters",
            "type",
            "source",
        }
    ]
    pre_cols = [c for c in df.columns if "pre" in c.lower() and "monsoon" in c.lower()]
    post_cols = [
        c for c in df.columns if "post" in c.lower() and "monsoon" in c.lower()
    ]

    rows = []
    for _, r in df.iterrows():
        base = {k: r.get(k, np.nan) for k in meta_cols}
        lat = r.get("Latitude", r.get("latitude", ""))
        lon = r.get("Longitude", r.get("longitude", ""))
        try:
            base["Latitude"] = (
                float(lat) if lat not in (None, "", "NA", "nan") else np.nan
            )
            base["Longitude"] = (
                float(lon) if lon not in (None, "", "NA", "nan") else np.nan
            )
        except Exception:
            base["Latitude"] = np.nan
            base["Longitude"] = np.nan

        for c in pre_cols:
            year = "".join(ch for ch in c if ch.isdigit())[:4]
            if year:
                rows.append(
                    {
                        **base,
                        "timestamp": f"{year}-05-01 00:00:00",
                        "freq": "seasonal",
                        "season": "pre_monsoon",
                        "groundwater_raw": r.get(c, np.nan),
                    }
                )
        for c in post_cols:
            year = "".join(ch for ch in c if ch.isdigit())[:4]
            if year:
                rows.append(
                    {
                        **base,
                        "timestamp": f"{year}-10-01 00:00:00",
                        "freq": "seasonal",
                        "season": "post_monsoon",
                        "groundwater_raw": r.get(c, np.nan),
                    }
                )

    s = pd.DataFrame(rows)
    if "well_depth_meters" in s.columns:
        s = s.rename(columns={"well_depth_meters": "Well_Depth_m"})
    if "Well_ID" in s.columns and "Station" not in s.columns:
        s = s.rename(columns={"Well_ID": "Station"})
    if "well_id" in s.columns and "Station" not in s.columns:
        s = s.rename(columns={"well_id": "Station"})
    if "Station" not in s.columns:
        s["Station"] = s.get("Site_Name", s.get("site_name", "UNKNOWN"))

    s["Station"] = s["Station"].astype(str).str.strip()
    s.loc[s["Station"].isin(["", "nan", "None", "UNKNOWN"]), "Station"] = (
        s.get("Site_Name", s.get("site_name", "UNKNOWN")).astype(str).str.strip()
        if ("Site_Name" in s.columns or "site_name" in s.columns)
        else "UNKNOWN"
    )
    # Last-resort stable IDs to preserve station consistency when source IDs are missing.
    missing_station = s["Station"].isin(["", "nan", "None", "UNKNOWN"]) | s["Station"].isna()
    if missing_station.any():
        s.loc[missing_station, "Station"] = [f"ATAL_UNKNOWN_{i}" for i in s.index[missing_station]]

    s["groundwater_raw"] = (
        s["groundwater_raw"]
        .replace({"NA": np.nan, "Dry": np.nan, "Filled up": np.nan, "filled up": np.nan})
    )
    s["groundwater_raw"] = pd.to_numeric(s["groundwater_raw"], errors="coerce")
    s["timestamp"] = pd.to_datetime(s["timestamp"], errors="coerce")
    return s


def _detect_cols(df):
    tcol = next(
        (c for c in df.columns if "data_acquisition_time" in c.lower() or "timestamp" in c.lower()),
        None,
    )
    if tcol is None:
        tcol = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    station_col = next(
        (c for c in df.columns if "station" in c.lower() or "well_id" in c.lower() or "site" in c.lower()),
        None,
    )
    gw_col = next(
        (c for c in df.columns if "groundwater" in c.lower() or "gwl" in c.lower()),
        None,
    )
    if tcol is None:
        tcol = df.columns[0]
    if station_col is None:
        station_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if gw_col is None:
        gw_col = df.columns[-1]
    return tcol, station_col, gw_col


def process_telemetry_chunked(path):
    size = path.stat().st_size
    use_chunks = size > CHUNK_FILESIZE_THRESHOLD
    logger.info("Telemetry size %.2f MB | chunked=%s", size / (1024 * 1024), use_chunks)

    sample = read_csv_with_fallback(path, nrows=20, dtype=str)
    sample.columns = [standardize_colname(c) for c in sample.columns]
    tcol, station_col, gw_col = _detect_cols(sample)

    total_input_rows = 0
    total_clean_rows = 0
    chunk_idx = 0
    tmp_files = []

    if use_chunks:
        reader = pd.read_csv(path, dtype=str, chunksize=TELEMETRY_CHUNK_ROWS)
    else:
        reader = [read_csv_with_fallback(path, dtype=str)]

    for chunk in reader:
        chunk_idx += 1
        chunk.columns = [standardize_colname(c) for c in chunk.columns]
        total_input_rows += len(chunk)

        if tcol not in chunk.columns or station_col not in chunk.columns or gw_col not in chunk.columns:
            tcol2, s2, g2 = _detect_cols(chunk)
            tcol = tcol if tcol in chunk.columns else tcol2
            station_col = station_col if station_col in chunk.columns else s2
            gw_col = gw_col if gw_col in chunk.columns else g2

        chunk["timestamp"] = pd.to_datetime(chunk[tcol], dayfirst=True, errors="coerce")
        chunk["Station"] = chunk[station_col].astype(str).str.strip()
        chunk["groundwater_raw"] = pd.to_numeric(chunk[gw_col], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp", "Station"])
        # normalize to date (midnight) without timezone
        chunk["date"] = pd.to_datetime(chunk["timestamp"], errors="coerce").dt.normalize()
        total_clean_rows += len(chunk)

        # Duplicate timestamp detection within chunk before daily aggregation
        chunk_dups = int(chunk.duplicated(subset=["Station", "timestamp"]).sum())

        # Aggregate per station-date
        agg = (
            chunk.groupby(["Station", "date"])
            .agg(
                groundwater_mean=("groundwater_raw", "mean"),
                groundwater_min=("groundwater_raw", "min"),
                groundwater_max=("groundwater_raw", "max"),
                groundwater_std=("groundwater_raw", "std"),
                obs_count=("groundwater_raw", "count"),
            )
            .reset_index()
        )
        agg["groundwater_sum"] = agg["groundwater_mean"] * agg["obs_count"]

        tmpf = TMP_DIR / f"telemetry_clean_chunk_{chunk_idx:05d}.csv"
        agg.to_csv(tmpf, index=False)
        tmp_files.append(tmpf)

        logger.info(
            "Chunk %d processed | in_rows=%d clean_rows=%d dup_ts=%d mem=%.1fMB",
            chunk_idx,
            total_input_rows,
            total_clean_rows,
            chunk_dups,
            mem_usage_mb(),
        )

    # Safe merge of chunk-level daily aggregates
    parts = [pd.read_csv(f, parse_dates=["date"]) for f in tmp_files]
    all_daily = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if not all_daily.empty:
        grouped = (
            all_daily.groupby(["Station", "date"], as_index=False)
            .agg(
                groundwater_sum=("groundwater_sum", "sum"),
                obs_count=("obs_count", "sum"),
                groundwater_min=("groundwater_min", "min"),
                groundwater_max=("groundwater_max", "max"),
            )
            .sort_values(["Station", "date"])
        )
        grouped["groundwater_mean"] = grouped["groundwater_sum"] / grouped["obs_count"]
        final_daily = grouped[[
            "Station",
            "date",
            "groundwater_mean",
            "groundwater_min",
            "groundwater_max",
            "obs_count",
        ]]
    else:
        final_daily = pd.DataFrame(
            columns=[
                "Station",
                "date",
                "groundwater_mean",
                "groundwater_min",
                "groundwater_max",
                "obs_count",
            ]
        )

    out_daily = TMP_DIR / "telemetry_daily_aggregated.csv"
    final_daily.to_csv(out_daily, index=False)

    stats = {
        "telemetry_file_mb": round(size / (1024 * 1024), 2),
        "used_chunks": use_chunks,
        "chunk_count": chunk_idx,
        "input_rows": int(total_input_rows),
        "clean_rows": int(total_clean_rows),
        "daily_rows": int(len(final_daily)),
        "intermediate_chunk_files": len(tmp_files),
    }
    return out_daily, stats


def load_rainfall_manual(path):
    df = read_csv_with_fallback(path, dtype=str)
    df.columns = [standardize_colname(c) for c in df.columns]
    tcol = next((c for c in df.columns if "data_acquisition_time" in c.lower() or "timestamp" in c.lower()), None)
    if tcol is None:
        tcol = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), df.columns[0])

    station_col = next((c for c in df.columns if "station" in c.lower() or "site" in c.lower()), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    valcol = next((c for c in df.columns if "rain" in c.lower()), df.columns[-1])

    df["timestamp"] = pd.to_datetime(df[tcol], dayfirst=True, errors="coerce")
    df["Station"] = df[station_col].astype(str).str.strip()
    df["rainfall_mm"] = pd.to_numeric(df[valcol], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.normalize()

    rain_daily = (
        df.groupby(["Station", "date"], as_index=False)
        .agg(rainfall_mm=("rainfall_mm", "sum"))
        .sort_values(["Station", "date"])
    )
    rain_daily["cum7"] = (
        rain_daily.groupby("Station")["rainfall_mm"].rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rain_daily["cum30"] = (
        rain_daily.groupby("Station")["rainfall_mm"].rolling(30, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rain_daily["cum90"] = (
        rain_daily.groupby("Station")["rainfall_mm"].rolling(90, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    return rain_daily


def compute_station_quality(tel_daily):
    missing = {}
    constant_stations = []
    for st, g in tel_daily.groupby("Station"):
        g = g.sort_values("date")
        observed = g["date"].nunique()
        if observed == 0:
            continue
        expected = (g["date"].max() - g["date"].min()).days + 1
        missing_days = max(0, expected - observed)
        missing_frac = missing_days / expected if expected else np.nan
        missing[st] = {
            "observed_days": int(observed),
            "expected_days": int(expected),
            "missing_days": int(missing_days),
            "missing_frac": float(missing_frac),
        }

        vals = pd.to_numeric(g["groundwater_mean"], errors="coerce").dropna()
        if len(vals) > 1 and float(vals.std()) == 0.0:
            constant_stations.append(st)

    excessive_missing = [st for st, s in missing.items() if s["missing_frac"] > 0.5]
    return missing, excessive_missing, constant_stations


def build_master(tel_daily, seasonal, rain_daily):
    tel_daily = tel_daily.copy()
    tel_daily = tel_daily.rename(columns={"date": "Timestamp"})

    master_daily = tel_daily[[
        "Station",
        "Timestamp",
        "groundwater_mean",
        "groundwater_min",
        "groundwater_max",
        "obs_count",
    ]].copy()
    master_daily["freq"] = "daily"
    master_daily["Groundwater_Level_MBGL"] = master_daily["groundwater_mean"]

    master_seasonal = pd.DataFrame()
    if seasonal is not None and not seasonal.empty:
        master_seasonal = seasonal[["Station", "timestamp", "groundwater_raw", "season"]].copy()
        master_seasonal = master_seasonal.rename(
            columns={"timestamp": "Timestamp", "groundwater_raw": "Groundwater_Level_MBGL"}
        )
        master_seasonal["freq"] = "seasonal"

    master = pd.concat([master_daily, master_seasonal], ignore_index=True, sort=False)

    # Preserve ordering and station consistency
    master["Timestamp"] = pd.to_datetime(master["Timestamp"], errors="coerce")
    master = master.dropna(subset=["Station", "Timestamp"]).sort_values(["Station", "Timestamp"]).reset_index(drop=True)

    # Duplicate detection after merge
    dup_count = int(master.duplicated(subset=["Station", "Timestamp"], keep=False).sum())

    # Safe de-duplication to prevent duplicate station/timestamp records in output.
    if dup_count > 0:
        numeric_cols = [c for c in master.columns if pd.api.types.is_numeric_dtype(master[c])]
        agg_dict = {c: "mean" for c in numeric_cols}
        for c in master.columns:
            if c not in ("Station", "Timestamp") and c not in agg_dict:
                agg_dict[c] = "first"
        master = (
            master.groupby(["Station", "Timestamp"], as_index=False)
            .agg(agg_dict)
            .sort_values(["Station", "Timestamp"])
            .reset_index(drop=True)
        )

    # Hydrological features preserved
    master["prev_gw"] = master.groupby("Station")["Groundwater_Level_MBGL"].shift(1)
    master["gw_diff"] = master["prev_gw"] - master["Groundwater_Level_MBGL"]
    master["gw_roll_mean_7d"] = master.groupby("Station")["Groundwater_Level_MBGL"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    master["gw_roll_std_7d"] = master.groupby("Station")["Groundwater_Level_MBGL"].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    )
    master["gw_roll_mean_30d"] = master.groupby("Station")["Groundwater_Level_MBGL"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    master["target_gw_1d"] = master.groupby("Station")["Groundwater_Level_MBGL"].shift(-1)

    # Merge rainfall by same station/date where possible
    if rain_daily is not None and not rain_daily.empty:
        rain = rain_daily.copy()
        rain["date"] = pd.to_datetime(rain["date"], errors="coerce")
        master["date"] = pd.to_datetime(master["Timestamp"], errors="coerce").dt.normalize()
        master = master.merge(
            rain[["Station", "date", "rainfall_mm", "cum7", "cum30", "cum90"]],
            on=["Station", "date"],
            how="left",
        )

    return master, dup_count


def process():
    t0 = time.time()
    logger.info("Pipeline start | memory=%.1fMB", mem_usage_mb())

    atal_path = DATA_DIR / "Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv"
    tel_path = DATA_DIR / "gwl_tel_6_hourly_karnataka_ka_2021_2025.csv"
    rain_path = DATA_DIR / "rainfall_manual_daily_karnataka_ka_1991_2020.csv"

    seasonal = pd.DataFrame()
    if atal_path.exists():
        logger.info("Loading seasonal dataset")
        seasonal = load_atal_jal(atal_path)
        logger.info("Seasonal rows=%d | memory=%.1fMB", len(seasonal), mem_usage_mb())

    rain_daily = pd.DataFrame()
    if rain_path.exists():
        logger.info("Loading rainfall dataset")
        rain_daily = load_rainfall_manual(rain_path)
        logger.info("Rain daily rows=%d | memory=%.1fMB", len(rain_daily), mem_usage_mb())

    telemetry_daily_path = None
    telemetry_stats = {}
    tel_daily = pd.DataFrame()
    if tel_path.exists():
        telemetry_daily_path, telemetry_stats = process_telemetry_chunked(tel_path)
        tel_daily = pd.read_csv(telemetry_daily_path, parse_dates=["date"])
        logger.info("Telemetry daily rows=%d | memory=%.1fMB", len(tel_daily), mem_usage_mb())

    missing_stats = {}
    excessive_missing = []
    constant_stations = []
    if not tel_daily.empty:
        missing_stats, excessive_missing, constant_stations = compute_station_quality(tel_daily)

    master, duplicate_station_timestamp_rows = build_master(tel_daily, seasonal, rain_daily)

    rows_before_merge = int(telemetry_stats.get("daily_rows", 0) + len(seasonal))
    rows_after_merge = int(len(master))

    # Final ordering and consistent schema
    master = master.sort_values(["Station", "Timestamp"]).reset_index(drop=True)
    master["timestamp"] = pd.to_datetime(master["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    master = master.rename(columns={"Station": "station_id"})
    drop_cols = [c for c in ["Timestamp"] if c in master.columns]
    if drop_cols:
        master = master.drop(columns=drop_cols)

    master.to_csv(OUT_CSV, index=False)
    logger.info("Wrote master dataset: %s (rows=%d)", OUT_CSV, len(master))

    # Write text report
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("NEERA Preprocessing Report\n")
        f.write(f"Generated (UTC): {datetime.utcnow().isoformat()}Z\n\n")

        f.write("[Row Count Validation]\n")
        f.write(f"Telemetry input rows: {telemetry_stats.get('input_rows', 0)}\n")
        f.write(f"Telemetry cleaned rows: {telemetry_stats.get('clean_rows', 0)}\n")
        f.write(f"Telemetry daily rows: {telemetry_stats.get('daily_rows', 0)}\n")
        f.write(f"Seasonal rows: {len(seasonal)}\n")
        f.write(f"Rows before merge (daily+seasonal): {rows_before_merge}\n")
        f.write(f"Rows after merge (master): {rows_after_merge}\n\n")

        f.write("[Chunking and Memory]\n")
        f.write(f"Telemetry file MB: {telemetry_stats.get('telemetry_file_mb', 0)}\n")
        f.write(f"Used chunked mode (>50MB): {telemetry_stats.get('used_chunks', False)}\n")
        f.write(f"Chunk count: {telemetry_stats.get('chunk_count', 0)}\n")
        f.write(f"Intermediate chunk files: {telemetry_stats.get('intermediate_chunk_files', 0)}\n")
        f.write(f"Final memory usage MB: {mem_usage_mb():.2f}\n\n")

        f.write("[Quality Checks]\n")
        f.write(f"Duplicate station+timestamp rows in master: {duplicate_station_timestamp_rows}\n")
        f.write(f"Stations with excessive missing data (>50%): {excessive_missing}\n")
        f.write(f"Telemetry stations with constant-value behavior: {constant_stations}\n\n")

        f.write("[Hydrological Features Preserved]\n")
        f.write("- prev_gw\n")
        f.write("- gw_diff\n")
        f.write("- gw_roll_mean_7d\n")
        f.write("- gw_roll_std_7d\n")
        f.write("- gw_roll_mean_30d\n")
        f.write("- target_gw_1d\n")

    logger.info("Wrote preprocessing report: %s", REPORT_TXT)
    logger.info("Pipeline complete in %.1fs | memory=%.1fMB", time.time() - t0, mem_usage_mb())


if __name__ == "__main__":
    process()
