#!/usr/bin/env python3
"""NEERA FastAPI Hardened Deployment Service.

Exposes REST endpoints for:
- /health: checks API and model status
- /predict: uncertainty-aware groundwater prediction with schema checking,
            imputation, and physical plausibility verification.
"""

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import logging
import json
import traceback
import time
from logging.handlers import RotatingFileHandler
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware

from weather_service import WeatherService
from trend_forecaster import TrendForecaster
from alert_engine import AlertEngine
from geo_service import GeoService
from sensor_service import SensorService
from sustainability_engine import SustainabilityEngine
from crisis_forecaster import CrisisForecaster
from ai_advisor import AIAdvisor
from recommendation_engine import RecommendationEngine

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs/models/best_model_validated.pkl"
Q10_PATH = ROOT / "outputs/models/model_q10.pkl"
Q50_PATH = ROOT / "outputs/models/model_q50.pkl"
Q90_PATH = ROOT / "outputs/models/model_q90.pkl"
DATA_PATH = ROOT / "data/training_master_engineered.csv"
LOG_PATH = ROOT / "outputs/predictions/inference_log.csv"

MODEL_VERSION = "2.0.0 (CatBoost Validated)"

# Setup Structured Rotating Logging
LOG_DIR = ROOT / "outputs/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "neera.log"

logger = logging.getLogger("neera")
logger.setLevel(logging.INFO)

# Structured formatter
class StructuredJSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        return json.dumps(log_data)

handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
handler.setFormatter(StructuredJSONFormatter())
logger.addHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

def log_info(msg_dict: dict):
    logger.info(json.dumps(msg_dict))

def log_error(msg_dict: dict):
    logger.error(json.dumps(msg_dict))

# Request Timing Middleware
class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000.0
        
        log_info({
            "type": "request_timing",
            "path": request.url.path,
            "method": request.method,
            "duration_ms": round(process_time, 2),
            "status_code": response.status_code
        })
        
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="NEERA Hardened Groundwater Forecasting API",
    description="Machine learning API for seasonal groundwater level predictions in Karnataka, India, with uncertainty intervals.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestTimingMiddleware)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = {
        "type": "unhandled_exception",
        "path": request.url.path,
        "method": request.method,
        "error": str(exc),
        "traceback": traceback.format_exc()
    }
    log_error(error_detail)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal system error occurred. Please consult backend logs."}
    )

# Enable CORS for Next.js frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models, database, and alerting services
models = {}
database_df = None
weather_service = None
trend_forecaster = None
alert_engine = None
geo_service = None
sensor_service = None
sustainability_engine = None
crisis_forecaster = None
ai_advisor = None
recommendation_engine = None

@app.on_event("startup")
def startup_event():
    global models, database_df, weather_service, trend_forecaster, alert_engine, geo_service, sensor_service
    global sustainability_engine, crisis_forecaster, ai_advisor, recommendation_engine
    
    # Environment variable startup checks
    owm_key = os.getenv("OPENWEATHER_API_KEY")
    mqtt_broker = os.getenv("MQTT_BROKER")
    mqtt_port = os.getenv("MQTT_PORT")
    
    if not owm_key:
        print("WARNING: OPENWEATHER_API_KEY environment variable is missing. "
              "Weather service will run in keyless Open-Meteo mode.")
    if not mqtt_broker:
        print("WARNING: MQTT_BROKER environment variable is missing. Telemetry service will use local SQLite ingress only.")
    if not mqtt_port:
        print("WARNING: MQTT_PORT environment variable is missing.")
        
    # Load all models
    for name, path in [("cb", MODEL_PATH), ("q10", Q10_PATH), ("q50", Q50_PATH), ("q90", Q90_PATH)]:
        if not path.exists():
            raise RuntimeError(f"Required model file not found at {path}. Please run training first.")
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    print("Models loaded successfully.")
    
    # Load database for lookups and imputation statistics
    if DATA_PATH.exists():
        database_df = pd.read_csv(DATA_PATH)
        database_df["date"] = pd.to_datetime(database_df["date"])
        database_df["year"] = database_df["date"].dt.year
        print(f"Database loaded successfully with {len(database_df)} records.")
    else:
        print(f"Warning: Database not found at {DATA_PATH}. Lookup predictions will be disabled.")

    # Initialize near-real-time service layers
    weather_service = WeatherService(database_path=DATA_PATH)
    trend_forecaster = TrendForecaster(weather_service)
    alert_engine = AlertEngine()
    geo_service = GeoService(database_path=DATA_PATH)
    sensor_service = SensorService()
    
    # Initialize sustainability & crisis engines
    sustainability_engine = SustainabilityEngine(database_path=DATA_PATH)
    crisis_forecaster = CrisisForecaster(warning_threshold=30.0, critical_threshold=50.0, collapse_threshold=70.0)
    ai_advisor = AIAdvisor()
    recommendation_engine = RecommendationEngine()
    
    print("Near-real-time service layers, geographic resolvers, and sustainability/crisis engines initialized successfully.")

# Pydantic request models
class FeaturesPayload(BaseModel):
    Groundwater_Level_MBGL: Optional[float] = Field(None, description="Current groundwater level (MBGL)")
    prev_gw: Optional[float] = Field(None, description="Previous groundwater level (MBGL)")
    gw_diff: Optional[float] = Field(None, description="Difference between current and previous groundwater levels")
    gw_roll_mean_7obs: Optional[float] = Field(None, description="Rolling mean of last 7 observations")
    gw_roll_std_7obs: Optional[float] = Field(None, description="Rolling standard deviation of last 7 observations")
    gw_roll_mean_30obs: Optional[float] = Field(None, description="Rolling mean of last 30 observations")
    rainfall_30d: Optional[float] = Field(None, description="Total rainfall in the last 30 days")
    rainfall_90d: Optional[float] = Field(None, description="Total rainfall in the last 90 days")
    rainfall_180d: Optional[float] = Field(None, description="Total rainfall in the last 180 days")
    district_rainfall_30d: Optional[float] = Field(None, description="District rainfall in the last 30 days")
    district_rainfall_90d: Optional[float] = Field(None, description="District rainfall in the last 90 days")
    district_rainfall_180d: Optional[float] = Field(None, description="District rainfall in the last 180 days")
    state_rainfall_30d: Optional[float] = Field(None, description="State rainfall in the last 30 days")
    state_rainfall_90d: Optional[float] = Field(None, description="State rainfall in the last 90 days")
    state_rainfall_180d: Optional[float] = Field(None, description="State rainfall in the last 180 days")
    season_sin: Optional[float] = Field(None, description="Sine component of seasonal cycle")
    season_cos: Optional[float] = Field(None, description="Cosine component of seasonal cycle")
    rainfall_station_observation_count: Optional[float] = Field(None, description="Observation count for rainfall station")
    rainfall_window_completeness_30d: Optional[float] = Field(None, description="Window completeness for 30d rainfall")
    rainfall_window_completeness_90d: Optional[float] = Field(None, description="Window completeness for 90d rainfall")
    rainfall_window_completeness_180d: Optional[float] = Field(None, description="Window completeness for 180d rainfall")
    rainfall_distance_km: Optional[float] = Field(None, description="Distance to rainfall station (km)")
    rainfall_district_match: Optional[float] = Field(None, description="Rainfall district match indicator")
    rainfall_fallback_used: Optional[int] = Field(None, description="Indicator if rainfall fallback was used")
    gw_jump_flag: Optional[bool] = Field(None, description="Groundwater jump flag")
    rainfall_extreme_flag: Optional[bool] = Field(None, description="Rainfall extreme flag")
    effective_rainfall_30d: Optional[float] = Field(None, description="Effective rainfall in the last 30 days")
    effective_rainfall_90d: Optional[float] = Field(None, description="Effective rainfall in the last 90 days")
    effective_rainfall_180d: Optional[float] = Field(None, description="Effective rainfall in the last 180 days")
    latitude: Optional[float] = Field(None, description="Station latitude")
    longitude: Optional[float] = Field(None, description="Station longitude")
    lag_1: Optional[float] = Field(None, description="Lag 1 groundwater observation")
    lag_2: Optional[float] = Field(None, description="Lag 2 groundwater observation")
    lag_3: Optional[float] = Field(None, description="Lag 3 groundwater observation")
    lag_4: Optional[float] = Field(None, description="Lag 4 groundwater observation")
    gw_expanding_mean: Optional[float] = Field(None, description="Expanding mean of groundwater levels")
    gw_expanding_std: Optional[float] = Field(None, description="Expanding standard deviation of groundwater levels")
    gw_ewm_mean_span3: Optional[float] = Field(None, description="EWM mean with span 3")
    gw_ewm_mean_span5: Optional[float] = Field(None, description="EWM mean with span 5")
    gw_seasonal_diff_2: Optional[float] = Field(None, description="Groundwater seasonal difference")
    gw_yoy_diff: Optional[float] = Field(None, description="Groundwater year-over-year difference")
    rainfall_intensity_30d: Optional[float] = Field(None, description="Rainfall intensity in last 30 days")
    rainfall_intensity_90d: Optional[float] = Field(None, description="Rainfall intensity in last 90 days")
    rainfall_intensity_180d: Optional[float] = Field(None, description="Rainfall intensity in last 180 days")
    rainfall_anomaly_30d: Optional[float] = Field(None, description="Rainfall anomaly in last 30 days")
    rainfall_anomaly_90d: Optional[float] = Field(None, description="Rainfall anomaly in last 90 days")
    rainfall_anomaly_180d: Optional[float] = Field(None, description="Rainfall anomaly in last 180 days")
    rainfall_ratio_30d_180d: Optional[float] = Field(None, description="Ratio of 30d to 180d rainfall")
    rainfall_ratio_90d_180d: Optional[float] = Field(None, description="Ratio of 90d to 180d rainfall")
    drought_indicator_180d: Optional[float] = Field(None, description="Drought indicator based on 180d rainfall")
    recharge_efficiency_proxy_90d: Optional[float] = Field(None, description="Recharge efficiency proxy for 90d")
    recharge_efficiency_proxy_180d: Optional[float] = Field(None, description="Recharge efficiency proxy for 180d")
    rainfall_trend_slope: Optional[float] = Field(None, description="Slope of rainfall trend")
    source_reliability_score: Optional[float] = Field(None, description="Source reliability score")
    season: Optional[str] = Field(None, description="Season name")
    rainfall_source_type: Optional[str] = Field(None, description="Rainfall source type (e.g. 'district', 'station')")
    mapping_method: Optional[str] = Field(None, description="Mapping method used")
    spatial_cluster: Optional[int] = Field(None, description="Spatial cluster ID")

class PredictionRequest(BaseModel):
    station_id: Optional[str] = Field(None, description="Station ID to look up features for (e.g. '020109B')")
    date: Optional[str] = Field(None, description="Specific date to look up features from (YYYY-MM-DD), defaults to latest")
    features: Optional[FeaturesPayload] = Field(None, description="Raw feature dict matching model inputs if station_id is not provided")

class PredictionResponse(BaseModel):
    station_id: str
    forecast_date: str
    predicted_p50_gw_mbgl: float
    predicted_p10_gw_mbgl: float
    predicted_p90_gw_mbgl: float
    interval_uncertainty_meters: float
    lookup_used: bool
    warnings: List[str]
    model_version: str

class ChatCopilotRequest(BaseModel):
    user_question: str
    station_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    query: Optional[str] = None

@app.get("/health")
def health_check():
    status = {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "models_loaded": len(models) == 4,
        "database_loaded": database_df is not None,
    }
    if not status["models_loaded"]:
        raise HTTPException(status_code=503, detail=status)
    return status

def impute_missing_features(df, features, db_df=None):
    df = df.copy()
    
    # 1. Determine baseline impute values
    impute_values = {}
    cat_cols = ["season", "rainfall_source_type", "mapping_method", "spatial_cluster"]
    
    if db_df is not None:
        train_df = db_df[db_df["year"] <= 2019]
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
        
    if (prediction < prev_gw - 2.0) and (rain_180 < 50.0):
        warnings.append("SUSPICIOUS_RECHARGE: Model predicts water table rise >2m despite extremely low rain (<50mm).")
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

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global models, database_df
    
    if len(models) != 4:
        raise HTTPException(status_code=503, detail="Models are not fully loaded.")
        
    features_list = getattr(models["cb"], "feature_names_", None)
    if features_list is None:
        raise HTTPException(status_code=500, detail="Could not extract feature schema from model.")
    features_list = list(features_list)
        
    # --- Mode 1: Lookup via Station ID ---
    if request.station_id:
        if database_df is None:
            raise HTTPException(status_code=501, detail="Station database is unavailable on this instance.")
            
        station_rows = database_df[database_df["station_id"] == request.station_id]
        if len(station_rows) == 0:
            raise HTTPException(status_code=404, detail=f"Station ID '{request.station_id}' not found.")
            
        if request.date:
            try:
                target_date = pd.to_datetime(request.date)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid date format '{request.date}'. Use YYYY-MM-DD.")
            station_rows["date_diff"] = (station_rows["date"] - target_date).abs()
            row = station_rows.sort_values(by="date_diff").iloc[0]
        else:
            row = station_rows.sort_values(by="date", ascending=False).iloc[0]
            
        input_df = pd.DataFrame([row])
        X = impute_missing_features(input_df, features_list, database_df)
        
        pred_val = float(models["cb"].predict(X)[0])
        p10 = float(models["q10"].predict(X)[0])
        p90 = float(models["q90"].predict(X)[0])
        p10 = min(p10, pred_val)
        p90 = max(p90, pred_val)
        
        warnings = run_physical_plausibility_checks(row, pred_val)
        log_prediction_history(row["station_id"], row["date"].strftime("%Y-%m-%d"), pred_val, p10, p90, warnings)
        
        return PredictionResponse(
            station_id=row["station_id"],
            forecast_date=row["date"].strftime("%Y-%m-%d"),
            predicted_p50_gw_mbgl=round(pred_val, 4),
            predicted_p10_gw_mbgl=round(p10, 4),
            predicted_p90_gw_mbgl=round(p90, 4),
            interval_uncertainty_meters=round(p90 - p10, 4),
            lookup_used=True,
            warnings=warnings,
            model_version=MODEL_VERSION
        )
        
    # --- Mode 2: Direct features payload ---
    elif request.features:
        features_dict = request.features.model_dump()
        input_df = pd.DataFrame([features_dict])
        X = impute_missing_features(input_df, features_list, database_df)
        
        pred_val = float(models["cb"].predict(X)[0])
        p10 = float(models["q10"].predict(X)[0])
        p90 = float(models["q90"].predict(X)[0])
        p10 = min(p10, pred_val)
        p90 = max(p90, pred_val)
        
        warnings = run_physical_plausibility_checks(features_dict, pred_val)
        log_prediction_history("CUSTOM_PAYLOAD", "CUSTOM_DATE", pred_val, p10, p90, warnings)
        
        return PredictionResponse(
            station_id="CUSTOM_PAYLOAD",
            forecast_date="CUSTOM_DATE",
            predicted_p50_gw_mbgl=round(pred_val, 4),
            predicted_p10_gw_mbgl=round(p10, 4),
            predicted_p90_gw_mbgl=round(p90, 4),
            interval_uncertainty_meters=round(p90 - p10, 4),
            lookup_used=False,
            warnings=warnings,
            model_version=MODEL_VERSION
        )
        
    else:
        raise HTTPException(
            status_code=400,
            detail="Request must contain either 'station_id' or 'features' object."
        )

@app.get("/stations", response_model=List[str])
def get_stations():
    global database_df
    if database_df is None:
        raise HTTPException(status_code=501, detail="Station database is unavailable on this instance.")
    return sorted(database_df["station_id"].unique().tolist())

@app.get("/stations/{station_id}/history")
def get_station_history(station_id: str, limit: int = 10):
    global database_df
    if database_df is None:
        raise HTTPException(status_code=501, detail="Station database is unavailable on this instance.")
    station_rows = database_df[database_df["station_id"] == station_id]
    if len(station_rows) == 0:
        raise HTTPException(status_code=404, detail=f"Station ID '{station_id}' not found.")
    history = station_rows.sort_values(by="date", ascending=False).head(limit)
    records = history.to_dict(orient="records")
    for r in records:
        if isinstance(r["date"], pd.Timestamp):
            r["date"] = r["date"].strftime("%Y-%m-%d")
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
    return records

@app.get("/api/weather")
def get_weather(station_id: str):
    global weather_service
    if weather_service is None:
        raise HTTPException(status_code=503, detail="Weather service is not initialized.")
    try:
        return weather_service.get_weather_by_station(station_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast")
def get_forecast(
    station_id: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    query: Optional[str] = None
):
    global trend_forecaster, geo_service, database_df, weather_service, alert_engine
    global sustainability_engine, crisis_forecaster, ai_advisor, recommendation_engine
    
    if trend_forecaster is None or database_df is None or geo_service is None or weather_service is None or alert_engine is None:
        raise HTTPException(status_code=503, detail="Forecast services or database are unavailable.")

    # 1. Resolve coordinates
    resolved_lat, resolved_lon = None, None
    resolved_name = ""

    if query:
        try:
            geo_res = geo_service.geocode(query)
            resolved_lat = geo_res["lat"]
            resolved_lon = geo_res["lon"]
            resolved_name = geo_res["display_name"]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to geocode query '{query}': {e}")
    elif lat is not None and lon is not None:
        resolved_lat = lat
        resolved_lon = lon
        resolved_name = f"Coordinates ({lat:.4f}, {lon:.4f})"
    elif station_id:
        station_meta = next((s for s in geo_service.stations_coords if s["station_id"] == station_id), None)
        if station_meta:
            resolved_lat = station_meta["latitude"]
            resolved_lon = station_meta["longitude"]
            resolved_name = f"Station {station_id}"
        else:
            raise HTTPException(status_code=404, detail=f"Station ID '{station_id}' not found.")
    else:
        raise HTTPException(status_code=400, detail="Must provide 'station_id', 'query', or 'lat' and 'lon'.")

    # 2. Resolve nearest station
    mapping = geo_service.resolve_nearest_station(resolved_lat, resolved_lon)
    if not mapping:
        raise HTTPException(status_code=404, detail="Could not map location to any telemetry station.")

    if mapping.get("disable_prediction", False):
        return {
            "error": "No nearby monitoring station within 250km.",
            "disable_prediction": True,
            "resolved_location": resolved_name,
            "nearest_station_id": mapping["station_id"],
            "distance_km": mapping["distance_km"]
        }

    target_station_id = mapping["station_id"]
    station_rows = database_df[database_df["station_id"] == target_station_id]
    if len(station_rows) == 0:
        raise HTTPException(status_code=404, detail=f"Nearest station '{target_station_id}' has no historical data.")

    # 3. Fetch live weather & forecast
    try:
        weather = weather_service.get_weather_by_coordinates(resolved_lat, resolved_lon, target_station_id)
    except Exception as e:
        print(f"Failed to fetch weather by coordinates: {e}")
        weather = weather_service.get_weather_by_station(target_station_id)

    # 4. Fetch historical groundwater
    history = station_rows.sort_values(by="date", ascending=False)
    latest_row = history.iloc[0]
    current_gw = float(latest_row["Groundwater_Level_MBGL"])
    
    prev_gw = None
    if len(history) > 1:
        prev_gw = float(history.iloc[1]["Groundwater_Level_MBGL"])
    else:
        prev_gw = float(latest_row.get("prev_gw", current_gw))

    recent_rain_180d = float(latest_row.get("effective_rainfall_180d", 200.0))

    try:
        # Run forecast daily trajectory
        fc = trend_forecaster.forecast_short_term(
            station_id=target_station_id,
            current_gw=current_gw,
            prev_gw=prev_gw,
            recent_rain_180d=recent_rain_180d
        )
        
        # Evaluate alert level
        alert = alert_engine.evaluate_station_alert(fc)
        
        # Inject coordinates details for map rendering
        mapping["latitude"] = resolved_lat
        mapping["longitude"] = resolved_lon
        
        # Compute sustainability metrics
        sustainability = sustainability_engine.compute_sustainability(
            station_id=target_station_id,
            current_gw=current_gw,
            forecast_rain_7d=fc["forecast_rainfall_accumulation_7d"],
            depletion_rate=fc["depletion_rate_m_day"],
            recent_rain_180d=recent_rain_180d
        )
        
        # Calculate crisis timeline
        crisis = crisis_forecaster.calculate_crisis_timeline(
            current_gw=current_gw,
            depletion_rate=fc["depletion_rate_m_day"],
            forecast_rain_7d=fc["forecast_rainfall_accumulation_7d"],
            long_term_slope=sustainability["metrics"]["long_term_depletion_slope_m_yr"]
        )
        
        # Get AI commentary/advisory
        weather_desc = weather.get("current", {}).get("weather_description", "clear sky")
        rdi = sustainability["metrics"]["rainfall_deficit_index"]
        
        ai_commentary = ai_advisor.get_sustainability_advisory(
            station_id=target_station_id,
            gw_level=current_gw,
            status=sustainability["sustainability_status"],
            days_warn=crisis["days_to_warning"],
            days_crit=crisis["days_to_critical"],
            days_coll=crisis["days_to_collapse"],
            sust_score=sustainability["sustainability_score"],
            weather_desc=weather_desc,
            rdi=rdi
        )
        
        # Get recommendations
        temp = weather.get("current", {}).get("temperature", 25.0)
        recs = recommendation_engine.get_recommendations(
            status=sustainability["sustainability_status"],
            weather_desc=weather_desc,
            rdi=rdi,
            temp=temp
        )
        
        return {
            "disable_prediction": False,
            "resolved_location": resolved_name,
            "nearest_station": mapping,
            "weather": weather,
            "forecast": fc,
            "alert": alert,
            "sustainability": sustainability,
            "crisis": crisis,
            "ai_commentary": ai_commentary,
            "recommendations": recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")

@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    global trend_forecaster, alert_engine, database_df
    if trend_forecaster is None or alert_engine is None or database_df is None:
        raise HTTPException(status_code=503, detail="Alert services or database are unavailable.")
    
    # Sort by date and get the latest observation for each station
    latest_records = database_df.sort_values(by="date").groupby("station_id").last().reset_index()
    # Sort by Groundwater_Level_MBGL descending to evaluate deepest (highest risk) first
    latest_records = latest_records.sort_values(by="Groundwater_Level_MBGL", ascending=False)
    
    active_alerts = []
    count = 0
    for _, row in latest_records.iterrows():
        if count >= limit:
            break
        station_id = row["station_id"]
        current_gw = float(row["Groundwater_Level_MBGL"])
        prev_gw = float(row.get("prev_gw", current_gw))
        recent_rain_180d = float(row.get("effective_rainfall_180d", 200.0))
        
        try:
            fc = trend_forecaster.forecast_short_term(
                station_id=station_id,
                current_gw=current_gw,
                prev_gw=prev_gw,
                recent_rain_180d=recent_rain_180d
            )
            alert = alert_engine.evaluate_station_alert(fc)
            if alert["alert_level"] != "SAFE":
                active_alerts.append(alert)
                count += 1
        except Exception:
            continue
            
    return active_alerts

@app.get("/api/risk-summary")
def get_risk_summary():
    global database_df, alert_engine, trend_forecaster
    if database_df is None or alert_engine is None or trend_forecaster is None:
        raise HTTPException(status_code=503, detail="Database or services are unavailable.")
    
    # Group by station to get latest state for each station
    latest_records = database_df.sort_values(by="date").groupby("station_id").last().reset_index()
    
    # Summarize metrics:
    cluster_counts = latest_records["spatial_cluster"].value_counts().to_dict()
    cluster_means = latest_records.groupby("spatial_cluster")["Groundwater_Level_MBGL"].mean().to_dict()
    
    # Find extreme hotspots (depth > 30m)
    hotspots = latest_records[latest_records["Groundwater_Level_MBGL"] > 30.0]
    
    # Get metadata for map markers (first 150 stations for visualization density)
    map_markers = []
    for _, row in latest_records.head(150).iterrows():
        map_markers.append({
            "station_id": row["station_id"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "depth": float(row["Groundwater_Level_MBGL"]),
            "cluster": int(row["spatial_cluster"])
        })
    
    summary = {
        "total_stations": len(latest_records),
        "deep_wells_count": len(hotspots),
        "average_depth_mbgl": round(float(latest_records["Groundwater_Level_MBGL"].mean()), 2),
        "max_depth_mbgl": round(float(latest_records["Groundwater_Level_MBGL"].max()), 2),
        "cluster_stats": [
            {
                "cluster_id": int(cid),
                "station_count": int(cluster_counts.get(cid, 0)),
                "avg_depth_mbgl": round(float(avg_depth), 2),
                "risk_level": "HIGH" if avg_depth > 20.0 else "MEDIUM" if avg_depth > 10.0 else "LOW"
            }
            for cid, avg_depth in cluster_means.items()
        ],
        "map_markers": map_markers
    }
    return summary

@app.get("/api/geocode")
def get_geocode(query: str):
    global geo_service
    if geo_service is None:
        raise HTTPException(status_code=503, detail="Geo service not initialized.")
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters long.")
    try:
        res = geo_service.geocode(query)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/geocode/autocomplete")
def get_geocode_autocomplete(query: str):
    global geo_service
    if geo_service is None:
        raise HTTPException(status_code=503, detail="Geo service not initialized.")
    try:
        return geo_service.autocomplete(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reverse-geocode")
def get_reverse_geocode(lat: float, lon: float):
    global geo_service
    if geo_service is None:
        raise HTTPException(status_code=503, detail="Geo service not initialized.")
    try:
        address = geo_service.reverse_geocode(lat, lon)
        return {"display_name": address}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/nearest-station")
def get_nearest_station(lat: float, lon: float):
    global geo_service
    if geo_service is None:
        raise HTTPException(status_code=503, detail="Geo service not initialized.")
    res = geo_service.resolve_nearest_station(lat, lon)
    if not res:
        raise HTTPException(status_code=404, detail="Could not resolve nearest station.")
    return res

@app.get("/api/environmental-risk")
def get_environmental_risk(station_id: str):
    global weather_service, database_df
    if weather_service is None or database_df is None:
        raise HTTPException(status_code=503, detail="Services not initialized.")
    
    station_rows = database_df[database_df["station_id"] == station_id]
    if len(station_rows) == 0:
        raise HTTPException(status_code=404, detail="Station not found.")
    
    latest_row = station_rows.sort_values(by="date", ascending=False).iloc[0]
    current_gw = float(latest_row["Groundwater_Level_MBGL"])
    
    # Fetch weather forecast
    weather = weather_service.get_weather_by_station(station_id)
    daily = weather.get("daily", [])
    forecast_rain = sum([day.get("rainfall", 0.0) for day in daily])
    max_temp = max([day.get("temperature", 25.0) for day in daily]) if daily else 28.0
    
    # Derived indicators
    dry_spell = "ACTIVE" if forecast_rain < 2.0 else "INACTIVE"
    heatwave = "CRITICAL" if max_temp > 40.0 else "WARNING" if max_temp > 37.0 else "SAFE"
    
    # Recharge potential
    recharge_pot = "HIGH" if (current_gw > 5.0 and forecast_rain > 40.0) else "MEDIUM" if forecast_rain > 15.0 else "LOW"
    
    import time
    return {
        "station_id": station_id,
        "recent_depth_mbgl": current_gw,
        "forecast_rainfall_7d_mm": round(forecast_rain, 2),
        "heatwave_stress": heatwave,
        "dry_spell_risk": dry_spell,
        "recharge_potential": recharge_pot,
        "derived_at": time.time()
    }

class SensorRegistrationRequest(BaseModel):
    station_id: str
    sensor_id: str
    model: str = Field("NEERA-IoT-Alpha", description="IoT sensor model number")
    latitude: float
    longitude: float

class SensorIngestPayload(BaseModel):
    sensor_id: str
    lat: float
    lon: float
    water_level: float
    battery: float
    temperature: float
    humidity: float
    timestamp: Optional[str] = None

@app.post("/api/sensors/register")
def post_sensor_register(request: SensorRegistrationRequest):
    global sensor_service
    if sensor_service is None:
        raise HTTPException(status_code=503, detail="Sensor service not initialized.")
    try:
        res = sensor_service.register_sensor(
            sensor_id=request.sensor_id,
            station_id=request.station_id,
            latitude=request.latitude,
            longitude=request.longitude,
            model=request.model
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sensors/ingest")
def post_sensor_ingest(payload: SensorIngestPayload):
    global sensor_service
    if sensor_service is None:
        raise HTTPException(status_code=503, detail="Sensor service not initialized.")
    try:
        res = sensor_service.ingest_telemetry(payload.model_dump())
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sensors/latest")
def get_sensors_latest():
    global sensor_service
    if sensor_service is None:
        raise HTTPException(status_code=503, detail="Sensor service not initialized.")
    try:
        return sensor_service.list_sensors()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sensors/history")
def get_sensor_history_endpoint(sensor_id: str, limit: int = 50):
    global sensor_service
    if sensor_service is None:
        raise HTTPException(status_code=503, detail="Sensor service not initialized.")
    try:
        res = sensor_service.get_sensor_history(sensor_id, limit)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/copilot/chat")
def post_copilot_chat(req: ChatCopilotRequest):
    global trend_forecaster, database_df, geo_service, weather_service, sustainability_engine, crisis_forecaster, ai_advisor
    if not ai_advisor:
        raise HTTPException(status_code=503, detail="Copilot chat service is unavailable.")

    # We try to get context if station or coordinates are specified
    context = {}
    
    # Resolve the station id if we can
    target_station_id = req.station_id
    resolved_lat, resolved_lon = req.lat, req.lon
    resolved_name = ""

    # Reuse resolution logic if query/lat/lon/station_id provided
    try:
        if req.query:
            geo_res = geo_service.geocode(req.query)
            resolved_lat = geo_res["lat"]
            resolved_lon = geo_res["lon"]
            resolved_name = geo_res["display_name"]
        elif req.lat is not None and req.lon is not None:
            resolved_lat = req.lat
            resolved_lon = req.lon
            resolved_name = f"Coordinates ({req.lat:.4f}, {req.lon:.4f})"
        
        if resolved_lat is not None and resolved_lon is not None:
            mapping = geo_service.resolve_nearest_station(resolved_lat, resolved_lon)
            if mapping and not mapping.get("disable_prediction", False):
                target_station_id = mapping["station_id"]
        
        if not target_station_id and database_df is not None and not database_df.empty:
            # Default to the first station in the database if nothing is specified or found
            target_station_id = str(database_df["station_id"].iloc[0])
            resolved_name = f"Station {target_station_id} (Default)"
            
        if target_station_id and database_df is not None:
            station_rows = database_df[database_df["station_id"] == target_station_id]
            if len(station_rows) > 0:
                history = station_rows.sort_values(by="date", ascending=False)
                latest_row = history.iloc[0]
                current_gw = float(latest_row["Groundwater_Level_MBGL"])
                
                prev_gw = None
                if len(history) > 1:
                    prev_gw = float(history.iloc[1]["Groundwater_Level_MBGL"])
                else:
                    prev_gw = float(latest_row.get("prev_gw", current_gw))

                recent_rain_180d = float(latest_row.get("effective_rainfall_180d", 200.0))
                
                # Fetch weather
                try:
                    if resolved_lat is not None and resolved_lon is not None:
                        weather = weather_service.get_weather_by_coordinates(resolved_lat, resolved_lon, target_station_id)
                    else:
                        weather = weather_service.get_weather_by_station(target_station_id)
                except Exception:
                    weather = weather_service.get_weather_by_station(target_station_id)

                fc = trend_forecaster.forecast_short_term(
                    station_id=target_station_id,
                    current_gw=current_gw,
                    prev_gw=prev_gw,
                    recent_rain_180d=recent_rain_180d
                )
                
                sust = sustainability_engine.compute_sustainability(
                    station_id=target_station_id,
                    current_gw=current_gw,
                    forecast_rain_7d=fc["forecast_rainfall_accumulation_7d"],
                    depletion_rate=fc["depletion_rate_m_day"],
                    recent_rain_180d=recent_rain_180d
                )
                
                crisis = crisis_forecaster.calculate_crisis_timeline(
                    current_gw=current_gw,
                    depletion_rate=fc["depletion_rate_m_day"],
                    forecast_rain_7d=fc["forecast_rainfall_accumulation_7d"],
                    long_term_slope=sust["metrics"]["long_term_depletion_slope_m_yr"]
                )
                
                context = {
                    "station_id": target_station_id,
                    "location_name": resolved_name or f"Station {target_station_id}",
                    "current_groundwater_mbgl": current_gw,
                    "sustainability_score": sust["sustainability_score"],
                    "sustainability_status": sust["sustainability_status"],
                    "long_term_depletion_slope_m_yr": sust["metrics"]["long_term_depletion_slope_m_yr"],
                    "weather": weather.get("current", {}),
                    "days_to_warning_threshold": crisis["days_to_warning"],
                    "days_to_critical_threshold": crisis["days_to_critical"],
                    "days_to_collapse_threshold": crisis["days_to_collapse"],
                    "depletion_acceleration": crisis["depletion_acceleration"]
                }
    except Exception as e:
        print(f"Error compiling context for copilot chat: {e}")
        # Proceed with empty/partial context if resolution failed
        
    answer = ai_advisor.answer_copilot_chat(context, req.user_question)
    return {
        "answer": answer,
        "context_used": context
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
