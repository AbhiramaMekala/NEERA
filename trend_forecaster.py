import time
import json
import numpy as np
from pathlib import Path
from weather_service import WeatherService

class TrendForecaster:
    def __init__(self, weather_service=None):
        self.weather_service = weather_service or WeatherService()

    def forecast_short_term(self, station_id, current_gw, prev_gw=None, recent_rain_180d=None):
        """Generates 3-day, 7-day, and 14-day groundwater trajectories and stress scores.
        
        Args:
            station_id (str): The station code.
            current_gw (float): Current water table level in MBGL.
            prev_gw (float, optional): Previous water table level in MBGL.
            recent_rain_180d (float, optional): Long-term rainfall in mm.
        """
        # Fetch weather forecast
        weather = self.weather_service.get_weather_by_station(station_id)
        daily_forecast = weather.get("daily", [])
        
        # Calculate daily depletion rate
        # Dry seasons and deep aquifers deplete faster
        current_month = time.localtime().tm_mon
        is_dry_season = not (6 <= current_month <= 9)
        
        # Base daily depletion rate (MBGL increases = table drops)
        base_depletion = 0.02 if is_dry_season else 0.005
        # Deep aquifers deplete faster due to extraction pressure
        depth_multiplier = max(1.0, current_gw / 30.0)
        depletion_rate = base_depletion * depth_multiplier
        
        # Calculate recent rate of change from historical lag
        trend_rate = 0.0
        if prev_gw is not None:
            # Assumes ~120 days between seasons
            trend_rate = (current_gw - prev_gw) / 120.0
            # Limit extreme values to prevent runaway predictions
            trend_rate = max(-0.1, min(0.1, trend_rate))

        # Calculate forecast rainfall accumulation
        forecast_rain = [day.get("rainfall", 0.0) for day in daily_forecast]
        total_forecast_rain = sum(forecast_rain)

        # Infiltration/Recharge model
        # Sandy/alluvial soils recharge faster, deep clay/aquifers recharge slower
        recharge_coeff = 0.08 if current_gw < 20 else 0.02
        recharge_lag = 2 # days delay for surface rain to hit water table

        # Generate daily projections for 14 days
        dates = []
        p50_trajectory = []
        p10_trajectory = []
        p90_trajectory = []
        
        now_epoch = int(time.time())
        running_gw = current_gw

        for day in range(1, 15):
            day_epoch = now_epoch + day * 86400
            date_str = time.strftime("%Y-%m-%d", time.localtime(day_epoch))
            dates.append(date_str)
            
            # Sum rain up to day - lag
            rain_idx = max(0, day - recharge_lag)
            rain_infiltrated = sum(forecast_rain[:rain_idx]) * recharge_coeff
            
            # Current projected value
            # MBGL decreases (closer to surface) with recharge, increases (deeper) with depletion
            day_gw = current_gw + (day * depletion_rate) + (day * trend_rate) - rain_infiltrated
            # Physical bounds constraint (cannot go above surface 0.0)
            day_gw = max(0.0, day_gw)
            p50_trajectory.append(round(day_gw, 4))
            
            # Uncertainty scale increases with time horizon (standard error expands)
            # Standard error model: 0.2m + 0.15m * sqrt(day)
            std_err = 0.2 + 0.15 * np.sqrt(day)
            p10 = max(0.0, day_gw - 1.28 * std_err)  # 10th percentile (shallower)
            p90 = day_gw + 1.28 * std_err            # 90th percentile (deeper)
            
            p10_trajectory.append(round(p10, 4))
            p90_trajectory.append(round(p90, 4))

        # Compile outputs at key horizons (3d, 7d, 14d)
        projections = {
            "3_day": {
                "date": dates[2],
                "p50": p50_trajectory[2],
                "p10": p10_trajectory[2],
                "p90": p90_trajectory[2]
            },
            "7_day": {
                "date": dates[6],
                "p50": p50_trajectory[6],
                "p10": p10_trajectory[6],
                "p90": p90_trajectory[6]
            },
            "14_day": {
                "date": dates[13],
                "p50": p50_trajectory[13],
                "p10": p10_trajectory[13],
                "p90": p90_trajectory[13]
            }
        }

        # Calculate Groundwater Stress Score (0 to 100)
        # Components:
        # 1. Depth penalty: deep aquifers represent higher stress (maxed at 60m)
        depth_score = min(100.0, (current_gw / 50.0) * 100.0)
        # 2. Depletion trend: positive trend_rate is depleting (water table gets deeper)
        trend_score = min(100.0, max(0.0, trend_rate * 1000.0))  # e.g., 0.05m/day depletion -> 50 score
        # 3. Rainfall deficit penalty: if 180d rain is low (drought proxy)
        rain_val = recent_rain_180d if recent_rain_180d is not None else 200.0
        rain_score = min(100.0, max(0.0, (1.0 - (rain_val / 600.0)) * 100.0))
        
        # Combine into weighted stress index
        stress_score = 0.5 * depth_score + 0.3 * rain_score + 0.2 * trend_score
        stress_score = round(float(min(100.0, max(0.0, stress_score))), 2)

        return {
            "station_id": station_id,
            "current_gw": current_gw,
            "forecast_rainfall_accumulation_7d": round(total_forecast_rain, 2),
            "depletion_rate_m_day": round(depletion_rate + trend_rate, 4),
            "stress_score": stress_score,
            "projections": projections,
            "trajectory_daily": {
                "dates": dates,
                "p10": p10_trajectory,
                "p50": p50_trajectory,
                "p90": p90_trajectory
            }
        }

if __name__ == "__main__":
    forecaster = TrendForecaster()
    # Test on station 020109B with current water table at 12.42m
    res = forecaster.forecast_short_term("020109B", current_gw=12.42, prev_gw=4.97, recent_rain_180d=94.5)
    print(json.dumps(res, indent=2))
