import json
import time
import urllib.request
import urllib.parse
from typing import Dict, Any
from weather_providers.base_provider import BaseWeatherProvider

class OpenWeatherMapProvider(BaseWeatherProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is missing.")

        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"

        req_curr = urllib.request.Request(current_url, headers={"User-Agent": "NEERA-Hydrology-Intelligence/1.0"})
        with urllib.request.urlopen(req_curr, timeout=5) as resp:
            current_data = json.loads(resp.read().decode())

        req_fore = urllib.request.Request(forecast_url, headers={"User-Agent": "NEERA-Hydrology-Intelligence/1.0"})
        with urllib.request.urlopen(req_fore, timeout=5) as resp:
            forecast_data = json.loads(resp.read().decode())

        return self._normalize_response(current_data, forecast_data, lat, lon)

    def _normalize_response(self, current: Dict[str, Any], forecast: Dict[str, Any], lat: float, lon: float) -> Dict[str, Any]:
        # 1. Normalize Current Weather
        main_c = current.get("main", {})
        wind_c = current.get("wind", {})
        clouds_c = current.get("clouds", {})
        weather_list_c = current.get("weather", [{}])
        rain_c = current.get("rain", {})
        
        current_rain = float(rain_c.get("1h", rain_c.get("3h", 0.0)))
        desc = weather_list_c[0].get("description", "clear sky")

        current_normalized = {
            "temperature": float(main_c.get("temp", 28.0)),
            "humidity": float(main_c.get("humidity", 60.0)),
            "rainfall": current_rain,
            "wind_speed": float(wind_c.get("speed", 5.0)) * 3.6,  # m/s to km/h (to align with Open-Meteo)
            "cloud_cover": float(clouds_c.get("all", 20.0)),
            "weather_description": desc,
            "precipitation_probability": 80.0 if current_rain > 0.0 else 10.0
        }

        # 2. Normalize Hourly Weather (takes 3-hourly and estimates 24 hours, first 8 entries)
        hourly_normalized = []
        list_f = forecast.get("list", [])
        
        for idx in range(min(8, len(list_f))):
            item = list_f[idx]
            main_i = item.get("main", {})
            wind_i = item.get("wind", {})
            clouds_i = item.get("clouds", {})
            rain_i = item.get("rain", {})
            weather_i = item.get("weather", [{}])
            
            hourly_normalized.append({
                "time": item.get("dt_txt", "").replace("T", " "),
                "epoch": int(item.get("dt", time.time())),
                "temperature": float(main_i.get("temp", 25.0)),
                "humidity": float(main_i.get("humidity", 60.0)),
                "rainfall": float(rain_i.get("3h", 0.0)) / 3.0,  # 3-hourly amount divided to estimate hourly rate
                "wind_speed": float(wind_i.get("speed", 5.0)) * 3.6,
                "cloud_cover": float(clouds_i.get("all", 20.0)),
                "precipitation_probability": float(item.get("pop", 0.0)) * 100.0
            })

        # 3. Normalize Daily Weather (Group 3-hourly forecasts by day)
        daily_normalized = []
        daily_groups = {}
        
        for item in list_f:
            dt_txt = item.get("dt_txt", "")
            if not dt_txt:
                continue
            date_str = dt_txt.split()[0]
            if date_str not in daily_groups:
                daily_groups[date_str] = []
            daily_groups[date_str].append(item)

        # Get up to 5 days
        sorted_dates = sorted(daily_groups.keys())[:5]
        for date_str in sorted_dates:
            items = daily_groups[date_str]
            max_temp = -999.0
            total_rain = 0.0
            sum_humidity = 0.0
            max_wind = 0.0
            max_pop = 0.0
            sum_cloud = 0.0

            for item in items:
                main_i = item.get("main", {})
                wind_i = item.get("wind", {})
                rain_i = item.get("rain", {})
                clouds_i = item.get("clouds", {})
                
                temp = float(main_i.get("temp_max", main_i.get("temp", -999.0)))
                if temp > max_temp:
                    max_temp = temp
                
                total_rain += float(rain_i.get("3h", 0.0))
                sum_humidity += float(main_i.get("humidity", 60.0))
                
                wind = float(wind_i.get("speed", 0.0)) * 3.6
                if wind > max_wind:
                    max_wind = wind
                
                pop = float(item.get("pop", 0.0)) * 100.0
                if pop > max_pop:
                    max_pop = pop
                
                sum_cloud += float(clouds_i.get("all", 30.0))

            n_items = len(items)
            daily_normalized.append({
                "date": date_str,
                "temperature": max_temp if max_temp > -900.0 else 28.0,
                "humidity": sum_humidity / n_items if n_items > 0 else 65.0,
                "rainfall": total_rain,
                "wind_speed": max_wind,
                "cloud_cover": sum_cloud / n_items if n_items > 0 else 30.0,
                "precipitation_probability": max_pop
            })

        return {
            "latitude": lat,
            "longitude": lon,
            "current": current_normalized,
            "hourly": hourly_normalized,
            "daily": daily_normalized
        }
