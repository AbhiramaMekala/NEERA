import json
import time
import urllib.request
import urllib.parse
from typing import Dict, Any
from weather_providers.base_provider import BaseWeatherProvider

class OpenMeteoProvider(BaseWeatherProvider):
    def fetch_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover,weather_code"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover,precipitation_probability"
            f"&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max,precipitation_probability_max"
            f"&timezone=auto"
        )
        
        req = urllib.request.Request(url, headers={
            "User-Agent": "NEERA-Hydrology-Intelligence/1.0 (abhiram@developer.neera)"
        })
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            
        return self._normalize_response(data, lat, lon)

    def _normalize_response(self, data: Dict[str, Any], lat: float, lon: float) -> Dict[str, Any]:
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        daily = data.get("daily", {})

        # Current normalization
        w_code = current.get("weather_code", 0)
        desc = self._map_wmo_code(w_code)
        
        current_rain = float(current.get("precipitation", 0.0))
        precip_prob = 80.0 if current_rain > 0.0 else 10.0

        current_normalized = {
            "temperature": float(current.get("temperature_2m", 28.0)),
            "humidity": float(current.get("relative_humidity_2m", 60.0)),
            "rainfall": current_rain,
            "wind_speed": float(current.get("wind_speed_10m", 5.0)),
            "cloud_cover": float(current.get("cloud_cover", 20.0)),
            "weather_description": desc,
            "precipitation_probability": precip_prob
        }

        # Hourly normalization (next 24 hours)
        hourly_normalized = []
        h_times = hourly.get("time", [])
        h_temps = hourly.get("temperature_2m", [])
        h_hums = hourly.get("relative_humidity_2m", [])
        h_rains = hourly.get("precipitation", [])
        h_winds = hourly.get("wind_speed_10m", [])
        h_clouds = hourly.get("cloud_cover", [])
        h_probs = hourly.get("precipitation_probability", [])

        now_epoch = int(time.time())
        for idx in range(min(24, len(h_times))):
            hourly_normalized.append({
                "time": h_times[idx].replace("T", " "),
                "epoch": now_epoch + idx * 3600,
                "temperature": float(h_temps[idx]) if idx < len(h_temps) else 25.0,
                "humidity": float(h_hums[idx]) if idx < len(h_hums) else 60.0,
                "rainfall": float(h_rains[idx]) if idx < len(h_rains) else 0.0,
                "wind_speed": float(h_winds[idx]) if idx < len(h_winds) else 5.0,
                "cloud_cover": float(h_clouds[idx]) if idx < len(h_clouds) else 20.0,
                "precipitation_probability": float(h_probs[idx]) if idx < len(h_probs) else 10.0
            })

        # Daily normalization (5 days)
        daily_normalized = []
        d_times = daily.get("time", [])
        d_temps = daily.get("temperature_2m_max", [])
        d_rains = daily.get("precipitation_sum", [])
        d_winds = daily.get("wind_speed_10m_max", [])
        d_probs = daily.get("precipitation_probability_max", [])

        for idx in range(min(5, len(d_times))):
            daily_normalized.append({
                "date": d_times[idx],
                "temperature": float(d_temps[idx]) if idx < len(d_temps) else 28.0,
                "humidity": 65.0,  # Estimated daily average relative humidity
                "rainfall": float(d_rains[idx]) if idx < len(d_rains) else 0.0,
                "wind_speed": float(d_winds[idx]) if idx < len(d_winds) else 8.0,
                "cloud_cover": 30.0,  # Estimated daily average cloud cover
                "precipitation_probability": float(d_probs[idx]) if idx < len(d_probs) and d_probs[idx] is not None else 20.0
            })

        return {
            "latitude": lat,
            "longitude": lon,
            "current": current_normalized,
            "hourly": hourly_normalized,
            "daily": daily_normalized
        }

    def _map_wmo_code(self, code: int) -> str:
        wmo_map = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            61: "light rain",
            63: "moderate rain",
            65: "heavy rain",
            71: "light snow",
            73: "moderate snow",
            75: "heavy snow",
            80: "light rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            95: "thunderstorm",
            96: "thunderstorm with light hail",
            99: "thunderstorm with heavy hail"
        }
        return wmo_map.get(code, "clear/overcast")
