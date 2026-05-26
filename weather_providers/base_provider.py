from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseWeatherProvider(ABC):
    @abstractmethod
    def fetch_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetches weather data for a given lat/lon.
        Returns a normalized dict format:
        {
            "latitude": float,
            "longitude": float,
            "current": {
                "temperature": float,
                "humidity": float,
                "rainfall": float,
                "wind_speed": float,
                "cloud_cover": float,
                "weather_description": str,
                "precipitation_probability": float
            },
            "hourly": [
                {
                    "time": str,
                    "epoch": int,
                    "temperature": float,
                    "humidity": float,
                    "rainfall": float,
                    "wind_speed": float,
                    "cloud_cover": float,
                    "precipitation_probability": float
                },
                ...
            ],
            "daily": [
                {
                    "date": str,
                    "temperature": float,
                    "humidity": float,
                    "rainfall": float,
                    "wind_speed": float,
                    "cloud_cover": float,
                    "precipitation_probability": float
                },
                ...
            ]
        }
        """
        pass
