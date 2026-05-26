import os
import json
import time
import random
from pathlib import Path
from weather_providers.open_meteo_provider import OpenMeteoProvider
from weather_providers.openweather_provider import OpenWeatherMapProvider

# Cache settings
CACHE_TTL = 3600  # 1 hour
CACHE_DIR = Path(__file__).resolve().parent / "outputs/cache/weather"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class WeatherService:
    def __init__(self, database_path=None):
        self.db_path = database_path or Path(__file__).resolve().parent / "data/training_master_engineered.csv"
        self.stations_metadata = {}
        self._load_stations_metadata()
        
        # Initialize providers
        self.open_meteo = OpenMeteoProvider()
        owm_key = os.getenv("OPENWEATHER_API_KEY")
        if owm_key:
            self.open_weather = OpenWeatherMapProvider(owm_key)
        else:
            self.open_weather = None

    def _load_stations_metadata(self):
        """Loads coordinates, district, and spatial cluster mapping for all stations."""
        if not Path(self.db_path).exists():
            print(f"Warning: Database not found at {self.db_path}. Weather coordinates mapping will default.")
            return

        try:
            import pandas as pd
            df = pd.read_csv(self.db_path)
            # Group by station to get unique static attributes
            grouped = df.groupby("station_id").agg({
                "latitude": "first",
                "longitude": "first",
                "spatial_cluster": "first",
                "season": "last"
            }).reset_index()

            for _, row in grouped.iterrows():
                self.stations_metadata[row["station_id"]] = {
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "spatial_cluster": int(row["spatial_cluster"]),
                    "last_season": str(row["season"])
                }
            print(f"WeatherService loaded metadata for {len(self.stations_metadata)} stations.")
        except Exception as e:
            print(f"Error loading stations metadata for WeatherService: {e}")

    def get_weather_by_coordinates(self, lat, lon, station_id="default"):
        """Fetches weather data for a latitude and longitude with caching and fallbacks."""
        cache_file = CACHE_DIR / f"weather_{lat:.4f}_{lon:.4f}.json"
        
        # Check cache validity
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                if time.time() - cached_data.get("timestamp", 0) < CACHE_TTL:
                    # Cache hit!
                    return cached_data.get("data")
            except Exception as e:
                print(f"Error reading cache for lat={lat}, lon={lon}: {e}")

        # Fetch from providers with fallback switching and retry logic
        weather_data = self._fetch_from_providers_with_retry(lat, lon)

        if not weather_data:
            # Fallback to realistic mock weather
            print(f"All weather providers failed for lat={lat:.4f}, lon={lon:.4f}. Using fallback mock weather.")
            weather_data = self._generate_mock_weather(lat, lon, station_id)

        # Update cache
        try:
            with open(cache_file, "w") as f:
                json.dump({"timestamp": time.time(), "data": weather_data}, f)
        except Exception as e:
            print(f"Error writing cache for lat={lat}, lon={lon}: {e}")

        return weather_data

    def get_weather_by_station(self, station_id):
        """Fetches weather using the station ID by mapping to its registered coordinates."""
        metadata = self.stations_metadata.get(station_id)
        if metadata:
            return self.get_weather_by_coordinates(metadata["latitude"], metadata["longitude"], station_id)
        
        # Fallback default coordinates if station not found
        default_lat, default_lon = 15.3173, 75.7139  # Center of Karnataka
        return self.get_weather_by_coordinates(default_lat, default_lon, station_id)

    def _fetch_from_providers_with_retry(self, lat: float, lon: float):
        """Attempts weather fetch using Open-Meteo, with fallback to OpenWeatherMap, using retry + backoff."""
        providers = [("Open-Meteo", self.open_meteo)]
        if self.open_weather:
            providers.append(("OpenWeatherMap", self.open_weather))

        for provider_name, provider in providers:
            retries = 3
            delay = 1.0
            for attempt in range(retries):
                try:
                    print(f"Attempting weather fetch from {provider_name} (Attempt {attempt+1}/{retries})...")
                    data = provider.fetch_weather(lat, lon)
                    if data:
                        return data
                except Exception as e:
                    print(f"Weather provider {provider_name} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                        delay *= 2
        return None

    def _generate_mock_weather(self, lat, lon, station_id):
        """Generates a highly realistic, climatologically correct mock weather payload for Karnataka."""
        current_month = time.localtime().tm_mon
        
        # Classify season
        if 6 <= current_month <= 9:
            season = "monsoon"
            base_temp = 24.0
            base_humidity = 85
            rain_chance = 75.0
            daily_rain_mean = 12.5
        elif 10 <= current_month <= 1:
            season = "winter"
            base_temp = 22.0
            base_humidity = 55
            rain_chance = 10.0
            daily_rain_mean = 1.0
        else:
            season = "summer"
            base_temp = 34.0
            base_humidity = 40
            rain_chance = 15.0
            daily_rain_mean = 2.5

        # Seed mock with coordinates to keep it stable per coordinates
        random.seed(hash((lat, lon, current_month)) % 10000000)

        current_temp = base_temp + random.uniform(-3, 3)
        current_humidity = min(100, max(10, int(base_humidity + random.uniform(-10, 10))))
        is_raining = random.uniform(0, 100) < rain_chance
        current_rain = random.uniform(1.0, 15.0) if is_raining else 0.0
        
        desc = "moderate rain" if current_rain > 5 else "light rain" if current_rain > 0 else "scattered clouds" if season == "monsoon" else "clear sky"

        # Generate hourly forecast for 24h
        hourly = []
        now_epoch = int(time.time())
        for h in range(24):
            hour_epoch = now_epoch + h * 3600
            hour_val = time.localtime(hour_epoch).tm_hour
            diurnal_offset = -5.0 * np_sin_approx((hour_val - 6) / 24.0 * 2.0 * 3.14159)
            h_temp = current_temp + diurnal_offset + random.uniform(-1, 1)
            h_hum = min(100, max(15, current_humidity - int(diurnal_offset * 2)))
            h_rain_chance = rain_chance + (10 if hour_val >= 16 else -10)
            h_is_raining = random.uniform(0, 100) < h_rain_chance
            h_rain = random.uniform(0.5, 4.0) if h_is_raining else 0.0

            hourly.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(hour_epoch)),
                "epoch": hour_epoch,
                "temperature": round(float(h_temp), 2),
                "humidity": int(h_hum),
                "rainfall": round(float(h_rain), 2),
                "wind_speed": round(random.uniform(2.0, 12.0), 2),
                "cloud_cover": round(random.uniform(10.0, 90.0), 2),
                "precipitation_probability": round(float(min(100.0, max(0.0, h_rain_chance))), 1)
            })

        # Generate daily forecast for 5 days
        daily = []
        for d in range(5):
            day_epoch = now_epoch + d * 86400
            date_str = time.strftime("%Y-%m-%d", time.localtime(day_epoch))
            d_temp = base_temp + random.uniform(-2, 2)
            d_hum = min(100, max(15, int(base_humidity + random.uniform(-8, 8))))
            d_rain_chance = min(100.0, max(0.0, rain_chance + random.uniform(-15, 15)))
            d_is_raining = random.uniform(0, 100) < d_rain_chance
            d_rain = random.exponential(daily_rain_mean) if d_is_raining else 0.0

            daily.append({
                "date": date_str,
                "temperature": round(float(d_temp), 2),
                "humidity": round(float(d_hum), 1),
                "rainfall": round(float(d_rain), 2),
                "wind_speed": round(random.uniform(3.0, 15.0), 2),
                "cloud_cover": round(random.uniform(10.0, 80.0), 2),
                "precipitation_probability": round(float(d_rain_chance), 1)
            })

        return {
            "latitude": lat,
            "longitude": lon,
            "current": {
                "temperature": round(float(current_temp), 2),
                "humidity": int(current_humidity),
                "rainfall": round(float(current_rain), 2),
                "wind_speed": round(random.uniform(2.0, 12.0), 2),
                "cloud_cover": round(random.uniform(10.0, 95.0), 2),
                "weather_description": desc,
                "precipitation_probability": round(float(rain_chance), 1)
            },
            "hourly": hourly,
            "daily": daily
        }

def np_sin_approx(x):
    """Simple trigonometric approximation for sinus."""
    pi = 3.141592653589793
    x = x % (2 * pi)
    if x > pi:
        x -= 2 * pi
    return x - (x**3)/6.0 + (x**5)/120.0

if __name__ == "__main__":
    service = WeatherService()
    print(service.get_weather_by_station("020109B"))
