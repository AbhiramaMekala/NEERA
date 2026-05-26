import math
import urllib.request
import urllib.parse
import json
from pathlib import Path

class GeoService:
    def __init__(self, database_path=None):
        self.db_path = database_path or Path(__file__).resolve().parent / "data/training_master_engineered.csv"
        self.stations_coords = []
        self._load_stations_coords()

    def _load_stations_coords(self):
        """Loads all station coordinates, IDs, and cluster info from the engineered database."""
        if not Path(self.db_path).exists():
            print(f"Warning: Database not found at {self.db_path}. Geo service cannot map station locations.")
            return

        try:
            import pandas as pd
            df = pd.read_csv(self.db_path)
            # Group by station to get unique spatial coordinates
            grouped = df.groupby("station_id").agg({
                "latitude": "first",
                "longitude": "first",
                "spatial_cluster": "first",
                "Groundwater_Level_MBGL": "last"
            }).reset_index()

            for _, row in grouped.iterrows():
                self.stations_coords.append({
                    "station_id": str(row["station_id"]),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "spatial_cluster": int(row["spatial_cluster"]),
                    "latest_depth": float(row["Groundwater_Level_MBGL"])
                })
            print(f"GeoService loaded coordinates for {len(self.stations_coords)} stations.")
        except Exception as e:
            print(f"Error loading stations coordinates in GeoService: {e}")

    def geocode(self, query):
        """Converts search query (city, village, block) into latitude and longitude using OSM Nominatim."""
        query_strip = query.strip()
        if len(query_strip) < 3:
            raise ValueError("Search query must be at least 3 characters long.")

        # Clean query
        query_encoded = urllib.parse.quote(query_strip)
        # Biased to India and bounded to Karnataka region
        url = f"https://nominatim.openstreetmap.org/search?q={query_encoded}&format=json&limit=1&countrycodes=in&viewbox=74.0,11.0,79.0,19.0"
        
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "NEERA-Hydrology-Intelligence/1.0 (abhiram@developer.neera)"
            })
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data:
                    res = data[0]
                    return {
                        "lat": float(res["lat"]),
                        "lon": float(res["lon"]),
                        "display_name": res["display_name"],
                        "class": res.get("class"),
                        "type": res.get("type")
                    }
        except Exception as e:
            print(f"Geocoding API failed for query '{query}': {e}. Using fallback coordinate bounds.")
        
        return self._fallback_geocode(query_strip)

    def autocomplete(self, query):
        """Returns autocomplete suggestions from Nominatim biased/restricted to Karnataka, India."""
        query_strip = query.strip()
        if len(query_strip) < 3:
            return []  # Reject queries < 3 chars
            
        query_encoded = urllib.parse.quote(query_strip)
        url = f"https://nominatim.openstreetmap.org/search?q={query_encoded}&format=json&limit=5&countrycodes=in&viewbox=74.0,11.0,79.0,19.0"
        
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "NEERA-Hydrology-Intelligence/1.0 (abhiram@developer.neera)"
            })
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
            results = []
            for item in data:
                results.append({
                    "display_name": item["display_name"],
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "class": item.get("class"),
                    "type": item.get("type")
                })
            return results
        except Exception as e:
            print(f"Autocomplete API failed for query '{query}': {e}")
            
        return self._fallback_autocomplete(query_strip)

    def reverse_geocode(self, lat, lon):
        """Converts latitude and longitude back to a descriptive address using OSM Nominatim."""
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "NEERA-Hydrology-Intelligence/1.0 (abhiram@developer.neera)"
            })
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data:
                    return data.get("display_name", f"Karnataka ({lat:.4f}, {lon:.4f})")
        except Exception as e:
            print(f"Reverse geocoding failed for lat={lat}, lon={lon}: {e}")
        
        return f"Karnataka Well Site ({lat:.4f}, {lon:.4f})"

    def resolve_nearest_station(self, lat, lon):
        """Finds the closest telemetry station to the input coordinates using the Haversine formula."""
        if not self.stations_coords:
            return None

        best_station = None
        min_distance = float("inf")

        for station in self.stations_coords:
            dist = self.haversine_distance(lat, lon, station["latitude"], station["longitude"])
            if dist < min_distance:
                min_distance = dist
                best_station = station

        if best_station:
            result = best_station.copy()
            result["distance_km"] = round(min_distance, 2)
            
            # Check 250km threshold
            if min_distance > 250.0:
                result["out_of_bounds"] = True
                result["message"] = f"No nearby monitoring station within 250km (Nearest: {result['station_id']} at {result['distance_km']}km)."
                result["disable_prediction"] = True
            else:
                result["out_of_bounds"] = False
                result["disable_prediction"] = False
                
            return result
        return None

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculates geodesic distance between two coordinate pairs in kilometers."""
        R = 6371.0 # Earth radius
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def _fallback_geocode(self, query):
        """Mock fallback for geocoding when OSM Nominatim is rate-limited or offline."""
        query_lower = query.lower()
        # Mock major Karnataka cities
        cities = {
            "bengaluru": {"lat": 12.9716, "lon": 77.5946, "display_name": "Bengaluru, Bangalore Urban, Karnataka, India"},
            "bangalore": {"lat": 12.9716, "lon": 77.5946, "display_name": "Bengaluru, Bangalore Urban, Karnataka, India"},
            "mysore": {"lat": 12.2958, "lon": 76.6394, "display_name": "Mysuru, Mysore District, Karnataka, India"},
            "mysuru": {"lat": 12.2958, "lon": 76.6394, "display_name": "Mysuru, Mysore District, Karnataka, India"},
            "hubli": {"lat": 15.3647, "lon": 75.1240, "display_name": "Hubballi, Dharwad, Karnataka, India"},
            "dharwad": {"lat": 15.4589, "lon": 75.0078, "display_name": "Dharwad, Dharwad District, Karnataka, India"},
            "mangalore": {"lat": 12.9141, "lon": 74.8560, "display_name": "Mangaluru, Dakshina Kannada, Karnataka, India"},
            "belgaum": {"lat": 15.8497, "lon": 74.4977, "display_name": "Belagavi, Belgaum District, Karnataka, India"},
            "vijayapura": {"lat": 16.8302, "lon": 75.7100, "display_name": "Vijayapura, Bijapur District, Karnataka, India"},
            "davangere": {"lat": 14.4644, "lon": 75.9218, "display_name": "Davangere, Davanagere District, Karnataka, India"}
        }

        for city, coords in cities.items():
            if city in query_lower:
                return coords

        # Try to parse query directly if coordinates are inputted
        try:
            parts = query.replace(",", " ").split()
            if len(parts) >= 2:
                lat = float(parts[0])
                lon = float(parts[1])
                if 11.0 <= lat <= 19.0 and 74.0 <= lon <= 79.0:
                    return {
                        "lat": lat,
                        "lon": lon,
                        "display_name": f"Custom Coordinates ({lat:.4f}, {lon:.4f})"
                    }
        except Exception:
            pass

        # Default center of Karnataka
        return {
            "lat": 15.3173,
            "lon": 75.7139,
            "display_name": f"Karnataka Center (Search Fallback: '{query}')"
        }

    def _fallback_autocomplete(self, query):
        query_lower = query.lower().strip()
        candidates = [
            {"display_name": "Bengaluru, Bangalore Urban, Karnataka, India", "lat": 12.9716, "lon": 77.5946, "class": "boundary", "type": "administrative"},
            {"display_name": "Mysuru, Mysore District, Karnataka, India", "lat": 12.2958, "lon": 76.6394, "class": "boundary", "type": "administrative"},
            {"display_name": "Hubballi, Dharwad, Karnataka, India", "lat": 15.3647, "lon": 75.1240, "class": "boundary", "type": "administrative"},
            {"display_name": "Dharwad, Dharwad District, Karnataka, India", "lat": 15.4589, "lon": 75.0078, "class": "boundary", "type": "administrative"},
            {"display_name": "Mangaluru, Dakshina Kannada, Karnataka, India", "lat": 12.9141, "lon": 74.8560, "class": "boundary", "type": "administrative"},
            {"display_name": "Belagavi, Belgaum District, Karnataka, India", "lat": 15.8497, "lon": 74.4977, "class": "boundary", "type": "administrative"},
            {"display_name": "Vijayapura, Bijapur District, Karnataka, India", "lat": 16.8302, "lon": 75.7100, "class": "boundary", "type": "administrative"},
            {"display_name": "Davangere, Davanagere District, Karnataka, India", "lat": 14.4644, "lon": 75.9218, "class": "boundary", "type": "administrative"},
            {"display_name": "Shivamogga, Shimoga District, Karnataka, India", "lat": 13.9299, "lon": 75.5681, "class": "boundary", "type": "administrative"},
            {"display_name": "Kalaburagi, Gulbarga District, Karnataka, India", "lat": 17.3291, "lon": 76.8343, "class": "boundary", "type": "administrative"}
        ]
        
        matches = []
        for c in candidates:
            if query_lower in c["display_name"].lower():
                matches.append(c)
        return matches[:5]

if __name__ == "__main__":
    service = GeoService()
    # Test geocoding
    coords = service.geocode("Bengaluru")
    print("Geocode:", coords)
    
    # Test nearest well mapping
    well = service.resolve_nearest_station(coords["lat"], coords["lon"])
    print("Nearest Well:", well)
