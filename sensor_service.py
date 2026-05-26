import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List

class SensorService:
    def __init__(self, db_path=None):
        self.db_path = db_path or Path(__file__).resolve().parent / "data/telemetry.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initializes SQLite database with tables for sensors and telemetry history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create sensors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensors (
                    sensor_id TEXT PRIMARY KEY,
                    station_id TEXT,
                    latitude REAL,
                    longitude REAL,
                    model TEXT,
                    registered_at REAL
                )
            """)
            
            # Create telemetry_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor_id TEXT,
                    water_level REAL,
                    battery REAL,
                    temperature REAL,
                    humidity REAL,
                    timestamp TEXT,
                    FOREIGN KEY (sensor_id) REFERENCES sensors(sensor_id)
                )
            """)
            conn.commit()

    def register_sensor(self, sensor_id: str, station_id: str, latitude: float, longitude: float, model: str = "NEERA-IoT-Alpha"):
        """Registers a telemetry sensor in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sensors (sensor_id, station_id, latitude, longitude, model, registered_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(sensor_id) DO UPDATE SET
                    station_id=excluded.station_id,
                    latitude=excluded.latitude,
                    longitude=excluded.longitude,
                    model=excluded.model
                """,
                (sensor_id, station_id, latitude, longitude, model, time.time())
            )
            conn.commit()
        return self.get_sensor_status(sensor_id)

    def ingest_telemetry(self, payload: Dict[str, Any]):
        """Validates, registers if new, and stores sensor telemetry in SQLite."""
        sensor_id = payload["sensor_id"]
        lat = float(payload.get("lat", 15.3173))
        lon = float(payload.get("lon", 75.7139))
        water_level = float(payload.get("water_level", 0.0))
        battery = float(payload.get("battery", 100.0))
        temp = float(payload.get("temperature", 25.0))
        hum = float(payload.get("humidity", 60.0))
        timestamp = payload.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Auto-register sensor if it doesn't exist
        if not self._sensor_exists(sensor_id):
            try:
                from geo_service import GeoService
                geo = GeoService()
                resolved = geo.resolve_nearest_station(lat, lon)
                station_id = resolved["station_id"] if resolved else "unknown"
            except Exception:
                station_id = "unknown"
            self.register_sensor(sensor_id, station_id, lat, lon)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO telemetry_history (sensor_id, water_level, battery, temperature, humidity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sensor_id, water_level, battery, temp, hum, timestamp)
            )
            conn.commit()

        return self.get_sensor_status(sensor_id)

    def _sensor_exists(self, sensor_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM sensors WHERE sensor_id = ?", (sensor_id,))
            return cursor.fetchone() is not None

    def list_sensors(self) -> List[Dict[str, Any]]:
        """Lists all registered sensors with their calculated status and alerts."""
        sensors_list = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sensors")
            rows = cursor.fetchall()
            
            for row in rows:
                sensor_id = row["sensor_id"]
                status = self.get_sensor_status(sensor_id)
                sensors_list.append(status)
        return sensors_list

    def get_sensor_status(self, sensor_id: str) -> Dict[str, Any]:
        """Calculates sensor metadata, latest reading, battery alerts, and online status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Fetch sensor details
            cursor.execute("SELECT * FROM sensors WHERE sensor_id = ?", (sensor_id,))
            sensor_row = cursor.fetchone()
            if not sensor_row:
                return {"error": f"Sensor {sensor_id} not found."}
            
            # Fetch latest telemetry reading
            cursor.execute(
                """
                SELECT * FROM telemetry_history 
                WHERE sensor_id = ? 
                ORDER BY timestamp DESC LIMIT 1
                """, 
                (sensor_id,)
            )
            telemetry_row = cursor.fetchone()
            
        # Calculate status
        last_seen = None
        status = "offline"
        latest_reading = {}
        alerts = []

        if telemetry_row:
            latest_reading = {
                "water_level_mbgl": telemetry_row["water_level"],
                "battery_pct": telemetry_row["battery"],
                "temperature_c": telemetry_row["temperature"],
                "humidity_pct": telemetry_row["humidity"],
                "timestamp": telemetry_row["timestamp"]
            }
            
            # Parse timestamp to calculate age
            # If ISO8601 formatted, try to parse
            try:
                from datetime import datetime
                ts_str = telemetry_row["timestamp"]
                # Parse common ISO formats
                for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d %H:%M:%S"):
                    try:
                        dt = datetime.strptime(ts_str.replace("Z", ""), fmt.split(".")[0])
                        last_seen = dt.timestamp()
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
            
            if not last_seen:
                last_seen = time.time()  # Fallback to current if unable to parse

            age_seconds = time.time() - last_seen
            if age_seconds < 86400:  # 24 hours
                status = "online"
            else:
                alerts.append("STALE_SENSOR_ALERT: No telemetry received in past 24 hours.")

            if telemetry_row["battery"] < 20:
                alerts.append("LOW_BATTERY_ALERT: Sensor battery level is below 20%.")
        else:
            alerts.append("STALE_SENSOR_ALERT: No telemetry history recorded.")

        return {
            "sensor_id": sensor_row["sensor_id"],
            "station_id": sensor_row["station_id"],
            "latitude": sensor_row["latitude"],
            "longitude": sensor_row["longitude"],
            "model": sensor_row["model"],
            "registered_at": sensor_row["registered_at"],
            "status": status,
            "last_seen_epoch": last_seen,
            "latest_reading": latest_reading,
            "alerts": alerts
        }

    def get_sensor_history(self, sensor_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves history of telemetry readings for a sensor."""
        history = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM telemetry_history 
                WHERE sensor_id = ? 
                ORDER BY timestamp DESC LIMIT ?
                """,
                (sensor_id, limit)
            )
            rows = cursor.fetchall()
            for r in rows:
                history.append({
                    "id": r["id"],
                    "water_level": r["water_level"],
                    "battery": r["battery"],
                    "temperature": r["temperature"],
                    "humidity": r["humidity"],
                    "timestamp": r["timestamp"]
                })
        return history
