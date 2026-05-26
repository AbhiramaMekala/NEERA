import numpy as np
from pathlib import Path
import pandas as pd

class SustainabilityEngine:
    def __init__(self, database_path=None):
        self.db_path = database_path or Path(__file__).resolve().parent / "data/training_master_engineered.csv"
        self._database = None
        self._load_database()

    def _load_database(self):
        if Path(self.db_path).exists():
            try:
                self._database = pd.read_csv(self.db_path)
                self._database["date"] = pd.to_datetime(self._database["date"])
            except Exception as e:
                print(f"Error loading database in SustainabilityEngine: {e}")

    def compute_sustainability(self, station_id: str, current_gw: float, forecast_rain_7d: float, depletion_rate: float, recent_rain_180d: float = 200.0) -> dict:
        """Computes key groundwater sustainability metrics and yields a score (0-100)."""
        # 1. Recharge vs Depletion Balance
        # Estimated recharge based on forecast rain (7d)
        recharge_coeff = 0.08 if current_gw < 20.0 else 0.02
        projected_recharge = forecast_rain_7d * recharge_coeff
        # Weekly projected depletion
        projected_depletion = abs(depletion_rate) * 7.0 if depletion_rate > 0 else 0.0
        balance = projected_recharge - projected_depletion

        # 2. Long-Term Depletion Slope (meters per year)
        slope = 0.0
        if self._database is not None:
            station_data = self._database[self._database["station_id"] == station_id]
            if len(station_data) > 3:
                # Sort by date
                station_data = station_data.sort_values(by="date")
                # Calculate time differences in years from start
                t_start = station_data["date"].min()
                years = (station_data["date"] - t_start).dt.days / 365.25
                depths = station_data["Groundwater_Level_MBGL"]
                
                # Fit linear regression line
                try:
                    coefs = np.polyfit(years, depths, 1)
                    slope = coefs[0] # positive slope means water table is dropping (MBGL is increasing)
                except Exception:
                    pass

        # 3. Rainfall Deficit Index (RDI)
        # Normal 180d rainfall in Karnataka is roughly 600mm
        normal_rain = 600.0
        rain_val = max(0.0, recent_rain_180d)
        rdi = max(0.0, min(1.0, (normal_rain - rain_val) / normal_rain))

        # 4. Recharge Efficiency
        # Ratio of historical water table recovery compared to precipitation
        recharge_efficiency = recharge_coeff

        # 5. Groundwater Stress Ratio (GSR)
        # Critical depth threshold is 50m. Limit GSR to 1.5
        gsr = min(1.5, current_gw / 50.0)

        # 6. Seasonal Resilience Score (0 to 100)
        # High resilience = shallow aquifer and high rain recharge potential
        depth_resilience = max(0.0, min(100.0, (1.0 - (current_gw / 60.0)) * 100.0))
        rain_resilience = max(0.0, min(100.0, (1.0 - rdi) * 100.0))
        resilience_score = round(0.6 * depth_resilience + 0.4 * rain_resilience, 2)

        # 7. Extraction Risk Estimate (0 to 100)
        # Risk increases with high slope and deep current water level
        slope_risk = min(100.0, max(0.0, slope * 25.0)) # slope of 4m/year is critical (100)
        depth_risk = min(100.0, (current_gw / 50.0) * 100.0)
        extraction_risk = round(0.5 * slope_risk + 0.5 * depth_risk, 2)

        # 8. Recovery Potential (0 to 100)
        recovery_potential = round(max(0.0, min(100.0, (projected_recharge / (projected_depletion + 0.01)) * 50.0)), 2)

        # Calculate Unified Sustainability Score (0 to 100)
        # Lower scores represent higher unsustainability / crisis
        # Score components:
        # A. Depth score (100 is shallow, 0 is deep)
        score_depth = max(0.0, min(100.0, (1.0 - (current_gw / 60.0)) * 100.0))
        # B. Trend score (100 is rising/stable, 0 is rapid depletion)
        score_trend = max(0.0, min(100.0, (1.0 - (slope / 3.0)) * 100.0)) if slope > 0 else 100.0
        # C. Rainfall score (100 is full rain, 0 is heavy deficit)
        score_rain = (1.0 - rdi) * 100.0

        sustainability_score = 0.4 * score_depth + 0.3 * score_trend + 0.3 * score_rain
        sustainability_score = round(float(max(0.0, min(100.0, sustainability_score))), 2)

        # Determine Sustainability Status
        if sustainability_score >= 80.0:
            status = "STABLE"
        elif sustainability_score >= 60.0:
            status = "STRESSED"
        elif sustainability_score >= 40.0:
            status = "DEPLETING"
        elif sustainability_score >= 20.0:
            status = "CRITICAL"
        else:
            status = "COLLAPSE RISK"

        return {
            "station_id": station_id,
            "sustainability_score": sustainability_score,
            "sustainability_status": status,
            "metrics": {
                "recharge_depletion_balance_mm": round(float(balance), 4),
                "long_term_depletion_slope_m_yr": round(float(slope), 4),
                "rainfall_deficit_index": round(float(rdi), 4),
                "recharge_efficiency": round(float(recharge_efficiency), 4),
                "groundwater_stress_ratio": round(float(gsr), 4),
                "seasonal_resilience_score": resilience_score,
                "extraction_risk_estimate": extraction_risk,
                "recovery_potential_estimate": recovery_potential
            }
        }

if __name__ == "__main__":
    engine = SustainabilityEngine()
    res = engine.compute_sustainability(
        station_id="020109B",
        current_gw=15.42,
        forecast_rain_7d=25.0,
        depletion_rate=0.012,
        recent_rain_180d=350.0
    )
    print(res)
