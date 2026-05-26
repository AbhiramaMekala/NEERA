import json
from pathlib import Path

class AlertEngine:
    def __init__(self):
        # Default thresholds
        # Note: higher MBGL depth = lower water table
        self.default_thresholds = {
            "warning_depth": 30.0,      # MBGL > 30m is Warning
            "critical_depth": 50.0,     # MBGL > 50m is Critical
            "rapid_depletion": 0.15,    # rate > 0.15m/day is rapid depletion (~1m/week)
            "rain_deficit_threshold": 100.0 # 180d rainfall < 100mm represents high deficit
        }
        
        # We can load station or district specific thresholds if configured
        self.station_thresholds = {}
        self.district_thresholds = {}

    def set_station_thresholds(self, station_id, warning_depth=None, critical_depth=None):
        if station_id not in self.station_thresholds:
            self.station_thresholds[station_id] = {}
        if warning_depth is not None:
            self.station_thresholds[station_id]["warning_depth"] = warning_depth
        if critical_depth is not None:
            self.station_thresholds[station_id]["critical_depth"] = critical_depth

    def evaluate_station_alert(self, forecast_res):
        """Evaluates short-term trend forecast results and triggers alerts.
        
        Args:
            forecast_res (dict): Output from TrendForecaster.forecast_short_term()
        """
        station_id = forecast_res["station_id"]
        current_gw = forecast_res["current_gw"]
        depletion_rate = forecast_res["depletion_rate_m_day"]
        rain_accum_7d = forecast_res["forecast_rainfall_accumulation_7d"]
        
        # Get active thresholds (prefer station specific, then default)
        thresh = self.default_thresholds.copy()
        if station_id in self.station_thresholds:
            thresh.update(self.station_thresholds[station_id])

        # Evaluate conditions
        severity = "SAFE"
        reasons = []
        recommended_actions = []

        # 1. Depth Check
        if current_gw > thresh["critical_depth"]:
            severity = "CRITICAL"
            reasons.append(f"Groundwater level is extremely deep ({current_gw:.2f} MBGL), exceeding critical threshold of {thresh['critical_depth']}m.")
            recommended_actions.extend([
                "RESTRICT ALL non-essential agricultural borewell extraction immediately.",
                "Mandate emergency drip irrigation protocols.",
                "Activate local artificial recharge wells."
            ])
        elif current_gw > thresh["warning_depth"]:
            severity = "WARNING"
            reasons.append(f"Groundwater level is deep ({current_gw:.2f} MBGL), exceeding warning threshold of {thresh['warning_depth']}m.")
            recommended_actions.extend([
                "Encourage community water auditing.",
                "Optimize irrigation schedule (water only in early morning/evening).",
                "Prepare rainwater harvesting channels."
            ])
        else:
            recommended_actions.append("Continue standard sustainable extraction and water table monitoring.")

        # 2. Depletion Rate Check
        # Positive depletion rate means water table is falling (MBGL increasing)
        if depletion_rate > thresh["rapid_depletion"]:
            # Upgrade severity
            if severity == "SAFE":
                severity = "WARNING"
            elif severity == "WARNING":
                severity = "CRITICAL"
            reasons.append(f"Rapid water table depletion detected: dropping at {depletion_rate * 7:.2f} meters/week (threshold: {thresh['rapid_depletion'] * 7:.2f}m/week).")
            recommended_actions.append("Investigate local borewells for excessive pumps drawdowns.")

        # 3. Rainfall deficit check
        # If forecast rainfall is zero and we have a deep current level
        if rain_accum_7d < 5.0 and current_gw > 15.0:
            if severity == "SAFE":
                severity = "MODERATE"
            reasons.append(f"Rainfall deficit persisting: near-zero forecast rain ({rain_accum_7d:.1f}mm) expected for the next 7 days.")
            recommended_actions.append("Minimize evaporation losses by mulching crops.")

        # Calculate predicted depletion timeline
        depletion_timeline = "N/A"
        if depletion_rate > 0:
            # How many days to hit warning or critical
            if current_gw < thresh["warning_depth"]:
                days_to_warning = (thresh["warning_depth"] - current_gw) / depletion_rate
                depletion_timeline = f"Projected to breach Warning threshold ({thresh['warning_depth']}m) in {days_to_warning:.1f} days."
            elif current_gw < thresh["critical_depth"]:
                days_to_critical = (thresh["critical_depth"] - current_gw) / depletion_rate
                depletion_timeline = f"Projected to breach Critical threshold ({thresh['critical_depth']}m) in {days_to_critical:.1f} days."
            else:
                depletion_timeline = "Already breached critical threshold."
        else:
            depletion_timeline = "Water table is rising or stable."

        if not reasons:
            reasons.append("Groundwater levels and rainfall forecasts are within safe operational limits.")

        # Clean duplicates in actions
        recommended_actions = list(dict.fromkeys(recommended_actions))

        return {
            "station_id": station_id,
            "alert_level": severity,
            "stress_score": forecast_res["stress_score"],
            "reasons": reasons,
            "depletion_timeline": depletion_timeline,
            "recommended_actions": recommended_actions
        }

if __name__ == "__main__":
    from trend_forecaster import TrendForecaster
    forecaster = TrendForecaster()
    engine = AlertEngine()
    
    # Test on station 020109B
    res = forecaster.forecast_short_term("020109B", current_gw=12.42, prev_gw=4.97)
    alert = engine.evaluate_station_alert(res)
    print(json.dumps(alert, indent=2))

    # Test critical station
    print("\n--- Test Critical Station ---")
    res_critical = forecaster.forecast_short_term("020109B", current_gw=52.42, prev_gw=48.0)
    alert_critical = engine.evaluate_station_alert(res_critical)
    print(json.dumps(alert_critical, indent=2))
