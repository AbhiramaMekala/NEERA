import time
import numpy as np
from typing import Dict, Any, List

class CrisisForecaster:
    def __init__(self, warning_threshold: float = 30.0, critical_threshold: float = 50.0, collapse_threshold: float = 70.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.collapse_threshold = collapse_threshold

    def calculate_crisis_timeline(self, current_gw: float, depletion_rate: float, forecast_rain_7d: float, long_term_slope: float = 0.0) -> dict:
        """Calculates days until warning, critical, and collapse thresholds across multiple weather scenarios."""
        # 1. Base Depletion Velocity (meters per day)
        # Ensure positive value represents falling water table (MBGL increasing)
        velocity = max(-0.1, min(0.3, depletion_rate))

        # Detect accelerating depletion
        # If current short-term depletion velocity is higher than long-term historical depletion slope, depletion is accelerating
        depletion_acceleration = False
        if velocity > 0 and long_term_slope > 0:
            # Convert yearly slope to daily rate
            daily_historical_rate = long_term_slope / 365.25
            if velocity > daily_historical_rate * 1.1:
                depletion_acceleration = True

        # 2. Multi-Scenario Models (projecting 90 days out)
        scenarios = {
            "normal": {"rain_mult": 1.0, "extraction_mult": 1.0},
            "drought": {"rain_mult": 0.0, "extraction_mult": 1.2},
            "monsoon": {"rain_mult": 1.5, "extraction_mult": 0.5},
            "heatwave": {"rain_mult": 0.5, "extraction_mult": 1.4}
        }
        
        projections = {}
        timelines = {}
        
        # Standard recharge coefficients
        recharge_coeff = 0.08 if current_gw < 20.0 else 0.02
        
        for name, config in scenarios.items():
            trajectory = []
            days_to_warn = -1
            days_to_crit = -1
            days_to_coll = -1
            
            running_gw = current_gw
            
            # Simple daily projection loop for 90 days
            for day in range(1, 91):
                # Daily rain estimate
                daily_rain = (forecast_rain_7d / 7.0) * config["rain_mult"]
                daily_recharge = daily_rain * recharge_coeff
                
                # Daily extraction drawdown
                daily_drawdown = velocity * config["extraction_mult"]
                
                # Update GW level
                running_gw = running_gw + daily_drawdown - daily_recharge
                running_gw = max(0.0, running_gw)
                trajectory.append(round(running_gw, 3))
                
                # Check threshold breaches
                if running_gw >= self.warning_threshold and days_to_warn == -1:
                    days_to_warn = day
                if running_gw >= self.critical_threshold and days_to_crit == -1:
                    days_to_crit = day
                if running_gw >= self.collapse_threshold and days_to_coll == -1:
                    days_to_coll = day
            
            # Handle cases already breached or never breached
            # If current level is already beyond, day to breach is 0
            if current_gw >= self.warning_threshold:
                days_to_warn = 0
            if current_gw >= self.critical_threshold:
                days_to_crit = 0
            if current_gw >= self.collapse_threshold:
                days_to_coll = 0
                
            projections[name] = trajectory
            timelines[name] = {
                "days_to_warning": days_to_warn if days_to_warn != -1 else 999,
                "days_to_critical": days_to_crit if days_to_crit != -1 else 999,
                "days_to_collapse": days_to_coll if days_to_coll != -1 else 999
            }

        # Calculate confidence metric (0.0 to 1.0)
        # Confidence drops as depth increases and as long-term variance increases
        confidence = 0.95 - min(0.4, current_gw / 150.0)
        if depletion_acceleration:
            confidence -= 0.05
        confidence = round(float(max(0.4, confidence)), 2)

        # Recharge failure probability (higher if rainfall deficit is high)
        recharge_failure_prob = 0.15
        if forecast_rain_7d < 2.0:
            recharge_failure_prob = 0.75
        elif forecast_rain_7d < 10.0:
            recharge_failure_prob = 0.45

        # Normal scenario timeline as primary status output
        norm_timeline = timelines["normal"]
        
        status = "STABLE"
        if current_gw >= self.collapse_threshold:
            status = "COLLAPSE"
        elif current_gw >= self.critical_threshold:
            status = "CRITICAL"
        elif current_gw >= self.warning_threshold:
            status = "WARNING"
        elif velocity > 0.05:
            status = "DEPLETING"
        elif velocity > 0.01:
            status = "STRESSED"

        return {
            "status": status,
            "days_to_warning": norm_timeline["days_to_warning"],
            "days_to_critical": norm_timeline["days_to_critical"],
            "days_to_collapse": norm_timeline["days_to_collapse"],
            "depletion_acceleration": depletion_acceleration,
            "confidence": confidence,
            "recharge_failure_probability": recharge_failure_prob,
            "timelines_by_scenario": timelines,
            "projections_by_scenario": projections,
            "thresholds": {
                "warning": self.warning_threshold,
                "critical": self.critical_threshold,
                "collapse": self.collapse_threshold
            }
        }

if __name__ == "__main__":
    forecaster = CrisisForecaster()
    res = forecaster.calculate_crisis_timeline(
        current_gw=25.4,
        depletion_rate=0.18, # 18cm/day drop
        forecast_rain_7d=5.0,
        long_term_slope=1.2 # 1.2m/year
    )
    print(res)
