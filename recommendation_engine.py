class RecommendationEngine:
    def get_recommendations(self, status: str, weather_desc: str, rdi: float, temp: float) -> list:
        """Generates dynamic recommendations based on groundwater severity and weather factors."""
        actions = []
        
        # 1. Base Severity Actions
        if status in ("CRITICAL", "COLLAPSE", "COLLAPSE RISK"):
            actions.extend([
                "EMERGENCY: Mandate immediate halt of all non-essential agricultural tube-well draws.",
                "Restrict municipal supply draws to drinking water purposes only.",
                "Enforce crop restriction directives: prohibit sowing water-heavy sugarcane/paddy crops.",
                "Deploy emergency village water tanker scheduling services."
            ])
        elif status == "DEPLETING":
            actions.extend([
                "Enforce mandatory transition to micro-drip networks for agricultural plots.",
                "Establish surface rainwater runoff trenches around all active extraction wells.",
                "Restrict active borewell pumping hours to a maximum of 4 hours daily.",
                "Recommend mulch layering on soil beds to conserve residual crop moisture."
            ])
        elif status == "STRESSED":
            actions.extend([
                "Recommend voluntary community water auditing logs.",
                "Transition public parks and green zones to native drought-tolerant xeriscaping.",
                "Schedule field irrigation only during early mornings or evenings to minimize evaporation."
            ])
        else:
            actions.extend([
                "Groundwater levels are stable. Maintain standard sustainable extraction guidelines.",
                "Perform routine telemetry checks and clear local recharge basin inlets."
            ])
            
        # 2. Weather Context Modifiers
        if rdi > 0.6:
            actions.append("CLIMATE RISK: Severe seasonal precipitation deficit. Stop aquifer groundwater draws for decorative or commercial building uses.")
        
        if temp > 37.0:
            actions.append("HEAT WARNING: High evapotranspiration risk. Implement shaded greenhouse canopies and prioritize soil-moisture preservation.")
            
        if "rain" in weather_desc.lower() or "drizzle" in weather_desc.lower():
            actions.append("RECHARGE WINDOW: Active rainfall detected. Clear silt from rainwater harvesting catchments to maximize aquifer infiltration.")

        return actions
