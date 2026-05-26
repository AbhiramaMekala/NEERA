from ai_providers.gemini_provider import GeminiProvider
import json

class AIAdvisor:
    def __init__(self, gemini_provider=None):
        self.provider = gemini_provider or GeminiProvider()

    def get_sustainability_advisory(self, station_id: str, gw_level: float, status: str, days_warn: int, days_crit: int, days_coll: int, sust_score: float, weather_desc: str, rdi: float) -> str:
        """Constructs prompt for Gemini to generate a contextual hydrological explanation."""
        prompt = f"""
        Act as a senior expert hydrologist and climate advisor for the NEERA Groundwater Intelligence platform in Karnataka, India.
        
        Analyze the following spatiotemporal groundwater sustainability indicators for monitoring Station {station_id}:
        - Current Groundwater Depth: {gw_level:.2f} meters below ground level (MBGL)
        - Sustainability Score (0-100): {sust_score:.1f}
        - Sustainability Status: {status}
        - Projected Days until Warning Threshold (30m MBGL): {days_warn} days
        - Projected Days until Critical Threshold (50m MBGL): {days_crit} days
        - Projected Days until Dry-Well / Collapse Scenario (70m MBGL): {days_coll} days
        - Live Weather Conditions: {weather_desc}
        - Climatological Rainfall Deficit Index: {rdi:.2f} (0 is normal, 1 is severe drought)

        Please generate a professional, scientific, and actionable Hydrological Analysis and Crisis Advisory in Markdown.
        Ensure it matches the following structure:
        
        ### 1. Hydrological Situation Assessment
        Provide a concise, 2-3 sentence overview of the region's current water table health.
        *Drafting Tip:* It MUST sound exactly like this:
        "Groundwater levels in this region are declining steadily due to insufficient recharge and rising extraction pressure. If current conditions persist, the region may enter a critical groundwater stress phase within approximately 4 months..."

        ### 2. Sustainability & Recharge Analysis
        Explain the physical recharge dynamics, evapotranspiration factors, and why the sustainability score is at {sust_score:.1f}. Connect the rainfall deficit index ({rdi:.2f}) and weather conditions ({weather_desc}) to the aquifer's recharge rates.

        ### 3. Crisis Action Plan
        List specific water conservation recommendations for:
        - Local Farmers (crop selection, irrigation schedules)
        - Citizens (graywater recycling, conservation)
        - Municipal Water Authorities (pumping limits, recharge interventions)
        
        Keep your output direct, professional, and do not include any meta-talk or introductory greetings. Start directly with the Markdown sections.
        """
        
        return self.provider.generate_text(prompt)

    def answer_copilot_chat(self, context_dict: dict, user_question: str) -> str:
        """Uses Gemini to answer user questions contextually based on live platform metrics."""
        prompt = f"""
        Act as the NEERA Hydro-Advisor AI Chat Copilot. Your job is to answer user questions regarding groundwater levels, sustainability, and water policy contextually using live platform metrics.
        
        Here is the current hydrological status of the selected site:
        {json.dumps(context_dict, indent=2)}

        User Question: "{user_question}"

        Provide a scientifically sound, helpful, and concise answer. Reference the metrics (sustainability score, days to crisis thresholds, current depth, weather) directly when answering. Do not assume data that is not provided.
        """
        return self.provider.generate_text(prompt)
