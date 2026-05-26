import os
import json
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from ai_providers.base_ai_provider import BaseAIProvider

CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs/cache/ai"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class GeminiProvider(BaseAIProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def generate_text(self, prompt: str) -> str:
        """Generates text from prompt, using local cache, retries, and fallbacks."""
        # 1. Check local cache
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        cache_file = CACHE_DIR / f"{prompt_hash}.txt"
        
        if cache_file.exists():
            try:
                # If cached less than 24 hours ago, use it
                if time.time() - cache_file.stat().st_mtime < 86400:
                    with open(cache_file, "r") as f:
                        return f.read()
            except Exception as e:
                print(f"Error reading AI cache: {e}")

        # 2. Call Gemini API if key is available
        result_text = None
        if self.api_key:
            result_text = self._call_api_with_retry(prompt)

        # 3. Fallback if API fails or key is missing
        if not result_text:
            print("Gemini API key missing or request failed. Generating rule-based advisory fallback.")
            result_text = self._generate_fallback_advisory(prompt)

        # 4. Save to cache
        try:
            with open(cache_file, "w") as f:
                f.write(result_text)
        except Exception as e:
            print(f"Error writing AI cache: {e}")

        return result_text

    def _call_api_with_retry(self, prompt: str) -> str:
        """Makes direct HTTP request to Gemini endpoint with retries and exponential backoff."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        body = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        data_bytes = json.dumps(body).encode("utf-8")
        
        retries = 3
        delay = 2.0
        
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=data_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    res_json = json.loads(response.read().decode())
                    
                # Extract text response from Gemini schema
                candidates = res_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        if text:
                            return text
                return None
            except urllib.error.HTTPError as e:
                print(f"Gemini API HTTP Error (Attempt {attempt+1}/{retries}): Code {e.code} - {e.reason}")
                # Handle rate limit (429) or temporary server error (503)
                if e.code in (429, 503, 500):
                    time.sleep(delay)
                    delay *= 2
                else:
                    break
            except Exception as e:
                print(f"Gemini API connection error (Attempt {attempt+1}/{retries}): {e}")
                time.sleep(delay)
                delay *= 2
                
        return None

    def _generate_fallback_advisory(self, prompt: str) -> str:
        """Extracts numerical context from prompt and builds a robust rule-based explanation."""
        # Simple parser to find keywords and numbers in prompt
        prompt_lower = prompt.lower()
        
        # Determine status
        status = "DEPLETING"
        import re
        status_match = re.search(r"sustainability status:\s*([a-z_ ]+)", prompt_lower)
        if status_match:
            status_val = status_match.group(1).strip().upper()
            if status_val in ("STABLE", "STRESSED", "DEPLETING", "CRITICAL", "COLLAPSE RISK", "COLLAPSE"):
                status = status_val
        else:
            if "stable" in prompt_lower:
                status = "STABLE"
            elif "stressed" in prompt_lower:
                status = "STRESSED"
            elif "critical" in prompt_lower:
                status = "CRITICAL"
            elif "collapse" in prompt_lower:
                status = "COLLAPSE"
            
        # Parse some depth numbers
        depths = re.findall(r"depth:\s*([\d\.]+)", prompt_lower)
        depth_val = float(depths[0]) if depths else 22.5
        
        days_warn = re.findall(r"warning:\s*([\d]+)", prompt_lower)
        days_warn_val = int(days_warn[0]) if days_warn else 45
        
        days_crit = re.findall(r"critical:\s*([\d]+)", prompt_lower)
        days_crit_val = int(days_crit[0]) if days_crit else 120
        
        # Build template
        if status in ("CRITICAL", "COLLAPSE"):
            return f"""**HYDROLOGICAL ANALYSIS & EMERGENCY WARNING**

**Condition Assessment:**
The region is currently in a **CRITICAL** state. The groundwater table is located at an extreme depth of **{depth_val:.2f} meters** below ground level. Extraction rates are heavily outstripping the natural replenishment capacity of the local aquifer.

**Projections & Timelines:**
If current extraction pressures persist, localized dry-well scenarios are expected. Immediate collapse warning triggers indicate that shallow and medium-depth borewells will deplete completely.

**Immediate Actions Required:**
1. **Government**: Cease all approvals for new borewells immediately. Implement mandatory community water quotas.
2. **Farmers**: Prioritize high-value drinking water over water-intensive cash crops. Strict mandate to transition to drip-irrigation networks.
3. **Citizens**: Implement intensive domestic graywater recycling and lock all non-potable hose usage."""
        
        elif status == "DEPLETING":
            return f"""**HYDROLOGICAL ANALYSIS & SUSTAINABILITY COMMENTARY**

**Condition Assessment:**
Groundwater levels in this region are declining steadily due to insufficient recharge and rising extraction pressure. The water table is at **{depth_val:.2f} meters** below ground level and showing a downward trajectory.

**Projections & Timelines:**
If current conditions persist, the region may enter a critical groundwater stress phase within approximately **{days_crit_val} days** (Warning threshold breach estimated in **{days_warn_val} days**).

**Sustainability Measures Recommended:**
1. **Agriculture**: Limit flood irrigation hours and adopt mulching to prevent evapotranspiration losses.
2. **Infrastructure**: Activate and clean local rainwater harvesting structures to maximize monsoon infiltration.
3. **Conservation**: Adopt voluntary consumption restrictions across commercial sites."""
        
        elif status == "STRESSED":
            return f"""**HYDROLOGICAL COMMENTARY**

**Condition Assessment:**
The local aquifer shows signs of moderate stress (Current Level: **{depth_val:.2f} meters**). While not in an active crisis, seasonal variations and localized high draws are creating localized cones of depression.

**Conservation Strategies:**
1. Maintain regular monitoring of borewell static levels.
2. Initiate community-scale aquifer recharge planning.
3. Adopt smart soil moisture sensors to minimize agricultural drawdowns."""
            
        else:
            return f"""**HYDROLOGICAL COMMENTARY (SYSTEM ASSESSMENT: NOMINAL)**

**Condition Assessment:**
The region's groundwater levels are currently **STABLE** (Water Table: **{depth_val:.2f} meters**). Natural seasonal recharge is in equilibrium with local agricultural extraction.

**Sustainable Practices:**
1. Continue standard water table telemetry checks.
2. Maintain local rainwater harvesting channels to preserve current aquifer health."""
