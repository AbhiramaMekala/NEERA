"use client";

import React, { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import { 
  Droplet, 
  CloudRain, 
  AlertTriangle, 
  MapPin, 
  TrendingUp, 
  Wind, 
  Thermometer, 
  Calendar, 
  Compass, 
  RefreshCw, 
  Activity, 
  Search,
  CheckCircle,
  HelpCircle,
  Database,
  Navigation,
  Sun,
  Flame,
  Zap,
  Info,
  Server
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend
} from "recharts";

// Dynamically load the Leaflet Map component with SSR disabled
const MapComponent = dynamic(() => import("../components/MapComponent"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[500px] bg-slate-950/80 border border-slate-800/80 rounded-xl flex items-center justify-center">
      <div className="flex flex-col items-center gap-2">
        <Compass className="w-8 h-8 text-cyan-500 animate-spin" />
        <span className="text-xs text-slate-500">Initializing GIS Map Layers...</span>
      </div>
    </div>
  )
});

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface WeatherData {
  latitude: number;
  longitude: number;
  current: {
    temperature: number;
    humidity: number;
    rainfall: number;
    wind_speed: number;
    cloud_cover: number;
    weather_description: string;
    precipitation_probability: number;
  };
  hourly: Array<{
    time: string;
    temperature: number;
    humidity: number;
    rainfall: number;
    wind_speed: number;
    cloud_cover: number;
    precipitation_probability: number;
  }>;
  daily: Array<{
    date: string;
    temperature: number;
    humidity: number;
    rainfall: number;
    wind_speed: number;
    cloud_cover: number;
    precipitation_probability: number;
  }>;
}

interface ForecastData {
  station_id: string;
  current_gw: number;
  forecast_rainfall_accumulation_7d: number;
  depletion_rate_m_day: number;
  stress_score: number;
  projections: {
    [key: string]: {
      date: string;
      p50: number;
      p10: number;
      p90: number;
    }
  };
  trajectory_daily: {
    dates: string[];
    p10: number[];
    p50: number[];
    p90: number[];
  };
}

interface AlertData {
  station_id: string;
  alert_level: string;
  stress_score: number;
  reasons: string[];
  depletion_timeline: string;
  recommended_actions: string[];
}

interface RiskSummary {
  total_stations: number;
  deep_wells_count: number;
  average_depth_mbgl: number;
  max_depth_mbgl: number;
  cluster_stats: Array<{
    cluster_id: number;
    station_count: number;
    avg_depth_mbgl: number;
    risk_level: string;
  }>;
  map_markers: Array<{
    station_id: string;
    latitude: number;
    longitude: number;
    depth: number;
    cluster: number;
  }>;
}

interface HistoryRecord {
  station_id: string;
  timestamp: string;
  date?: string;
  Groundwater_Level_MBGL: number;
  prev_gw: number | null;
  effective_rainfall_180d: number;
}

interface EnvRisk {
  station_id: string;
  recent_depth_mbgl: number;
  forecast_rainfall_7d_mm: number;
  heatwave_stress: string;
  dry_spell_risk: string;
  recharge_potential: string;
}

interface IotSensor {
  sensor_id: string;
  station_id: string;
  latitude: number;
  longitude: number;
  model: string;
  status: string;
  latest_reading: {
    water_level_mbgl?: number;
    battery_pct?: number;
    temperature_c?: number;
    humidity_pct?: number;
    timestamp?: string;
  };
  alerts: string[];
}

export default function Dashboard() {
  // Navigation & Location States
  const [selectedStation, setSelectedStation] = useState<string>("020109B");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [activeAddress, setActiveAddress] = useState<string>("Vijayapura Taluk, Karnataka, India");
  const [mappedDistance, setMappedDistance] = useState<number | null>(null);
  const [errorState, setErrorState] = useState<{ message: string; nearest_station_id?: string; distance_km?: number } | null>(null);
  
  // Autocomplete Suggestions
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);

  // Connection & IoT telemetry
  const [apiStatus, setApiStatus] = useState<"connected" | "disconnected" | "checking">("checking");
  const [iotSensors, setIotSensors] = useState<IotSensor[]>([]);

  // Telemetry States
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [riskSummary, setRiskSummary] = useState<RiskSummary | null>(null);
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [envRisk, setEnvRisk] = useState<EnvRisk | null>(null);
  
  // UI states
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [geocoding, setGeocoding] = useState<boolean>(false);
  const [isLiveTelemetry, setIsLiveTelemetry] = useState<boolean>(false);
  const [telemetryLogs, setTelemetryLogs] = useState<string[]>([]);
  
  const telemetryIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const telemetryEndRef = useRef<HTMLDivElement | null>(null);

  // Check API health periodically
  const checkApiHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        setApiStatus("connected");
      } else {
        setApiStatus("disconnected");
      }
    } catch {
      setApiStatus("disconnected");
    }
  };

  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Load initial listings
  useEffect(() => {
    fetchBaseData();
    fetchLocationData({ station_id: "020109B" });
  }, []);

  // Autocomplete search suggestions debouncer
  useEffect(() => {
    if (searchQuery.trim().length < 3) {
      setSuggestions([]);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/geocode/autocomplete?query=${encodeURIComponent(searchQuery)}`);
        if (res.ok) {
          const list = await res.json();
          setSuggestions(list);
        }
      } catch (err) {
        console.error("Autocomplete fetch failed:", err);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Live Telemetry simulation pushing POST payloads to SQLite
  useEffect(() => {
    if (isLiveTelemetry) {
      addTelemetryLog("Telemetry Service: Activating automated IoT ingestion client...");
      addTelemetryLog("Telemetry Service: Initializing HTTP POST ingestion broker...");
      
      telemetryIntervalRef.current = setInterval(async () => {
        const sensorId = `iot-${selectedStation.substring(0, 5).toLowerCase()}`;
        const payload = {
          sensor_id: sensorId,
          lat: weather ? weather.latitude : 15.3173,
          lon: weather ? weather.longitude : 75.7139,
          water_level: Number((10 + Math.random() * 25).toFixed(2)),
          battery: Math.random() < 0.05 ? 15 : Math.floor(80 + Math.random() * 20),
          temperature: Number((24 + Math.random() * 6).toFixed(1)),
          humidity: Math.floor(50 + Math.random() * 40),
          timestamp: new Date().toISOString()
        };

        try {
          const res = await fetch(`${API_BASE_URL}/api/sensors/ingest`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
          });
          if (res.ok) {
            const timeStr = new Date().toLocaleTimeString();
            addTelemetryLog(
              `[${timeStr}] Ingested payload for ${payload.sensor_id}: ` + 
              `Depth=${payload.water_level}m | Batt=${payload.battery}% | Temp=${payload.temperature}°C`
            );
            
            // Refresh sensor list
            const listRes = await fetch(`${API_BASE_URL}/api/sensors/latest`);
            if (listRes.ok) {
              const list = await listRes.json();
              setIotSensors(list);
            }
          }
        } catch (err) {
          console.error("Sensor ingestion failed:", err);
          addTelemetryLog(`Ingest Error: Failed to push telemetry to SQLite. API offline.`);
        }
      }, 4000);
    } else {
      if (telemetryIntervalRef.current) {
        clearInterval(telemetryIntervalRef.current);
      }
      addTelemetryLog("Live telemetry push client stopped.");
    }
    return () => {
      if (telemetryIntervalRef.current) clearInterval(telemetryIntervalRef.current);
    };
  }, [isLiveTelemetry, selectedStation, weather]);

  // Auto-scroll telemetry log console
  useEffect(() => {
    if (telemetryEndRef.current) {
      telemetryEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [telemetryLogs]);

  const addTelemetryLog = (msg: string) => {
    setTelemetryLogs(prev => [...prev.slice(-49), msg]);
  };

  const fetchBaseData = async () => {
    try {
      const summaryRes = await fetch(`${API_BASE_URL}/api/risk-summary`);
      const summary = await summaryRes.json();
      setRiskSummary(summary);

      const alertsRes = await fetch(`${API_BASE_URL}/api/alerts?limit=15`);
      const alertsList = await alertsRes.json();
      setAlerts(alertsList);

      const listRes = await fetch(`${API_BASE_URL}/api/sensors/latest`);
      if (listRes.ok) {
        const list = await listRes.json();
        setIotSensors(list);
      }
    } catch (e) {
      console.error("Error fetching base listings:", e);
      addTelemetryLog("Network Error: Failed to contact NEERA FastAPI backend server.");
    }
  };

  const fetchLocationData = async (params: { station_id?: string; lat?: number; lon?: number; query?: string }) => {
    setRefreshing(true);
    setErrorState(null);
    try {
      let queryString = "";
      if (params.station_id) queryString = `station_id=${params.station_id}`;
      else if (params.lat !== undefined && params.lon !== undefined) queryString = `lat=${params.lat}&lon=${params.lon}`;
      else if (params.query) queryString = `query=${encodeURIComponent(params.query)}`;

      addTelemetryLog("API Request: Resolving forecast data for query/coords...");
      const res = await fetch(`${API_BASE_URL}/api/forecast?${queryString}`);
      if (!res.ok) {
        throw new Error(`Inference service error: ${res.statusText}`);
      }
      
      const data = await res.json();
      
      if (data.disable_prediction) {
        setErrorState({
          message: data.error,
          nearest_station_id: data.nearest_station_id,
          distance_km: data.distance_km
        });
        setActiveAddress(data.resolved_location);
        setMappedDistance(data.distance_km);
        setSelectedStation(data.nearest_station_id);
        
        setForecast(null);
        setWeather(null);
        setEnvRisk(null);
        setHistory([]);
        
        addTelemetryLog(`Boundary Error: Resolved location is ${data.distance_km}km from nearest station (limit: 250km).`);
        return;
      }

      // Safe within boundaries
      setActiveAddress(data.resolved_location);
      setMappedDistance(data.nearest_station.distance_km);
      setSelectedStation(data.nearest_station.station_id);
      setForecast(data.forecast);
      setWeather(data.weather);
      
      setEnvRisk({
        station_id: data.nearest_station.station_id,
        recent_depth_mbgl: data.forecast.current_gw,
        forecast_rainfall_7d_mm: data.forecast.forecast_rainfall_accumulation_7d,
        heatwave_stress: data.alert.reasons.some((r: string) => r.toLowerCase().includes("heatwave") || r.toLowerCase().includes("hot")) ? "CRITICAL" : "SAFE",
        dry_spell_risk: data.forecast.forecast_rainfall_accumulation_7d < 2.0 ? "ACTIVE" : "INACTIVE",
        recharge_potential: data.forecast.forecast_rainfall_accumulation_7d > 15.0 ? "HIGH" : "LOW"
      });

      // Insert new forecast alert to list
      setAlerts(prev => {
        const filtered = prev.filter(a => a.station_id !== data.nearest_station.station_id);
        return [data.alert, ...filtered];
      });

      // Load history
      const historyRes = await fetch(`${API_BASE_URL}/stations/${data.nearest_station.station_id}/history?limit=15`);
      if (historyRes.ok) {
        const historyData = await historyRes.json();
        setHistory(historyData.reverse());
      } else {
        setHistory([]);
      }
      
      addTelemetryLog(`Forecast updated successfully for ${data.nearest_station.station_id} (${data.resolved_location}).`);

    } catch (e: any) {
      console.error(e);
      addTelemetryLog(`Inference Error: ${e.message}`);
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  };

  const handleGeocodeSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;
    
    setGeocoding(true);
    setShowSuggestions(false);
    addTelemetryLog(`Geocoding Search: Locating '${searchQuery}'...`);
    await fetchLocationData({ query: searchQuery });
    setGeocoding(false);
  };

  const handleUseMyLocation = () => {
    if (!navigator.geolocation) {
      addTelemetryLog("Geolocation Error: Browser does not support GPS features.");
      return;
    }

    addTelemetryLog("Geolocation: Accessing browser device coordinates...");
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        setSearchQuery(`${lat.toFixed(4)}, ${lon.toFixed(4)}`);
        addTelemetryLog(`Device coordinates acquired: Lat ${lat.toFixed(4)}, Lon ${lon.toFixed(4)}.`);
        await fetchLocationData({ lat, lon });
      },
      (err) => {
        addTelemetryLog(`Geolocation Error: Access denied.`);
      }
    );
  };

  const handleMapClick = async (lat: number, lon: number) => {
    addTelemetryLog(`Map Clicked: Reverse geocoding coords lat=${lat.toFixed(4)}, lon=${lon.toFixed(4)}...`);
    setSearchQuery(`${lat.toFixed(4)}, ${lon.toFixed(4)}`);
    await fetchLocationData({ lat, lon });
  };

  const getProjectionsChartData = () => {
    if (!forecast) return [];
    
    const dates = forecast.trajectory_daily.dates;
    const p10 = forecast.trajectory_daily.p10;
    const p50 = forecast.trajectory_daily.p50;
    const p90 = forecast.trajectory_daily.p90;

    return dates.map((date, idx) => ({
      date: date.substring(5),
      P10_Shallower: p10[idx],
      P50_Expected: p50[idx],
      P90_Deeper: p90[idx]
    }));
  };

  const getHistoryChartData = () => {
    return history.map(item => ({
      date: item.timestamp || item.date || "N/A",
      GW_Depth_MBGL: item.Groundwater_Level_MBGL,
      Effective_Rain_180d: item.effective_rainfall_180d
    }));
  };

  const getAlertColor = (level: string) => {
    if (level === "CRITICAL") return "text-red-500 bg-red-950/40 border-red-800";
    if (level === "WARNING") return "text-orange-500 bg-orange-950/40 border-orange-800";
    if (level === "MODERATE") return "text-yellow-500 bg-yellow-950/40 border-yellow-800";
    return "text-green-500 bg-green-950/40 border-green-800";
  };

  const getAlertDotColor = (level: string) => {
    if (level === "CRITICAL") return "bg-red-500 ring-red-400";
    if (level === "WARNING") return "bg-orange-500 ring-orange-400";
    if (level === "MODERATE") return "bg-yellow-500 ring-yellow-400";
    return "bg-green-500 ring-green-400";
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans p-4 md:p-6 selection:bg-cyan-500 selection:text-slate-950">
      
      {/* Header Panel */}
      <header className="border-b border-slate-900 pb-5 mb-6 flex flex-col xl:flex-row justify-between items-start xl:items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl shadow-lg shadow-cyan-500/20">
            <Droplet className="w-8 h-8 text-slate-950 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-3xl font-black tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                NEERA Geographic Intelligence
              </h1>
              <div className="flex items-center gap-1 text-[10px] bg-slate-900 border border-slate-800/80 px-2.5 py-0.5 rounded-full text-slate-400">
                <span className={`w-1.5 h-1.5 rounded-full ${apiStatus === "connected" ? "bg-green-500 animate-pulse" : apiStatus === "disconnected" ? "bg-red-500" : "bg-yellow-500 animate-pulse"}`}></span>
                <span className="font-bold">{apiStatus === "connected" ? "API Live" : apiStatus === "disconnected" ? "API Offline" : "Connecting..."}</span>
              </div>
            </div>
            <p className="text-xs font-semibold text-slate-400 mt-0.5">
              Civic-Tech Hydrology Operations & Drought Early-Warning Platform
            </p>
          </div>
        </div>

        {/* Unified Search Control Center with Autocomplete */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 w-full xl:w-auto">
          <div className="relative w-full sm:w-96">
            <form onSubmit={handleGeocodeSearch} className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 w-full shadow-inner focus-within:border-cyan-500 transition-colors">
              <div className="pl-3 text-slate-500">
                <Search className="w-4 h-4" />
              </div>
              <input
                type="text"
                className="w-full bg-transparent border-0 outline-none px-3 py-2 text-sm text-slate-200 placeholder-slate-500"
                placeholder="Search by city, village, or coordinates..."
                value={searchQuery}
                onFocus={() => setShowSuggestions(true)}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 250)}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setShowSuggestions(true);
                }}
              />
              <button
                type="submit"
                disabled={geocoding || !searchQuery.trim()}
                className="px-4 py-1.5 text-xs bg-cyan-600 hover:bg-cyan-500 text-white rounded font-bold transition-all mr-1 disabled:opacity-50 cursor-pointer"
              >
                {geocoding ? "Locating..." : "Locate"}
              </button>
            </form>

            {showSuggestions && suggestions.length > 0 && (
              <ul className="absolute left-0 right-0 mt-1 bg-slate-900 border border-slate-800 rounded-lg shadow-2xl max-h-60 overflow-y-auto z-50 text-xs divide-y divide-slate-850">
                {suggestions.map((s, idx) => (
                  <li 
                    key={idx}
                    onClick={() => {
                      setSearchQuery(s.display_name);
                      setShowSuggestions(false);
                      fetchLocationData({ lat: s.lat, lon: s.lon });
                    }}
                    className="px-4 py-3 hover:bg-slate-800 cursor-pointer text-slate-300 hover:text-white transition-colors"
                  >
                    <div className="font-semibold">{s.display_name.split(",")[0]}</div>
                    <div className="text-[10px] text-slate-500 truncate">{s.display_name}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleUseMyLocation}
              className="px-4 py-2.5 text-xs bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg flex items-center gap-1.5 text-slate-300 font-semibold cursor-pointer"
            >
              <Navigation className="w-3.5 h-3.5" />
              <span>Use My Location</span>
            </button>

            <button
              onClick={() => setIsLiveTelemetry(!isLiveTelemetry)}
              className={`px-4 py-2.5 text-xs rounded-lg border font-bold flex items-center gap-1.5 transition-all cursor-pointer ${isLiveTelemetry ? "bg-red-500/10 text-red-400 border-red-500/30 hover:bg-red-500/20" : "bg-cyan-500/10 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/20"}`}
            >
              <Activity className={`w-3.5 h-3.5 ${isLiveTelemetry ? "animate-pulse" : ""}`} />
              <span>{isLiveTelemetry ? "Disconnect Telemetry" : "Stream Telemetry"}</span>
            </button>
          </div>
        </div>
      </header>

      {loading ? (
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <Droplet className="w-12 h-12 text-cyan-500 animate-bounce" />
          <p className="text-slate-400 animate-pulse text-sm">Synchronizing regional coordinates databases...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          
          {/* LEFT SIDEBAR - Geographic selectors, active station details, environmental cards (5 Columns) */}
          <div className="xl:col-span-5 flex flex-col gap-6">
            
            {/* Active Node Info Panel */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 backdrop-blur-md relative overflow-hidden">
              <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/5 rounded-full blur-xl -mr-6 -mt-6"></div>
              
              <span className="text-[10px] font-black tracking-wider text-cyan-400 uppercase bg-cyan-950/40 border border-cyan-900/60 px-2.5 py-1 rounded-full">
                Active Geography
              </span>
              
              <h2 className="text-xl font-black text-slate-100 mt-3 truncate">{activeAddress}</h2>
              
              <div className="mt-3.5 grid grid-cols-2 gap-3 text-xs border-b border-slate-800/60 pb-3.5">
                <div>
                  <span className="text-slate-500 block">Nearest Monitor</span>
                  <span className="font-bold text-slate-200">Station {selectedStation}</span>
                </div>
                <div>
                  <span className="text-slate-500 block">Relative Distance</span>
                  <span className="font-bold text-slate-200">
                    {mappedDistance !== null ? `${mappedDistance.toFixed(2)} km` : "0.00 km (Exact Lookup)"}
                  </span>
                </div>
              </div>

              {forecast && (
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div>
                    <span className="text-[10px] uppercase text-slate-500 font-bold block">Water Table</span>
                    <span className="text-2xl font-black text-slate-200">{forecast.current_gw.toFixed(2)}m</span>
                    <span className="text-[9px] text-slate-500 block">MBGL (Meters Below Ground)</span>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase text-slate-500 font-bold block">Weekly Change Velocity</span>
                    <span className={`text-lg font-bold block mt-1 ${forecast.depletion_rate_m_day > 0 ? "text-red-400" : "text-green-400"}`}>
                      {forecast.depletion_rate_m_day > 0 ? `+${(forecast.depletion_rate_m_day * 7).toFixed(2)}m` : `${(forecast.depletion_rate_m_day * 7).toFixed(2)}m`}
                    </span>
                    <span className="text-[9px] text-slate-500 block">MBGL change projection</span>
                  </div>
                </div>
              )}
            </div>

            {/* Error state card if station is out of bounds (> 250km) */}
            {errorState && (
              <div className="bg-red-950/20 border border-red-900/50 rounded-xl p-5 flex flex-col gap-3">
                <div className="flex items-center gap-2 text-red-400 font-bold text-sm">
                  <AlertTriangle className="w-5 h-5 text-red-500 stroke-[2.5]" />
                  <span>Out of Telemetry Bounds</span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {errorState.message} The resolved location **{activeAddress}** lies too far from our active monitoring grid to make spatiotemporally reliable groundwater predictions.
                </p>
                <div className="grid grid-cols-2 gap-3 mt-1.5 border-t border-slate-900 pt-3 text-[11px] text-slate-500">
                  <div>
                    <span className="block text-slate-600">Nearest Well ID</span>
                    <span className="font-bold text-slate-450">{errorState.nearest_station_id}</span>
                  </div>
                  <div>
                    <span className="block text-slate-600">Measured Distance</span>
                    <span className="font-bold text-slate-455 text-red-450">{errorState.distance_km?.toFixed(2)} km</span>
                  </div>
                </div>
              </div>
            )}

            {/* Environmental Intelligence widget cards */}
            {envRisk && (
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between backdrop-blur-md">
                  <div className="flex justify-between items-center text-slate-500 mb-2">
                    <span className="text-[9px] uppercase font-bold tracking-wider font-sans">Dry Spell</span>
                    <Sun className="w-4 h-4 text-orange-400" />
                  </div>
                  <div>
                    <span className="text-xs text-slate-500 block font-sans">Risk Indicator</span>
                    <span className={`text-sm font-black ${envRisk.dry_spell_risk === "ACTIVE" ? "text-orange-400" : "text-green-400"}`}>
                      {envRisk.dry_spell_risk}
                    </span>
                  </div>
                </div>

                <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between backdrop-blur-md">
                  <div className="flex justify-between items-center text-slate-500 mb-2">
                    <span className="text-[9px] uppercase font-bold tracking-wider font-sans">Heatwave</span>
                    <Flame className="w-4 h-4 text-red-400" />
                  </div>
                  <div>
                    <span className="text-xs text-slate-500 block font-sans">Temp Stress</span>
                    <span className={`text-sm font-black ${envRisk.heatwave_stress === "CRITICAL" ? "text-red-500" : "text-green-400"}`}>
                      {envRisk.heatwave_stress}
                    </span>
                  </div>
                </div>

                <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between backdrop-blur-md">
                  <div className="flex justify-between items-center text-slate-500 mb-2">
                    <span className="text-[9px] uppercase font-bold tracking-wider font-sans">Recharge</span>
                    <Zap className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div>
                    <span className="text-xs text-slate-500 block font-sans">Potential Capacity</span>
                    <span className={`text-sm font-black ${envRisk.recharge_potential === "HIGH" ? "text-green-400" : "text-cyan-400"}`}>
                      {envRisk.recharge_potential}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Interactive Leaflet GIS Map Layer */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 backdrop-blur-md flex flex-col">
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5 font-sans">
                  <Compass className="w-4 h-4 text-cyan-500" />
                  Karnataka GIS Groundwater Map
                </h3>
                <span className="text-[10px] text-slate-500 bg-slate-950 border border-slate-800 px-2 py-0.5 rounded-full">Interactive OSM</span>
              </div>
              
              {riskSummary && (
                <MapComponent
                  markers={riskSummary.map_markers}
                  selectedStationId={selectedStation}
                  onStationSelect={(id) => {
                    setSelectedStation(id);
                    fetchLocationData({ station_id: id });
                  }}
                  onMapClick={handleMapClick}
                />
              )}
            </div>

            {/* IoT Grid Status Panel */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 backdrop-blur-md">
              <div className="flex items-center justify-between mb-3 border-b border-slate-800/60 pb-2">
                <h3 className="text-xs font-bold uppercase text-slate-400 flex items-center gap-1.5 font-sans">
                  <Database className="w-4 h-4 text-cyan-400" />
                  IoT Sensor Telemetry Grid
                </h3>
                <span className="text-[10px] text-slate-500 bg-slate-950 px-2 py-0.5 rounded border border-slate-850">SQLite Persistent</span>
              </div>
              {iotSensors.length === 0 ? (
                <div className="text-[11px] text-slate-500 p-2 text-center">No active physical IoT sensors registered in database.</div>
              ) : (
                <div className="space-y-2.5 max-h-48 overflow-y-auto pr-1">
                  {iotSensors.map((s, idx) => (
                    <div key={idx} className="bg-slate-950/60 border border-slate-900 p-2.5 rounded-lg flex flex-col gap-1.5 transition-all hover:border-slate-850">
                      <div className="flex justify-between items-center text-xs">
                        <div className="flex items-center gap-1.5">
                          <span className={`w-1.5 h-1.5 rounded-full ${s.status === "online" ? "bg-green-500 animate-pulse" : "bg-slate-600"}`}></span>
                          <span className="font-bold text-slate-200">{s.sensor_id}</span>
                        </div>
                        <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold ${s.status === "online" ? "bg-green-950/80 text-green-400" : "bg-red-950/80 text-red-405"}`}>
                          {s.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-[10px] text-slate-500 border-t border-slate-900 pt-1.5 mt-0.5">
                        <div>
                          <span className="block text-[8px] uppercase text-slate-600 font-sans">Water Lvl</span>
                          <span className="font-semibold text-slate-300">{s.latest_reading?.water_level_mbgl?.toFixed(2) || "N/A"}m</span>
                        </div>
                        <div>
                          <span className="block text-[8px] uppercase text-slate-600 font-sans">Battery</span>
                          <span className={`font-semibold ${s.latest_reading?.battery_pct !== undefined && s.latest_reading.battery_pct < 20 ? "text-red-400 animate-pulse font-bold" : "text-slate-300"}`}>
                            {s.latest_reading?.battery_pct !== undefined ? `${s.latest_reading.battery_pct}%` : "N/A"}
                          </span>
                        </div>
                        <div>
                          <span className="block text-[8px] uppercase text-slate-600 font-sans">Temperature</span>
                          <span className="font-semibold text-slate-300">{s.latest_reading?.temperature_c !== undefined ? `${s.latest_reading.temperature_c.toFixed(1)}°C` : "N/A"}</span>
                        </div>
                      </div>
                      {s.alerts && s.alerts.length > 0 && (
                        <div className="text-[9px] text-red-400 bg-red-950/20 border border-red-900/30 p-1.5 rounded mt-1 space-y-0.5">
                          {s.alerts.map((al: string, i: number) => (
                            <div key={i} className="flex gap-1 items-start">
                              <AlertTriangle className="w-2.5 h-2.5 text-red-500 mt-0.5 shrink-0" />
                              <span>{al}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Live Telemetry Logger Console */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 backdrop-blur-md flex-1 flex flex-col min-h-[160px]">
              <div className="flex items-center justify-between mb-3 border-b border-slate-800/60 pb-2">
                <h3 className="text-xs font-bold uppercase text-slate-400 flex items-center gap-1.5 font-sans">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  Telemetry Log Receiver
                </h3>
                <span className={`w-2 h-2 rounded-full ${isLiveTelemetry ? "bg-green-500 animate-pulse" : "bg-slate-700"}`}></span>
              </div>
              
              <div className="bg-slate-950 font-mono text-[10px] p-3 rounded-lg overflow-y-auto max-h-48 xl:max-h-[220px] flex-1 text-cyan-500/80 space-y-1 leading-relaxed border border-slate-900 shadow-inner">
                {telemetryLogs.map((log, i) => (
                  <div key={i} className={log.includes("Error") ? "text-red-400" : log.includes("ingestion") || log.includes("Ingested") ? "text-yellow-450 font-bold" : ""}>
                    {log}
                  </div>
                ))}
                <div ref={telemetryEndRef} />
              </div>
            </div>

          </div>

          {/* RIGHT GRID CONTENT - Trajectory forecasts, weather summaries, alert logs (7 Columns) */}
          <div className="xl:col-span-7 flex flex-col gap-6">
            
            {/* Severe warnings Panel */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 backdrop-blur-md">
              <div className="flex items-center gap-2 mb-3.5">
                <AlertTriangle className="w-5.5 h-5.5 text-orange-500 stroke-[2.5]" />
                <h3 className="text-base font-bold text-slate-200 font-sans">Active Hydrological Warning Panel</h3>
              </div>

              {alerts.length === 0 ? (
                <div className="bg-slate-950/40 border border-slate-800/60 rounded-lg p-5 flex flex-col items-center justify-center text-center">
                  <CheckCircle className="w-10 h-10 text-green-500 mb-2" />
                  <p className="text-sm font-bold text-slate-300">All Groundwater Stations Safe</p>
                  <p className="text-xs text-slate-500 mt-1 font-sans">No threshold breaches or anomalous depletion rates logged.</p>
                </div>
              ) : (
                <div className="space-y-3 max-h-60 overflow-y-auto pr-1">
                  {(() => {
                    const activeAlert = alerts.find(a => a.station_id === selectedStation);
                    if (!activeAlert) return (
                      <div className="p-3 bg-slate-950/40 border border-slate-800/60 rounded-lg flex items-center gap-2.5 text-xs text-slate-400 font-sans">
                        <Info className="w-4.5 h-4.5 text-cyan-500" />
                        <span>No severe warnings logged for closest station {selectedStation}. System status nominal.</span>
                      </div>
                    );
                    return (
                      <div className={`p-4 border rounded-xl flex flex-col gap-2 ${getAlertColor(activeAlert.alert_level)}`}>
                        <div className="flex justify-between items-center">
                          <span className="text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 font-sans">
                            <span className={`w-2 h-2 rounded-full ring-4 ${getAlertDotColor(activeAlert.alert_level)}`}></span>
                            {activeAlert.alert_level} STATUS - Station {activeAlert.station_id}
                          </span>
                          <span className="text-[10px] bg-slate-950/60 px-2 py-0.5 rounded-full font-bold">Closest Node</span>
                        </div>
                        <ul className="text-xs space-y-1 text-slate-350 list-disc list-inside">
                          {activeAlert.reasons.map((r, i) => (
                            <li key={i}>{r}</li>
                          ))}
                        </ul>
                        <div className="text-[10px] text-slate-400 mt-1.5 border-t border-slate-800/40 pt-1.5 font-sans">
                          <span className="font-bold text-slate-300">Depletion Timeline:</span> {activeAlert.depletion_timeline}
                        </div>
                        <div className="text-[10px] text-slate-350 space-y-1 font-sans">
                          <span className="font-bold block text-slate-400">Actions Recommended:</span>
                          {activeAlert.recommended_actions.map((act, i) => (
                            <div key={i} className="flex gap-1.5 items-start pl-2">
                              <span>•</span>
                              <span>{act}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>

            {/* Recharts chart showing 14-day projection trajectory and confidence intervals */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 backdrop-blur-md">
              <h3 className="text-sm font-bold uppercase text-slate-400 mb-4 flex items-center gap-1.5 font-sans">
                <TrendingUp className="w-4.5 h-4.5 text-cyan-500" />
                Short-Term Daily Projections & Confidence Intervals (Next 14 Days)
              </h3>
              
              <div className="h-64 md:h-80 w-full bg-slate-950/40 border border-slate-900 rounded-xl p-3">
                {forecast ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={getProjectionsChartData()} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" stroke="#64748b" fontSize={10} />
                      <YAxis stroke="#64748b" fontSize={10} reversed domain={["auto", "auto"]} label={{ value: "MBGL (Deeps Down)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 10 }} />
                      <Tooltip contentStyle={{ backgroundColor: "#020617", border: "1px solid #334155" }} labelStyle={{ color: "#38bdf8", fontWeight: "bold" }} />
                      <Legend verticalAlign="top" height={36} iconType="circle" />
                      
                      <Line type="monotone" dataKey="P90_Deeper" name="P90 Deeper Limit" stroke="#ef4444" strokeDasharray="5 5" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="P50_Expected" name="P50 Expected (Trend)" stroke="#06b6d4" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                      <Line type="monotone" dataKey="P10_Shallower" name="P10 Shallow Limit" stroke="#22c55e" strokeDasharray="5 5" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-500 text-xs">No forecast projections loaded.</div>
                )}
              </div>
            </div>

            {/* Historical observations plot */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 backdrop-blur-md">
              <h3 className="text-sm font-bold uppercase text-slate-400 mb-4 flex items-center gap-1.5 font-sans">
                <Database className="w-4.5 h-4.5 text-cyan-500" />
                Historical observation cycles & Ingress Rainfall
              </h3>
              
              <div className="h-64 md:h-80 w-full bg-slate-950/40 border border-slate-900 rounded-xl p-3">
                {history.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={getHistoryChartData()} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorGw" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0}/>
                        </linearGradient>
                        <linearGradient id="colorRain" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" stroke="#64748b" fontSize={9} />
                      <YAxis stroke="#64748b" fontSize={9} reversed label={{ value: "GW Level (MBGL)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 10 }} />
                      <Tooltip contentStyle={{ backgroundColor: "#020617", border: "1px solid #334155" }} />
                      <Legend verticalAlign="top" height={36} />
                      
                      <Area type="monotone" dataKey="GW_Depth_MBGL" name="GW Level (MBGL)" stroke="#06b6d4" strokeWidth={2.5} fillOpacity={1} fill="url(#colorGw)" />
                      <Area type="monotone" dataKey="Effective_Rain_180d" name="Routed Rain 180d (mm)" stroke="#3b82f6" strokeWidth={1.5} fillOpacity={1} fill="url(#colorRain)" />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-500 text-xs">No historical observations available.</div>
                )}
              </div>
            </div>

            {/* Meteorological Open-Meteo card */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 backdrop-blur-md">
              <div className="flex items-center gap-2 mb-4 border-b border-slate-800/60 pb-2.5">
                <CloudRain className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-bold text-slate-200 font-sans">Local Weather Outlook (Open-Meteo API)</h3>
              </div>

              {weather ? (
                <div className="flex flex-col gap-6">
                  {/* Current Weather Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 bg-slate-950/40 border border-slate-800/60 rounded-xl p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-3 bg-cyan-950/60 rounded-xl text-cyan-400 border border-cyan-900/40">
                        <Thermometer className="w-5 h-5" />
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-bold block font-sans">Temperature</span>
                        <span className="text-base font-black text-slate-200">{weather.current.temperature.toFixed(1)}°C</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className="p-3 bg-blue-950/60 rounded-xl text-blue-400 border border-blue-900/40">
                        <Wind className="w-5 h-5" />
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-bold block font-sans">Wind Speed</span>
                        <span className="text-base font-black text-slate-200">{weather.current.wind_speed.toFixed(1)} km/h</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className="p-3 bg-cyan-950/60 rounded-xl text-cyan-400 border border-cyan-900/40">
                        <CloudRain className="w-5 h-5" />
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-bold block font-sans">Hourly Rain</span>
                        <span className="text-base font-black text-slate-200">{weather.current.rainfall.toFixed(2)} mm</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className="p-3 bg-slate-900 rounded-xl text-slate-400 border border-slate-800/40">
                        <Sun className="w-5 h-5" />
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-bold block font-sans">Condition</span>
                        <span className="text-sm font-bold text-slate-200 capitalize truncate block max-w-[120px]">{weather.current.weather_description}</span>
                      </div>
                    </div>
                  </div>

                  {/* 5-Day Forecast Grid */}
                  <div>
                    <span className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2.5 block font-sans">5-Day Meteorological Outlook</span>
                    <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                      {weather.daily.map((day, idx) => (
                        <div key={idx} className="bg-slate-950/40 border border-slate-800/60 rounded-lg p-3 text-center flex flex-col items-center">
                          <span className="text-xs text-slate-400 font-bold block mb-1 font-sans">
                            {idx === 0 ? "Today" : day.date.substring(5)}
                          </span>
                          <CloudRain className={`w-5 h-5 mb-2 ${day.rainfall > 5.0 ? "text-cyan-400 animate-pulse" : day.rainfall > 0.0 ? "text-slate-400" : "text-slate-600"}`} />
                          <span className="text-sm font-black text-slate-200 block">{day.temperature.toFixed(1)}°C</span>
                          <span className="text-[10px] text-slate-400 block mt-1">Rain: {day.rainfall.toFixed(1)}mm</span>
                          <span className="text-[9px] text-slate-500 block font-sans">Pop: {day.precipitation_probability.toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center p-6 text-slate-500 text-xs">No weather information loaded.</div>
              )}
            </div>

          </div>

        </div>
      )}

      {/* Footer System Info */}
      <footer className="border-t border-slate-900 mt-10 pt-4 text-center text-xs text-slate-600 flex flex-col sm:flex-row justify-between items-center gap-3">
        <span>© 2026 NEERA Platform. Developed under strict spatiotemporal and geographic validation constraints.</span>
        <span className="font-mono text-[9px] bg-slate-950/80 border border-slate-800/60 px-3 py-1 rounded-md text-slate-500">
          Core Model: CatBoost Validated • Mapping: Nominatim & Open-Meteo • Cache: OK
        </span>
      </footer>

    </div>
  );
}
