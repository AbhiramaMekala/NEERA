"use client";

import React, { useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix leaflet icon issue for default markers if any are used
if (typeof window !== "undefined") {
  // @ts-ignore
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
    iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
    shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
  });
}

interface MapMarker {
  station_id: string;
  latitude: number;
  longitude: number;
  depth: number;
  cluster: number;
}

interface MapComponentProps {
  markers: MapMarker[];
  selectedStationId: string;
  onStationSelect: (stationId: string) => void;
  onMapClick: (lat: number, lon: number) => void;
}

// Sub-component to sync map view focus to selected station coordinates
function MapRecenter({ lat, lon }: { lat: number; lon: number }) {
  const map = useMap();
  useEffect(() => {
    if (lat && lon) {
      map.setView([lat, lon], 9, { animate: true });
    }
  }, [lat, lon, map]);
  return null;
}

// Sub-component to handle map click events
function MapEventsHandler({ onMapClick }: { onMapClick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      onMapClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function MapComponent({
  markers,
  selectedStationId,
  onStationSelect,
  onMapClick,
}: MapComponentProps) {
  // Default map center (center of Karnataka)
  const defaultCenter: [number, number] = [15.3173, 75.7139];
  const defaultZoom = 7;

  // Find coordinates of selected station to recenter map if found
  const selectedMarker = markers.find(m => m.station_id === selectedStationId);
  const selectedCoords = selectedMarker
    ? { lat: selectedMarker.latitude, lon: selectedMarker.longitude }
    : null;

  return (
    <div className="w-full h-[500px] rounded-xl overflow-hidden border border-slate-800 shadow-lg relative z-10">
      <MapContainer
        center={defaultCenter}
        zoom={defaultZoom}
        style={{ width: "100%", height: "100%", background: "#020617" }}
      >
        {/* Dark map style from CartoDB Voyager/DarkMatter */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        <MapEventsHandler onMapClick={onMapClick} />

        {selectedCoords && (
          <MapRecenter lat={selectedCoords.lat} lon={selectedCoords.lon} />
        )}

        {/* Plot all markers */}
        {markers.map(marker => {
          const isSelected = marker.station_id === selectedStationId;
          const color = marker.depth > 30.0 ? "#ef4444" : marker.depth > 15.0 ? "#f97316" : "#22c55e";
          
          return (
            <CircleMarker
              key={marker.station_id}
              center={[marker.latitude, marker.longitude]}
              radius={isSelected ? 10 : 6}
              fillColor={color}
              color={isSelected ? "#ffffff" : "#020617"}
              weight={isSelected ? 2 : 1}
              fillOpacity={0.8}
              eventHandlers={{
                click: (e) => {
                  // Prevent map click triggering
                  L.DomEvent.stopPropagation(e);
                  onStationSelect(marker.station_id);
                }
              }}
            >
              <Popup className="leaflet-popup-dark">
                <div className="text-slate-900 font-sans p-1">
                  <div className="font-bold text-sm">Station {marker.station_id}</div>
                  <div className="text-xs mt-1">Groundwater Depth: <span className="font-semibold text-cyan-600">{marker.depth.toFixed(2)}m MBGL</span></div>
                  <div className="text-[10px] text-slate-500 mt-1">Cluster {marker.cluster} • Coordinates: {marker.latitude.toFixed(4)}, {marker.longitude.toFixed(4)}</div>
                  <button 
                    onClick={() => onStationSelect(marker.station_id)}
                    className="mt-2 w-full text-center py-1 text-[10px] bg-cyan-600 hover:bg-cyan-700 text-white rounded font-bold cursor-pointer"
                  >
                    Select Station
                  </button>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}
