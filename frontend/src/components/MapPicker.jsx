import { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default marker icons in React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

// Custom marker icons by priority tier
const createTierIcon = (tier) => {
  const colors = { 1: '#ef4444', 2: '#f59e0b', 3: '#3b82f6' };
  const color = colors[tier] || colors[3];
  return L.divIcon({
    className: 'custom-marker',
    html: `<div style="
      width: 28px; height: 28px; border-radius: 50%;
      background: ${color}; border: 3px solid white;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      display: flex; align-items: center; justify-content: center;
      color: white; font-weight: bold; font-size: 12px;
    ">${tier === 1 ? '🏥' : tier === 2 ? '🏫' : '📍'}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
};

const defaultIcon = L.divIcon({
  className: 'custom-marker',
  html: `<div style="
    width: 28px; height: 28px; border-radius: 50%;
    background: #6366f1; border: 3px solid white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: bold; font-size: 14px;
  ">+</div>`,
  iconSize: [28, 28],
  iconAnchor: [14, 14],
});

function MapClickHandler({ onMapClick, disabled }) {
  useMapEvents({
    click(e) {
      if (!disabled) {
        onMapClick(e.latlng);
      }
    },
  });
  return null;
}

function CityRecenter({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.setView(center, 13);
    }
  }, [center, map]);
  return null;
}

export default function MapPicker({ intersections, onIntersectionsChange, cityCenter, disabled, poiResults }) {
  const mapCenter = cityCenter || [28.6139, 77.2090]; // Default: Delhi

  const handleMapClick = (latlng) => {
    if (intersections.length >= 3) return; // Max 3
    const newPin = {
      lat: parseFloat(latlng.lat.toFixed(6)),
      lon: parseFloat(latlng.lng.toFixed(6)),
      name: `Pin ${intersections.length + 1}`,
    };
    onIntersectionsChange([...intersections, newPin]);
  };

  const removePin = (index) => {
    if (disabled) return;
    const updated = intersections.filter((_, i) => i !== index);
    // Rename remaining pins
    const renamed = updated.map((pin, i) => ({ ...pin, name: `Pin ${i + 1}` }));
    onIntersectionsChange(renamed);
  };

  // Merge POI results into markers if available
  const getMarkerData = (index) => {
    if (poiResults && poiResults[index]) {
      return poiResults[index];
    }
    return null;
  };

  return (
    <div className="map-picker-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <span style={{ fontSize: '13px', color: '#94a3b8' }}>
          Click map to place pins ({intersections.length}/3)
        </span>
        {intersections.length > 0 && !disabled && (
          <button
            onClick={() => onIntersectionsChange([])}
            style={{
              fontSize: '11px', padding: '2px 8px', borderRadius: '4px',
              background: 'rgba(239,68,68,0.2)', color: '#f87171',
              border: '1px solid rgba(239,68,68,0.3)', cursor: 'pointer',
            }}
          >
            Clear All
          </button>
        )}
      </div>
      
      <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
        <MapContainer
          center={mapCenter}
          zoom={13}
          style={{ height: '280px', width: '100%' }}
          scrollWheelZoom={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />
          <CityRecenter center={mapCenter} />
          <MapClickHandler onMapClick={handleMapClick} disabled={disabled || intersections.length >= 3} />
          
          {intersections.map((pin, index) => {
            const poiData = getMarkerData(index);
            const tier = poiData ? poiData.tier : null;
            const icon = tier ? createTierIcon(tier) : defaultIcon;
            
            return (
              <Marker
                key={index}
                position={[pin.lat, pin.lon]}
                icon={icon}
                eventHandlers={{
                  click: () => removePin(index),
                }}
              >
                <Popup>
                  <div style={{ color: '#1e293b', fontSize: '12px' }}>
                    <strong>{pin.name}</strong>
                    <br />
                    {pin.lat.toFixed(4)}, {pin.lon.toFixed(4)}
                    {poiData && (
                      <>
                        <br />
                        <span style={{ fontWeight: 'bold', color: tier === 1 ? '#dc2626' : tier === 2 ? '#d97706' : '#2563eb' }}>
                          {poiData.tier_emoji} {poiData.tier_label}
                        </span>
                        {poiData.detected_pois?.length > 0 && (
                          <>
                            <br />
                            <em>{poiData.detected_pois.slice(0, 3).join(', ')}</em>
                          </>
                        )}
                      </>
                    )}
                    {!disabled && <><br /><em style={{ color: '#94a3b8' }}>Click to remove</em></>}
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>

      {/* Pin list */}
      {intersections.length > 0 && (
        <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {intersections.map((pin, i) => {
            const poiData = getMarkerData(i);
            return (
              <div key={i} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '6px 10px', borderRadius: '8px',
                background: 'rgba(255,255,255,0.05)', fontSize: '12px',
              }}>
                <span>
                  {poiData ? poiData.tier_emoji : '📍'} <strong>{pin.name}</strong>
                  <span style={{ color: '#64748b', marginLeft: '6px' }}>
                    {pin.lat.toFixed(4)}, {pin.lon.toFixed(4)}
                  </span>
                  {poiData && (
                    <span style={{ 
                      marginLeft: '8px', padding: '1px 6px', borderRadius: '4px', fontSize: '10px',
                      background: poiData.tier === 1 ? 'rgba(239,68,68,0.2)' : poiData.tier === 2 ? 'rgba(245,158,11,0.2)' : 'rgba(59,130,246,0.2)',
                      color: poiData.tier === 1 ? '#f87171' : poiData.tier === 2 ? '#fbbf24' : '#60a5fa',
                    }}>
                      {poiData.tier_label}
                    </span>
                  )}
                </span>
                {!disabled && (
                  <button onClick={() => removePin(i)} style={{
                    background: 'none', border: 'none', color: '#ef4444',
                    cursor: 'pointer', fontSize: '14px', padding: '0 4px',
                  }}>✕</button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
