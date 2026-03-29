import { useState, useEffect, useRef } from 'react';

export default function LocationInput({ intersections, onIntersectionsChange, city, disabled }) {
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const debounceRef = useRef(null);
  const containerRef = useRef(null);

  // Close suggestions on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Debounced search as user types
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    
    if (!inputValue.trim() || inputValue.trim().length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      setIsLoading(true);
      try {
        const searchQuery = `${inputValue}, ${city}, India`;
        const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=5&addressdetails=1`;
        const response = await fetch(url, {
          headers: { 'User-Agent': 'TrafficSimApp/1.0' }
        });
        const results = await response.json();
        const mapped = results.map(r => ({
          lat: parseFloat(r.lat),
          lon: parseFloat(r.lon),
          displayName: r.display_name.split(',').slice(0, 3).join(', '),
          shortName: r.display_name.split(',')[0],
          type: r.type || '',
        }));
        setSuggestions(mapped);
        setShowSuggestions(mapped.length > 0);
      } catch (err) {
        console.error('Search failed:', err);
        setSuggestions([]);
      }
      setIsLoading(false);
    }, 400); // 400ms debounce

    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [inputValue, city]);

  const selectSuggestion = (suggestion) => {
    // Limit removed as per user request
    const newPin = {
      lat: suggestion.lat,
      lon: suggestion.lon,
      name: suggestion.shortName,
    };
    onIntersectionsChange([...intersections, newPin]);
    setInputValue('');
    setSuggestions([]);
    setShowSuggestions(false);
    setError('');
  };

  const removeLocation = (index) => {
    if (disabled) return;
    onIntersectionsChange(intersections.filter((_, i) => i !== index));
  };

  return (
    <div ref={containerRef}>
      {/* Input with suggestions */}
      <div style={{ position: 'relative' }}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onFocus={() => { if (suggestions.length > 0) setShowSuggestions(true); }}
          disabled={disabled}
          placeholder={`Search a place in ${city}...`}
          style={{
            width: '100%', padding: '10px 14px', borderRadius: '8px', fontSize: '13px',
            background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
            color: '#e2e8f0', outline: 'none', boxSizing: 'border-box',
          }}
        />
        {isLoading && (
          <span style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', color: '#64748b', fontSize: '12px' }}>
            ...
          </span>
        )}

        {/* Suggestion Dropdown */}
        {showSuggestions && suggestions.length > 0 && (
          <div style={{
            position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 50,
            marginTop: '4px', borderRadius: '8px', overflow: 'hidden',
            background: '#1e293b', border: '1px solid rgba(255,255,255,0.15)',
            boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
          }}>
            {suggestions.map((s, i) => (
              <div
                key={i}
                onClick={() => selectSuggestion(s)}
                style={{
                  padding: '10px 14px', cursor: 'pointer', fontSize: '13px',
                  borderBottom: i < suggestions.length - 1 ? '1px solid rgba(255,255,255,0.06)' : 'none',
                  transition: 'background 0.15s',
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(99,102,241,0.15)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
              >
                <div style={{ color: '#e2e8f0', fontWeight: '500' }}>{s.shortName}</div>
                <div style={{ color: '#64748b', fontSize: '11px', marginTop: '2px' }}>{s.displayName}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <p style={{ color: '#f87171', fontSize: '12px', marginTop: '6px' }}>{error}</p>
      )}

      {/* Location list */}
      <div className="custom-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '8px', maxHeight: '200px', overflowY: 'auto', paddingRight: '4px' }}>
        {intersections.map((loc, i) => (
          <div key={i} style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 12px', borderRadius: '8px',
            background: 'rgba(255,255,255,0.06)', fontSize: '13px',
            border: '1px solid rgba(255,255,255,0.08)',
          }}>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontWeight: '600', color: '#e2e8f0', fontSize: '11px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '180px' }}>📍 {loc.name}</span>
              <span style={{ color: '#64748b', fontSize: '10px' }}>
                ({loc.lat.toFixed(3)}, {loc.lon.toFixed(3)})
              </span>
            </div>
            {!disabled && (
              <button 
                onClick={() => removeLocation(i)} 
                title="Remove location"
                style={{
                  background: 'none', border: 'none', color: '#f87171',
                  cursor: 'pointer', fontSize: '14px', padding: '0 4px',
                  opacity: 0.6, transition: 'opacity 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                onMouseLeave={(e) => e.currentTarget.style.opacity = '0.6'}
              >✕</button>
            )}
          </div>
        ))}
        {intersections.length === 0 && <div className="text-center py-4 text-[10px] text-slate-600 italic">No pins placed yet</div>}
      </div>

      <p style={{ color: '#64748b', fontSize: '10px', marginTop: '6px', textAlign: 'center' }}>
        {intersections.length} location{intersections.length !== 1 ? 's' : ''} • Search & place on map
      </p>
    </div>
  );
}
