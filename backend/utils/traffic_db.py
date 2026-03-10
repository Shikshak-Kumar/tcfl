import sqlite3
import datetime
import os
from typing import List, Dict, Any, Optional

# For PostgreSQL support on Render/production
try:
    import psycopg2
    import psycopg2.extras
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "traffic_data.db")

# Predefined time slots for aggregation
TIME_SLOTS = [
    {"name": "Morning Rush", "start": "05:00", "end": "06:00"},
    {"name": "Mid-Morning", "start": "10:00", "end": "11:00"},
    {"name": "Afternoon", "start": "13:30", "end": "14:30"},
    {"name": "Evening Rush", "start": "17:00", "end": "18:00"},
    {"name": "Night", "start": "21:00", "end": "22:00"},
]

def get_connection():
    """Returns a connection to either SQLite or PostgreSQL based on environment."""
    if DATABASE_URL and DATABASE_URL.startswith("postgres"):
        if not HAS_POSTGRES:
            raise ImportError("psycopg2 is required for PostgreSQL. Please install it.")
        conn = psycopg2.connect(DATABASE_URL)
        return conn, "postgres"
    else:
        conn = sqlite3.connect(DB_PATH)
        return conn, "sqlite"

def init_db():
    """Initializes the database schema."""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    if db_type == "postgres":
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_records (
                id SERIAL PRIMARY KEY,
                location_id TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                current_speed REAL,
                free_flow_speed REAL,
                congestion_ratio REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_time ON traffic_records(location_id, timestamp)')
    else:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_id TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                current_speed REAL,
                free_flow_speed REAL,
                congestion_ratio REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_time ON traffic_records(location_id, timestamp)')
    
    conn.commit()
    conn.close()

def insert_record(location_id: str, lat: float, lon: float, current_speed: float, free_flow_speed: float, congestion_ratio: float, timestamp: Optional[datetime.datetime] = None):
    """Inserts a new traffic record into the database."""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    if timestamp:
        ts = timestamp
    else:
        ts = datetime.datetime.now()

    placeholder = "%s" if db_type == "postgres" else "?"
    
    query = f'''
        INSERT INTO traffic_records (location_id, lat, lon, current_speed, free_flow_speed, congestion_ratio, timestamp)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
    '''
    
    cursor.execute(query, (location_id, lat, lon, current_speed, free_flow_speed, congestion_ratio, ts))
    
    conn.commit()
    conn.close()

def cleanup_old_records(days: int = 30):
    """Deletes traffic records older than the specified number of days."""
    conn, db_type = get_connection()
    cursor = conn.cursor()
    
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    placeholder = "%s" if db_type == "postgres" else "?"
    
    cursor.execute(f'DELETE FROM traffic_records WHERE timestamp < {placeholder}', (cutoff,))
    deleted_count = cursor.rowcount
    
    conn.commit()
    conn.close()
    if deleted_count > 0:
        print(f"[Database] Cleaned up {deleted_count} records older than {days} days.")
    return deleted_count

def get_time_slot_aggregations(location_id: Optional[str] = None, target_date: Optional[datetime.date] = None) -> Dict[str, Any]:
    """Retrieves and aggregates traffic data for the predefined time slots."""
    if target_date is None:
        target_date = datetime.date.today()
        
    conn, db_type = get_connection()
    if db_type == "postgres":
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    else:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
    
    results = {
        "date": target_date.isoformat(),
        "location_id": location_id if location_id else "all",
        "slots": []
    }
    
    for slot in TIME_SLOTS:
        start_time = slot["start"]
        end_time = slot["end"]
        
        if db_type == "postgres":
            # Postgres date/time extraction
            query = '''
                SELECT 
                    COUNT(*) as record_count,
                    AVG(current_speed) as avg_speed,
                    AVG(free_flow_speed) as avg_free_flow,
                    AVG(congestion_ratio) as avg_congestion
                FROM traffic_records
                WHERE timestamp::date = %s
                AND timestamp::time >= %s::time
                AND timestamp::time < %s::time
            '''
            params = [target_date, start_time, end_time]
        else:
            # SQLite date/time extraction
            query = '''
                SELECT 
                    COUNT(*) as record_count,
                    AVG(current_speed) as avg_speed,
                    AVG(free_flow_speed) as avg_free_flow,
                    AVG(congestion_ratio) as avg_congestion
                FROM traffic_records
                WHERE date(timestamp) = ?
                AND time(timestamp) >= ? 
                AND time(timestamp) < ?
            '''
            params = [target_date.isoformat(), start_time + ":00", end_time + ":00"]
        
        if location_id:
            placeholder = "%s" if db_type == "postgres" else "?"
            query += f" AND location_id = {placeholder}"
            params.append(location_id)
            
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        record_count = row['record_count'] if row['record_count'] else 0
        
        if record_count > 0:
            avg_speed = round(float(row['avg_speed']), 2)
            avg_free_flow = round(float(row['avg_free_flow']), 2)
            avg_congestion = round(float(row['avg_congestion']), 2)
            estimated_density = max(5, round((avg_congestion - 0.5) * 40)) 
            
            if avg_congestion < 1.1: level = "Low"
            elif avg_congestion < 1.5: level = "Moderate"
            elif avg_congestion < 2.0: level = "High"
            else: level = "Severe"
        else:
            avg_speed, avg_free_flow, avg_congestion, estimated_density, level = 0, 0, 0, 0, "Unknown"
            
        results["slots"].append({
            "slot_name": slot["name"],
            "time_range": f"{start_time}-{end_time}",
            "record_count": record_count,
            "metrics": {
                "avg_speed_kmh": avg_speed,
                "avg_free_flow_kmh": avg_free_flow,
                "congestion_ratio": avg_congestion,
                "congestion_level": level,
                "estimated_density_veh_km": estimated_density
            }
        })
        
    conn.close()
    return results

# Initialize DB on import
init_db()
