import sqlite3
import os

DB_PATH = "traffic_data.db"

def view_recent_records(limit=15):
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, location_id, current_speed, free_flow_speed, congestion_ratio, timestamp 
            FROM traffic_records 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        headers = ["ID", "Location", "Speed", "Free Flow", "Congestion", "Timestamp"]
        
        print(f"\n--- Showing last {len(rows)} traffic records ---")
        # Manual formatting for simplicity
        format_str = "{:<4} {:<12} {:<8} {:<10} {:<10} {:<20}"
        print(format_str.format(*headers))
        print("-" * 70)
        for row in rows:
            # Round speeds and congestion for cleaner display
            clean_row = (row[0], row[1], round(row[2], 1), round(row[3], 1), round(row[4], 2), row[5])
            print(format_str.format(*clean_row))
            
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    view_recent_records()
