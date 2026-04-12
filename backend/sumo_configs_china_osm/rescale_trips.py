"""
Rescales osm.passenger.trips.xml for the China OSM scenario so that
exactly 500 vehicles depart within the first 800 simulation seconds.

New period = 800 / 500 = 1.6s between successive departures.
"""
import xml.etree.ElementTree as ET
import os

TARGET_VEHICLES = 500
EPISODE_DURATION = 800
NEW_PERIOD = EPISODE_DURATION / TARGET_VEHICLES  # 1.6s

def rescale_trips(in_path, out_path, new_period):
    tree = ET.parse(in_path)
    root = tree.getroot()
    trips = root.findall('trip')
    print(f"\nProcessing: {in_path}")
    print(f"  Total trips found: {len(trips)}")

    for i, trip in enumerate(trips):
        new_depart = round(i * new_period, 2)
        trip.set('depart', f'{new_depart:.2f}')
        trip.set('id', str(i))

    in_ep = sum(1 for t in trips if float(t.get('depart', 9999)) < EPISODE_DURATION)
    print(f"  New period:                  {new_period}s")
    print(f"  Departing before t={EPISODE_DURATION}s: {in_ep}  (target: {TARGET_VEHICLES})")
    print(f"  Last depart time:            {trips[-1].get('depart')}s")

    ET.indent(tree, space='    ')
    tree.write(out_path, encoding='unicode', xml_declaration=False)

    with open(out_path, 'r', encoding='utf-8') as f:
        content = f.read()

    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n\n'
        f'<!-- Rescaled by rescale_trips.py: period={new_period:.2f}s, '
        f'{TARGET_VEHICLES} vehicles in first {EPISODE_DURATION}s -->\n\n'
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(header + content)

    print(f"  Written to: {out_path}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))

    rescale_trips(
        os.path.join(base, 'osm.passenger.trips.xml'),
        os.path.join(base, 'osm.passenger.trips.xml'),
        NEW_PERIOD,
    )

    print(f"\nDone! Trip file rescaled.")
    print(f"Period: {NEW_PERIOD}s  =>  {TARGET_VEHICLES} vehicles in first {EPISODE_DURATION}s episode.")
