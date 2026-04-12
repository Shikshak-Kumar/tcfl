# Train AdaptFlow on Pikhuwa (UP) rural OSM map — headless / no GUI
# Results stored in: results_adaptflow_india_rural_pikhuwa_osm\
# Map: pikhuwa/  | Nodes: 6 | Clusters: 2 | Rounds: 5 | Steps: 800
# Trip file: density=8.2 → ~499 vehicles per 800-step episode

python train_adaptflow.py `
    --nodes 6 `
    --clusters 2 `
    --rounds 5 `
    --steps 800 `
    --sumo-scenario pikhuwa_osm `
    --sumo-headless
