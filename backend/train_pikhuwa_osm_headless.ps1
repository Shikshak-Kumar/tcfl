# Train AdaptFlow on Pikhuwa (UP) rural OSM map — headless / no GUI
# Results stored in: results_adaptflow_india_rural_pikhuwa_osm\
# Map: pikhuwa/  | Nodes: 6 | Clusters: 2 | Rounds: 5 | Steps: 200

python train_adaptflow.py `
    --nodes 6 `
    --clusters 2 `
    --rounds 5 `
    --steps 200 `
    --sumo-scenario pikhuwa_osm `
    --sumo-headless
