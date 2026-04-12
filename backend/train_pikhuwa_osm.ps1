# Train AdaptFlow on Pikhuwa (UP) rural OSM map — GUI mode
# Results stored in: results_adaptflow_india_rural_pikhuwa_osm\
# Map: pikhuwa/  | Nodes: 6 | Clusters: 2 | Rounds: 5 | Steps: 2000
# Trip file: density=6.5 → ~495 departed, ~396 arrived, ~80% TP

python train_adaptflow.py `
    --nodes 6 `
    --clusters 2 `
    --rounds 5 `
    --steps 2000 `
    --sumo-scenario pikhuwa_osm `
    --gui
