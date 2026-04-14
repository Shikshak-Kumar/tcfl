# AdaptFlow — China Rural OSM (1500 steps, ~515 departed/node, GUI)
# Trip file: density=1.6, end=4500, --validate
# Config:    6 nodes, 2 clusters, 5 rounds, 1500 steps
# Results:   results_adaptflow_china_rural_osm
# Compare with: results_adaptflow_india_rural_pikhuwa_osm (same config)

python train_adaptflow.py `
    --nodes 6 `
    --clusters 2 `
    --rounds 5 `
    --steps 1500 `
    --sumo-scenario china_rural_osm `
    --gui
