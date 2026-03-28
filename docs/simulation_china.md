AdaptFlow China OSM — Full Results Analysis (3 Rounds, 6 Nodes)
Run Configuration
Field	Value
Mode	SUMO-GUI (real TraCI microsimulation)
Scenario	sumo_configs_china_osm
Nodes	6 (node_0 … node_5)
Clusters	2
Steps per episode	200
Total vehicles per run	94 departed
SUMO configs cycling	osm_client1.sumocfg / osm_client2.sumocfg alternating
Per-Node Metrics — All Rounds
Round 1 (Static Clustering — Initial Assignment)
Cluster 0 = {node_0, node_1, node_2} · Cluster 1 = {node_3, node_4, node_5}

Node	Cluster	Reward	Avg Wait (s)	Avg Queue	Max Queue	Throughput	Loss
node_0	cluster_0	193.80	9.234	0.170	2	24.47%	0.5358
node_1	cluster_0	196.80	8.989	0.115	1	24.47%	0.5465
node_2	cluster_0	191.95	9.436	0.225	2	24.47%	0.5462
node_3	cluster_1	196.79	9.277	0.110	1	24.47%	0.5324
node_4	cluster_1	187.59	9.574	0.350	2	24.47%	0.5077
node_5	cluster_1	193.76	9.351	0.210	2	24.47%	0.5856
Best performer: node_1 (reward 196.80, wait 8.99 s, queue 0.115)
Runner-up: node_3 (reward 196.79, lowest queue 0.110)
Worst performer: node_4 (reward 187.59, highest queue 0.35)
Reward spread: 9.21 points (healthily heterogeneous)
Loss spread: 0.5077 – 0.5856 — all nodes still actively learning
Round 2 (Dynamic Re-Clustering triggered by congestion fingerprints)
5 nodes changed cluster. node_0, node_2 → cluster_1; node_3, node_4, node_5 → cluster_0. Cluster 0 = {node_1, node_3, node_4, node_5} · Cluster 1 = {node_0, node_2}

Node	Cluster	Reward	Avg Wait (s)	Avg Queue	Max Queue	Loss
node_0	cluster_1	185.85 ▼	9.755 ▲	0.330 ▲	2	0.4501 ▼
node_1	cluster_0	185.85 ▼	9.755 ▲	0.330 ▲	2	0.4770 ▼
node_2	cluster_1	194.82	9.266	0.165	2	0.4720 ▼
node_3	cluster_0	185.85 ▼	9.755 ▲	0.330 ▲	2	0.4651 ▼
node_4	cluster_0	185.85 ▼	9.755 ▲	0.330 ▲	2	0.4069 ▼
node_5	cluster_0	185.85 ▼	9.755 ▲	0.330 ▲	2	0.4428 ▼
5 of 6 nodes collapsed to identical reward (185.85), wait (9.755 s), queue (0.33) — a homogenisation effect
Only node_2 retained distinct performance (reward 194.82, wait 9.266 s, queue 0.165) after being paired with node_0 in cluster_1
All losses dropped significantly (~0.05–0.10) — networks are converging even as outcome metrics declined
Round 3 (Re-Clustering again — node_1 → cluster_1, node_2 → cluster_0)
Cluster 0 = {node_2, node_3, node_4, node_5} · Cluster 1 = {node_0, node_1}

Node	Cluster	Reward	Avg Wait (s)	Avg Queue	Max Queue	Loss
node_0	cluster_1	185.85	9.755	0.330	2	0.4506
node_1	cluster_1	185.85	9.755	0.330	2	0.4690
node_2	cluster_0	185.85 ▼	9.755 ▲	0.330 ▲	2	0.4318 ▼
node_3	cluster_0	185.85	9.755	0.330	2	0.4658
node_4	cluster_0	185.85	9.755	0.330	2	0.3891 ▼
node_5	cluster_0	185.85	9.755	0.330	2	0.4327
All 6 nodes now perfectly identical in reward, wait, and queue — full convergence
node_2's round-2 advantage was erased when it was moved into cluster_0 alongside 3 heavy congestion nodes
node_4 leads in learning rate: its loss (0.3891) is lowest of any node across all rounds
Node Trajectories (Reward across rounds)
Node	Round 1	Round 2	Round 3	Trend
node_0	193.80	185.85	185.85	▼ then flat
node_1	196.80	185.85	185.85	▼ sharply then flat
node_2	191.95	194.82	185.85	▲ then ▼
node_3	196.79	185.85	185.85	▼ sharply then flat
node_4	187.59	185.85	185.85	▼ slightly then flat
node_5	193.76	185.85	185.85	▼ then flat
Loss Trajectory (Learning Progress)
Node	Round 1	Round 2	Round 3	Total Drop
node_0	0.5358	0.4501	0.4506	−0.085
node_1	0.5465	0.4770	0.4690	−0.078
node_2	0.5462	0.4720	0.4318	−0.114
node_3	0.5324	0.4651	0.4658	−0.067
node_4	0.5077	0.4069	0.3891	−0.119
node_5	0.5856	0.4428	0.4327	−0.153
node_5 started with the highest loss but dropped the most in absolute terms. node_4 consistently has the lowest loss at every round — indicating the fastest, cleanest gradient convergence.

Clustering Dynamics
Round	Transitions	Stability
1	None (static init)	N/A
2	5 nodes switched clusters	Highly unstable
3	2 nodes switched (node_1, node_2)	Partially stable
The root cause of instability: by round 2 all fingerprints except node_2's converged to identical values [9.755, 0.33, 0.245, 2.0, 0.0, 0.0]. When fingerprints are identical, k-means++ ties break arbitrarily — cluster membership becomes random noise, not meaningful congestion grouping.

By round 3, all 6 fingerprints are identical → similarity matrix is uniformly 1.0 everywhere → any clustering assignment is equivalent. The algorithm has no useful signal to work with.