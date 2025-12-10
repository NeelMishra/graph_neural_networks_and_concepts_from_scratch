import numpy as np
from collections import deque

class IndependentCascade:

    def __init__(self, G, default_edge_prob = 0.1):
        self.G = G
        self.default_val = default_edge_prob
        
        self.nodes = list(self.G.nodes())
        self.idx_to_node = {idx : node for idx, node in enumerate(self.nodes)}
        self.node_to_idx = {node : idx for idx, node in enumerate(self.nodes)}
        self.idx_to_label = {idx : 0 for idx, node in enumerate(self.nodes)}
        self.idx_to_infected_time = {idx : -1 for idx in self.idx_to_node}

        self.init_edge_prob()

    def init_edge_prob(self):
        # Sets a default edge probability value for the edges with no probability value
        for edge in self.G.edges(data=True):
            u, v, data = edge
            data.setdefault('prob', self.default_val)
        return

    def set_labels(self, nodes, labels):
        for node, label in zip(nodes, labels):
            node_idx = self.node_to_idx[node]
            self.idx_to_label[node_idx] = label
        return
    
    def reset(self):
        self.idx_to_label = {idx : 0 for idx, node in enumerate(self.nodes)}
        self.idx_to_infected_time = {idx : -1 for idx in self.idx_to_node}
        return

    def bfs_traversal(self, source_nodes):
        self.reset()
        queue = deque()

        total_nodes_infected = source_nodes.copy()
        for node in source_nodes:
            i = self.node_to_idx[node]
            self.idx_to_label[i] = 1
            self.idx_to_infected_time[i] = 0
            queue.append([node, 0])

        while queue:
            current, time = queue.popleft()

            for neighbor in self.G.neighbors(current):
                neighbor_idx = self.node_to_idx[neighbor]
                successful_infection = (np.random.random()) < self.G[current][neighbor].get("prob", self.default_val)
                if self.idx_to_label[neighbor_idx] == 0 and successful_infection:
                    queue.append([neighbor, time + 1])
                    self.idx_to_label[neighbor_idx] = 1
                    self.idx_to_infected_time[neighbor_idx] = time+1
                    total_nodes_infected.append(neighbor)
        return total_nodes_infected


### LLM Generated Driver Code

import os, gzip, urllib.request
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------- Download + load real-world graph -------------------------

def _download(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, path)

def load_email_eu_core(data_dir="data/email_eu_core"):
    edges_gz  = os.path.join(data_dir, "email-Eu-core.txt.gz")

    _download("https://snap.stanford.edu/data/email-Eu-core.txt.gz", edges_gz)

    G = nx.DiGraph()
    with gzip.open(edges_gz, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = line.split()
            u, v = int(u), int(v)
            G.add_edge(u, v)

    # keep largest weakly connected component (helps diffusion be meaningful)
    if G.number_of_nodes() > 0:
        wcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(wcc).copy()

    return G

# ------------------------- Utilities: seeds, infection curve, exposure curve -------------------------

def choose_seeds_top_outdeg(G: nx.DiGraph, k=5):
    return [n for n, _ in sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:k]]

def infected_time_dict_from_ic(ic):
    # ic.idx_to_infected_time: idx -> time, ic.idx_to_node: idx -> node
    return {ic.idx_to_node[i]: ic.idx_to_infected_time[i] for i in ic.idx_to_infected_time}

def infection_time_series(infected_time):
    times = [t for t in infected_time.values() if t >= 0]
    if not times:
        return np.array([0]), np.array([0]), np.array([0])

    T = max(times)
    new = np.zeros(T + 1, dtype=int)
    for t in times:
        new[t] += 1
    cum = np.cumsum(new)
    return np.arange(T + 1), new, cum

def exposure_curve(G, infected_time, seed_set):
    """
    Exposure k for a node v:
      - if v adopted at time t_v: k = #predecessors infected at time < t_v
      - if v never adopted:       k = #predecessors infected at any time
    Then we estimate P(adopt | k) by binning nodes with EXACT exposure k.
    Seeds are excluded (since they are forced adopters).
    """
    directed = G.is_directed()

    def influencers(v):
        return G.predecessors(v) if directed else G.neighbors(v)

    ks = []
    ys = []

    for v in G.nodes():
        if v in seed_set:
            continue

        tv = infected_time.get(v, -1)
        if tv >= 0:
            k = 0
            for u in influencers(v):
                tu = infected_time.get(u, -1)
                if tu >= 0 and tu < tv:
                    k += 1
            y = 1
        else:
            k = 0
            for u in influencers(v):
                if infected_time.get(u, -1) >= 0:
                    k += 1
            y = 0

        ks.append(k)
        ys.append(y)

    if not ks:
        return np.array([0]), np.array([0.0]), np.array([0])

    max_k = max(ks)
    denom = np.zeros(max_k + 1, dtype=int)
    numer = np.zeros(max_k + 1, dtype=int)

    for k, y in zip(ks, ys):
        denom[k] += 1
        numer[k] += y

    prob = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom > 0)
    return np.arange(max_k + 1), prob, denom

# ------------------------- Run many cascades + average curves -------------------------

# knobs
EDGE_P = 0.05          # constant per-edge activation prob (you can change)
NUM_SEEDS = 5
RUNS = 30              # Monte-Carlo runs to smooth the curve
BASE_SEED = 0          # reproducibility

G = load_email_eu_core()
print(f"Loaded email-Eu-core: n={G.number_of_nodes()} m={G.number_of_edges()} (directed={G.is_directed()})")

seeds = choose_seeds_top_outdeg(G, k=NUM_SEEDS)
seed_set = set(seeds)
print("Seeds (top out-degree):", seeds)

# Accumulators for averaged exposure curve
total_numer = np.zeros(1, dtype=float)
total_denom = np.zeros(1, dtype=float)

cascade_sizes = []
final_infection_curves = []  # store cumulative infected for a few runs (optional)

for r in range(RUNS):
    np.random.seed(BASE_SEED + r)

    ic = IndependentCascade(G, default_edge_prob=EDGE_P)
    ic.bfs_traversal(seeds)
    infected_time = infected_time_dict_from_ic(ic)

    active = [n for n, t in infected_time.items() if t >= 0]
    cascade_sizes.append(len(active))

    # exposure curve for this run
    k_vals, prob, denom = exposure_curve(G, infected_time, seed_set)
    numer = prob * denom

    # grow accumulators if needed
    if len(k_vals) > len(total_numer):
        pad = len(k_vals) - len(total_numer)
        total_numer = np.pad(total_numer, (0, pad))
        total_denom = np.pad(total_denom, (0, pad))

    total_numer[:len(k_vals)] += numer
    total_denom[:len(k_vals)] += denom

    # infection curve (optional)
    t_axis, new_inf, cum_inf = infection_time_series(infected_time)
    final_infection_curves.append((t_axis, cum_inf))

avg_prob = np.divide(total_numer, total_denom, out=np.zeros_like(total_numer), where=total_denom > 0)
k_axis = np.arange(len(avg_prob))

print("\nCascade size stats over runs:")
print("  mean:", float(np.mean(cascade_sizes)))
print("  std :", float(np.std(cascade_sizes)))
print("  min :", int(np.min(cascade_sizes)))
print("  max :", int(np.max(cascade_sizes)))

# ------------------------- Plots -------------------------

# 1) Exposure curve
plt.figure()
plt.plot(k_axis, avg_prob, marker="o", linewidth=2)
plt.xlabel("k = # infected influencers (predecessors) before adoption")
plt.ylabel("P(adopt | k)  (binned by exact k)")
plt.title(f"Exposure curve on email-Eu-core (IC), p={EDGE_P}, seeds={NUM_SEEDS}, runs={RUNS}")
plt.grid(True)
plt.show()

# 2) Infection curve (show one representative run: the median cascade size run)
median_idx = int(np.argsort(cascade_sizes)[len(cascade_sizes)//2])
t_axis, cum_inf = final_infection_curves[median_idx]

plt.figure()
plt.plot(t_axis, cum_inf, marker="o", linewidth=2)
plt.xlabel("time step")
plt.ylabel("cumulative infected nodes")
plt.title("Cumulative infections over time (one representative run)")
plt.grid(True)
plt.show()
