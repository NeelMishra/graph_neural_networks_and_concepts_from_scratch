import networkx as nx
import numpy as np
from collections import deque

class LTM:

    def __init__(self, G: nx.DiGraph, threshold: float):
        self.G = G
        self.threshold = threshold

        self.nodes = list(self.G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        self.idx_to_label = {idx: 0 for idx in self.idx_to_node}

    def activate_nodes(self, nodes, labels):
        
        for node, label in zip(nodes, labels):
            node_idx = self.node_to_idx[node]
            self.idx_to_label[node_idx] = label


    def propogate(self):
        changed = []
        new_labels = self.idx_to_label.copy()

        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            if self.idx_to_label[node_idx] == 1:
                continue
            weighted_sum = 0

            for neighbor in self.G.predecessors(node):
                neighbor_idx = self.node_to_idx[neighbor]
                if self.idx_to_label[neighbor_idx] == 1:
                    weighted_sum += self.G[neighbor][node].get("weight", 0.0)
                

            if weighted_sum >= self.threshold:
                changed.append(node)
                new_labels[node_idx] = 1

        self.idx_to_label = new_labels
        return changed
    

    def fit(self, active_nodes, labels, epochs):
        self.activate_nodes(active_nodes, labels)
        for epoch in range(epochs):
            changed_nodes = self.propogate()
            if not changed_nodes:
                break
        return




####### LLM GENERATED DRIVER CODE


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- Your class (paste your LTM here) ----------
# (Make sure it has: activate_nodes(nodes, labels) and propogate())

# ---------- 1) Load real-world graph ----------
G_und = nx.karate_club_graph()  # real social network (34 nodes)

# Convert to DiGraph with reciprocal influence
G = nx.DiGraph()
G.add_nodes_from(G_und.nodes(data=True))
for u, v in G_und.edges():
    G.add_edge(u, v)
    G.add_edge(v, u)

# ---------- 2) Assign weights (incoming-normalized per destination) ----------
# b_{v,u} = 1 / indeg(v) so sum_u b_{v,u} = 1 for nodes with indeg>0
for v in G.nodes():
    preds = list(G.predecessors(v))
    if len(preds) == 0:
        continue
    w = 1.0 / len(preds)
    for u in preds:
        G[u][v]["weight"] = w

# ---------- 3) Pick seeds + run LTM ----------
threshold = 0.30  # try 0.2 .. 0.6
ltm = LTM(G, threshold=threshold)

# seed = top-k by degree (on undirected graph, easier intuition)
k = 2
seeds = sorted(G_und.degree, key=lambda x: x[1], reverse=True)[:k]
seed_nodes = [n for n, _ in seeds]
ltm.activate_nodes(seed_nodes, [1]*len(seed_nodes))

# Simulate step-by-step to capture history
max_steps = 50
history_active_sets = []
history_changed = []

def current_active_set():
    return {ltm.idx_to_node[i] for i, lab in ltm.idx_to_label.items() if lab == 1}

history_active_sets.append(current_active_set())
for t in range(max_steps):
    changed = ltm.propogate()
    history_changed.append(changed)
    history_active_sets.append(current_active_set())
    if not changed:
        break

print(f"Seeds: {seed_nodes}")
print(f"Steps run: {len(history_active_sets)-1}")
print(f"Final active: {len(history_active_sets[-1])}/{G.number_of_nodes()}")

# ---------- 4) Visualizations ----------
pos = nx.spring_layout(G_und, seed=7)  # fixed layout for consistent frames

# (A) Growth curve
active_counts = [len(s) for s in history_active_sets]
plt.figure()
plt.plot(active_counts, marker="o")
plt.title("LTM cascade size over time")
plt.xlabel("Step")
plt.ylabel("# Active nodes")
plt.show()

# (B) Final graph colored
final_active = history_active_sets[-1]
node_colors = ["tab:red" if n in final_active else "tab:gray" for n in G_und.nodes()]
plt.figure(figsize=(6, 5))
nx.draw_networkx(G_und, pos=pos, node_color=node_colors, with_labels=True)
plt.title(f"Final active set (threshold={threshold})")
plt.axis("off")
plt.show()

# (C) Animation: activation over time
fig, ax = plt.subplots(figsize=(6, 5))
ax.axis("off")

def update(frame):
    ax.clear()
    ax.axis("off")
    active = history_active_sets[frame]
    colors = ["tab:red" if n in active else "tab:gray" for n in G_und.nodes()]
    nx.draw_networkx(G_und, pos=pos, node_color=colors, with_labels=True, ax=ax)
    ax.set_title(f"Step {frame} | Active={len(active)}/{G_und.number_of_nodes()}")

anim = FuncAnimation(fig, update, frames=len(history_active_sets), interval=800, repeat=False)
plt.show()

# Optional: save animation (requires ffmpeg or pillow)
# anim.save("ltm_karate.gif", writer="pillow", fps=2)
# anim.save("ltm_karate.mp4", writer="ffmpeg", fps=2)
