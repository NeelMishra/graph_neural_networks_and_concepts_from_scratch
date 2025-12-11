import networkx as nx
import numpy as np
from collections import deque
from heapq import heappush, heapreplace, nsmallest

class sketch_greedy_hill_climbing:

    def __init__(self, G: nx.DiGraph, worlds: int, budget_k: int, sketch_size: int, default_edge_prob: float=0.1):
        self.G = G
        self.R = worlds
        self.K = budget_k
        self.sketch_size = sketch_size

        self.nodes = list(self.G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        self.idx_to_sketch = [[[] for _ in range(self.n)] for _ in range(self.R)]

        self.seed_set = set()
        self.default_edge_prob = default_edge_prob

        self.current_influence_set = {}
        self.infected_nodes = [set() for _ in range(self.R)]

        self.random_worlds = []
        self.generate_random_worlds()
        
        for random_world in self.random_worlds:
            self.assign_random_rank(random_world)

        self.build_sketches()
    
    def solve(self):
        for _ in range(self.K):
            u, _ = self.iterate()
            if u is None:
                break
            self.seed_set.add(u)
            self.mark_infected_by_seed(u)

        # temporarily ignore infected mask to get total influence
        saved_infected = self.infected_nodes
        self.infected_nodes = [set() for _ in range(self.R)]
        total_inf = self.estimate_influence(self.seed_set)
        self.infected_nodes = saved_infected

        return self.seed_set, total_inf

    
    def iterate(self):
        best_fit = -float('inf')
        best_candidate = None

        for node in self.nodes:
            if node in self.seed_set:
                continue

            influence_val = self.estimate_influence(self.seed_set.union({node}))
            if influence_val > best_fit:
                best_candidate = node
                best_fit = influence_val

        return best_candidate, best_fit

    def build_sketches(self):
        for world_idx, random_world in enumerate(self.random_worlds):
            order = sorted(random_world.nodes(), key=lambda v: random_world.nodes[v]['rank'])
            for node in order:
                self.reverse_bfs(node, random_world, world_idx)

    
    def reverse_bfs(self, source, random_world, world_idx):
        queue = deque([source])
        r = random_world.nodes[source]["rank"]
        elem = source  # element identity in this world
        visited = {source}

        while queue:
            cur = queue.popleft()
            cur_idx = self.node_to_idx[cur]
            heap_neg = self.idx_to_sketch[world_idx][cur_idx]  # [(-rank, elem), ...]

            if len(heap_neg) == self.sketch_size and (-heap_neg[0][0] <= r):
                continue

            changed = False
            if len(heap_neg) < self.sketch_size:
                heappush(heap_neg, (-r, elem))
                changed = True
            else:
                if -heap_neg[0][0] > r:
                    heapreplace(heap_neg, (-r, elem))
                    changed = True

            if not changed:
                continue

            for parent in random_world.predecessors(cur):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
    
    def estimate_influence(self, seed_set):
        """
        Sketch-based influence estimate.
        For each world:
          union bottom-c sketches of seeds (dedupe by elem), ignore infected,
          if union size < c -> exact, else -> (c-1)/tau where tau is c-th smallest rank.
        Returns average across worlds.
        """
        total = 0.0
        c = self.sketch_size

        for world_idx in range(self.R):
            best_rank_by_elem = {}
            infected = self.infected_nodes[world_idx]

            for s in seed_set:
                s_idx = self.node_to_idx[s]
                for neg_r, elem in self.idx_to_sketch[world_idx][s_idx]:
                    if elem in infected:
                        continue
                    r = -neg_r
                    prev = best_rank_by_elem.get(elem)
                    if prev is None or r < prev:
                        best_rank_by_elem[elem] = r

            m = len(best_rank_by_elem)
            if m == 0:
                continue

            if m < c:
                total += m
            else:
                tau = nsmallest(c, best_rank_by_elem.values())[-1]
                total += (c - 1) / tau

        return total / self.R

    def generate_random_worlds(self):

        random_worlds = []
        for world_idx in range(self.R):
            current_world = self.G.copy()
            curr_edges = list(current_world.edges(data=True))

            for edge in curr_edges:
                src, dest, attr = edge
                prob = attr.get('prob', self.default_edge_prob)

                if np.random.random() > prob:
                    current_world.remove_edge(src, dest)
            random_worlds.append(current_world)


        self.random_worlds = random_worlds

    def assign_random_rank(self, random_graph):

        for node in random_graph.nodes:
            random_graph.nodes[node]['rank'] = np.random.random()

    def forward_reach(self, seed, world_idx):
        G = self.random_worlds[world_idx]
        q = deque([seed])
        seen = {seed}

        while q:
            cur = q.popleft()
            for nxt in G.successors(cur):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return seen

    def mark_infected_by_seed(self, seed):
        for world_idx in range(self.R):
            reached = self.forward_reach(seed, world_idx)
            self.infected_nodes[world_idx].update(reached)

#### LLM generated driver code

if __name__ == "__main__":
    # ----- Build a small test graph -----
    # Example graph:
    # 0 -> 1 -> 2
    # 0 -> 3 -> 2
    # 1 -> 3
    G = nx.DiGraph()
    edge_prob = 0.5  # default per-edge probability if you don't set 'prob'

    # Add edges with 'prob' attribute
    G.add_edge(0, 1, prob=0.8)
    G.add_edge(1, 2, prob=0.6)
    G.add_edge(0, 3, prob=0.7)
    G.add_edge(3, 2, prob=0.9)
    G.add_edge(1, 3, prob=0.5)

    # ----- Instantiate your sketch-greedy solver -----
    worlds = 100        # number of live-edge samples
    budget_k = 2        # how many seeds to pick
    sketch_size = 32    # bottom-c size

    solver = sketch_greedy_hill_climbing(
        G=G,
        worlds=worlds,
        budget_k=budget_k,
        sketch_size=sketch_size,
        default_edge_prob=edge_prob
    )

    # ----- (Optional) Check influence of single-node seeds before greedy -----
    print("Estimated influence of single-node seeds (no greedy yet):")
    for v in solver.nodes:
        # temporarily clear infected to get pure spread of {v}
        saved_infected = solver.infected_nodes
        solver.infected_nodes = [set() for _ in range(solver.R)]
        inf_v = solver.estimate_influence({v})
        solver.infected_nodes = saved_infected

        print(f"  Node {v}: {inf_v:.3f} expected influenced nodes")

    # ----- Run greedy hill climbing -----
    seeds, total_inf = solver.solve()

    print("\n=== Greedy result ===")
    print(f"Chosen seed set (size {len(seeds)}): {seeds}")
    print(f"Estimated total influence of seed set: {total_inf:.3f}")

