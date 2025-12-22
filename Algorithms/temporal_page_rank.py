import networkx as nx
import numpy as np
from bisect import bisect_left, bisect_right


class TemporalPageRank:

    ''' This implementation is for directed graph only, check using self.G.is_directed()'''
    def __init__(self, G: nx.DiGraph, alpha:float=0.9,  beta: float = 0.9):
        
        if not G.is_directed():
            raise NotImplementedError("The current implementation is for directed graphs only")

        self.G = G
        self.alpha = alpha # damping
        self.beta = beta # temporal decay

        all_times = set()

        for u, v, data in self.G.edges(data=True):
            times_list = data.get("times", None)
            if times_list is None:
                raise ValueError(f"Edge ({u}->{v}) is missing 'times' attribute")
            if len(times_list) == 0:
                raise ValueError(f"Edge ({u}->{v}) has empty 'times' list")

            for t in times_list:
                all_times.add(t)

        self.times = sorted(all_times)
        self.T = len(self.times)

        if self.T == 0:
            raise ValueError("No timesteps found in the graph")

        self.nodes = sorted(list(self.G.nodes()))
        self.num_of_nodes = len(self.nodes)

        self.idx_to_node = {idx : node for idx, node in enumerate(self.nodes)}
        self.node_to_idx = {node : idx for idx, node in enumerate(self.nodes)}
        self.time_to_idx = {t: i for i, t in enumerate(self.times)}
        self.idx_to_time = {i: t for i, t in enumerate(self.times)}

        self.num_states = self.num_of_nodes * self.T
        self.page_rank = np.ones((self.num_states, 1), dtype=np.float64)/self.num_states # N x 1, N => Number of nodes

        self.out_degree = {node : self.G.out_degree(node) for node in self.nodes}
        
        self.out_events = {u_idx: [] for u_idx in range(self.num_of_nodes)}

        for u, v, data in self.G.edges(data=True):
            time_list = data["times"] 
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]

            for t in time_list:
                t_idx = self.time_to_idx[t]
                self.out_events[u_idx].append((t_idx, v_idx))

        self.out_times = {}
        for u_idx in range(self.num_of_nodes):
            self.out_events[u_idx].sort(key=lambda x:x[0]) # sort by time (index 0)
            self.out_times[u_idx] = [t_idx for (t_idx, _) in self.out_events[u_idx]]

    def state_id(self, u_idx, t_idx):
        # Function to facilitate storing the page rank into 1d vector
        return u_idx * self.T + t_idx

    def gamma_count(self, u_idx: int, t1_idx: int, t2_idx: int) -> int:
        if t2_idx < t1_idx:
            return 0
        
        out_times_u = self.out_times[u_idx]

        left = bisect_left(out_times_u, t1_idx)
        right = bisect_right(out_times_u, t2_idx)

        return right - left
    
    def temporal_out(self, u_idx, t1_idx):
        out_times_u = self.out_times[u_idx]
        out_events_u = self.out_events[u_idx]

        if not out_events_u:
            return []

        left = bisect_left(out_times_u, t1_idx)

        transitions = []
        total_w = 0.0

        for t2_idx, v_idx in out_events_u[left:]:
            w = self.beta ** self.gamma_count(u_idx, t1_idx, t2_idx)
            transitions.append((v_idx, t2_idx, w))
            total_w += w

        return [ (v_idx, t2_idx, w/total_w) for v_idx, t2_idx, w in transitions ]


    def update_r(self):
        r_old = self.page_rank
        r_new = np.zeros((self.num_states, 1), dtype=np.float64)

        dangling_mass = 0.0
        for u_idx in range(self.num_of_nodes):
            for t1_idx in range(self.T):
                s = self.state_id(u_idx, t1_idx)
                outs = self.temporal_out(u_idx, t1_idx)

                if not outs:
                    dangling_mass += float(r_old[s])
                    continue

                for v_idx, t2_idx, p in outs:
                    dst = self.state_id(v_idx, t2_idx)
                    r_new[dst] += self.alpha * r_old[s] * p

        
        # Teleportation exists in real life

        r_new += (1 - self.alpha + self.alpha * dangling_mass) / self.num_states

        self.page_rank = r_new 
        return np.linalg.norm(r_new - r_old)
    
    def node_scores(self):
        r = self.page_rank.reshape(self.num_of_nodes, self.T)  # shape (N, T)
        scores = r.sum(axis=1)  # shape (N,)
        return {self.idx_to_node[u_idx]: float(scores[u_idx]) for u_idx in range(self.num_of_nodes)}


    def fit(self, epochs, tol=1e-8):

        for epoch in range(epochs):
            diff = self.update_r()
            if diff < tol:
                break
        return self.node_scores()



### LLM Generated driver code


# driver_temporal_pagerank.py
# Assumes your TemporalPageRank class is defined ABOVE this block in the same file.

import argparse
import numpy as np
import networkx as nx


def make_directed_karate_with_times(T: int = 20, max_events_per_edge: int = 3, seed: int = 0) -> nx.DiGraph:
    """
    Build a directed version of Zachary's Karate Club graph and attach a 'times' list
    to every directed edge, as required by your TemporalPageRank implementation.
    """
    rng = np.random.default_rng(seed)
    base = nx.karate_club_graph()  # undirected, built-in dataset

    G = nx.DiGraph()
    G.add_nodes_from(base.nodes())

    for u, v in base.edges():
        # add both directions
        for a, b in ((u, v), (v, u)):
            k = int(rng.integers(1, max_events_per_edge + 1))  # at least 1 event
            # choose k distinct timesteps in [0, T-1]
            times = rng.choice(np.arange(T), size=min(k, T), replace=False)
            times = sorted(int(x) for x in times)
            G.add_edge(a, b, times=times)

    return G


def topk(d: dict, k: int = 10):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TemporalPageRank on a NetworkX dataset (Karate Club).")
    parser.add_argument("--alpha", type=float, default=0.85, help="PageRank damping (follow links prob).")
    parser.add_argument("--beta", type=float, default=0.9, help="Temporal decay (lecture beta).")
    parser.add_argument("--T", type=int, default=20, help="Number of global timesteps to sample.")
    parser.add_argument("--max-events-per-edge", type=int, default=3, help="Max events per directed edge.")
    parser.add_argument("--epochs", type=int, default=200, help="Max power-iterations.")
    parser.add_argument("--tol", type=float, default=1e-10, help="Convergence tolerance.")
    parser.add_argument("--topk", type=int, default=10, help="How many top nodes to print.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()

    # Build dataset graph with required 'times' lists
    G = make_directed_karate_with_times(T=args.T, max_events_per_edge=args.max_events_per_edge, seed=args.seed)

    # Run your TemporalPageRank
    tpr = TemporalPageRank(G, alpha=args.alpha, beta=args.beta)
    scores = tpr.fit(epochs=args.epochs, tol=args.tol)

    print("\nTemporalPageRank (node scores, summed over time):")
    for node, score in topk(scores, args.topk):
        print(f"  node {node:>2}: {score:.6f}")
    print(f"sum(scores) = {sum(scores.values()):.12f}")

    # Optional sanity: compare against vanilla PageRank on the same directed graph (ignoring time)
    pr_static = nx.pagerank(G, alpha=args.alpha)
    print("\nVanilla PageRank on same directed graph (ignoring time):")
    for node, score in topk(pr_static, args.topk):
        print(f"  node {node:>2}: {score:.6f}")
    print(f"sum(static) = {sum(pr_static.values()):.12f}")
