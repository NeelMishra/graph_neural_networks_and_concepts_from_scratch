'''
This is a part of influence maximization algorithms, stanford graph neural network 2019 : lecture 14
'''
import numpy as np
import networkx as nx
from independent_cascade import IndependentCascade

class GHC:

    def __init__(self, G: nx.DiGraph, trials: int, budget_k: int):
        self.G = G
        self.trials = trials
        self.K = budget_k

        self.nodes = list(self.G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        self.idx_to_label = {idx: 0 for idx in self.idx_to_node}

        self.seed_set = set()
        self.IC = IndependentCascade(self.G)

    def perform_mc_sampling(self, seed_set):

        total_fetched = 0
        for trial in range(self.trials):
            total_influenced = self.IC.bfs_traversal(list(seed_set))
            total_fetched += len(total_influenced)

        return total_fetched/self.trials
    
    def iterate(self):
        best_fit = -float('inf')
        best_candidate = None

        for node in self.nodes:
            if node in self.seed_set:
                continue
            avg_influence_of_node = self.perform_mc_sampling(self.seed_set.union({node}))
            if avg_influence_of_node > best_fit:
                best_candidate = node
                best_fit = avg_influence_of_node

        return best_candidate, best_fit
    
    def fit(self):
        
        for i in range(self.K):
            best_candidate, best_fit = self.iterate()
            self.seed_set.add(best_candidate)
        return
    


### LLM Generated driver code
import os, gzip, urllib.request, time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from independent_cascade import IndependentCascade


# -------------------------
# 1) Download + load dataset
# -------------------------
def _download(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, path)

def load_email_eu_core(data_dir="data/email_eu_core"):
    edges_gz = os.path.join(data_dir, "email-Eu-core.txt.gz")
    _download("https://snap.stanford.edu/data/email-Eu-core.txt.gz", edges_gz)

    G = nx.DiGraph()
    with gzip.open(edges_gz, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            u, v = s.split()
            G.add_edge(int(u), int(v))

    # largest weakly connected component (so diffusion isn't fragmented)
    if G.number_of_nodes() > 0:
        wcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(wcc).copy()

    return G


# -------------------------
# 2) Candidate pool helpers
# -------------------------
def top_outdeg_nodes(G: nx.DiGraph, m: int):
    return [n for n, _ in sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:m]]

def sprinkle_random_nodes(G: nx.DiGraph, m: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    m = min(m, len(nodes))
    return list(rng.choice(nodes, size=m, replace=False))

def make_candidate_pool(G: nx.DiGraph, top_m=200, random_m=0, seed=0):
    pool = []
    pool.extend(top_outdeg_nodes(G, top_m))
    if random_m > 0:
        pool.extend(sprinkle_random_nodes(G, random_m, seed=seed))
    # de-dup, keep order
    seen = set()
    out = []
    for x in pool:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# -------------------------
# 3) Monte Carlo spread estimate
#    (common random numbers per round)
# -------------------------
def mc_expected_spread(ic: IndependentCascade, seed_set, trials: int, base_seed: int):
    """
    Returns mean spread E[|infected|] over 'trials' runs.
    Uses deterministic seeding: np.random.seed(base_seed + t)
    """
    seed_list = list(seed_set)
    total = 0
    for t in range(trials):
        np.random.seed(base_seed + t)
        infected = ic.bfs_traversal(seed_list)
        total += len(infected)
    return total / trials


# -------------------------
# 4) Greedy selection with progress
# -------------------------
def pick_next_seed_with_progress(ic, current_seeds, candidates, trials, round_idx,
                                 log_every=25, base_seed_round=1_000_000):
    """
    One greedy step:
    argmax_v f(S ∪ {v}) estimated by MC.
    Progress prints every log_every candidates checked.
    """
    best_node = None
    best_score = -float("inf")

    # common randomness within this round:
    # every candidate uses the same (base_seed + t) sequence
    base_seed = base_seed_round + round_idx * 10_000

    start = time.perf_counter()
    checked = 0
    total = sum(1 for v in candidates if v not in current_seeds)

    for v in candidates:
        if v in current_seeds:
            continue
        checked += 1

        score = mc_expected_spread(ic, current_seeds | {v}, trials=trials, base_seed=base_seed)

        if score > best_score:
            best_score = score
            best_node = v

        if (checked % log_every == 0) or (checked == total):
            elapsed = time.perf_counter() - start
            print(f"  [round {round_idx+1}] checked {checked}/{total} | best={best_node} score={best_score:.2f} | elapsed={elapsed:.1f}s")

    return best_node, best_score


def greedy_hill_climb_ic(G, edge_p=0.05, trials_select=5, K=8,
                         top_m=200, random_m=0, candidate_seed=0,
                         log_every=25):
    """
    Returns ordered list of selected seeds.
    """
    # IC will default missing probs to edge_p internally
    ic = IndependentCascade(G, default_edge_prob=edge_p)

    candidates = make_candidate_pool(G, top_m=top_m, random_m=random_m, seed=candidate_seed)
    print(f"Candidate pool size = {len(candidates)} (top_m={top_m}, random_m={random_m})")

    seeds = set()
    seed_list = []

    for r in range(min(K, len(candidates))):
        v, score = pick_next_seed_with_progress(
            ic=ic,
            current_seeds=seeds,
            candidates=candidates,
            trials=trials_select,
            round_idx=r,
            log_every=log_every
        )
        if v is None:
            print("No candidate found (pool exhausted). Stopping.")
            break

        seeds.add(v)
        seed_list.append(v)
        print(f"[GHC] picked {v} (est spread={score:.2f}) | seeds so far: {seed_list}")

    return seed_list


# -------------------------
# 5) Baselines + evaluation plots
# -------------------------
def random_seed_set(G, k, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    k = min(k, len(nodes))
    return list(rng.choice(nodes, size=k, replace=False))

def eval_spread_curve(G, greedy_seeds_ordered, edge_p=0.05, trials_eval=30):
    ic = IndependentCascade(G, default_edge_prob=edge_p)

    Ks = list(range(1, len(greedy_seeds_ordered) + 1))
    ghc_means = []
    deg_means = []
    rnd_means = []

    # fixed random baseline seeds (same for all K slices)
    rnd_all = random_seed_set(G, len(greedy_seeds_ordered), seed=123)

    deg_all = top_outdeg_nodes(G, len(greedy_seeds_ordered))

    for i, k in enumerate(Ks):
        base_seed = 50_000 + i * 1_000  # stable across K

        s_ghc = set(greedy_seeds_ordered[:k])
        s_deg = set(deg_all[:k])
        s_rnd = set(rnd_all[:k])

        ghc_means.append(mc_expected_spread(ic, s_ghc, trials=trials_eval, base_seed=base_seed))
        deg_means.append(mc_expected_spread(ic, s_deg, trials=trials_eval, base_seed=base_seed))
        rnd_means.append(mc_expected_spread(ic, s_rnd, trials=trials_eval, base_seed=base_seed))

        print(f"[eval] K={k}: GHC={ghc_means[-1]:.2f} | DEG={deg_means[-1]:.2f} | RND={rnd_means[-1]:.2f}")

    plt.figure()
    plt.plot(Ks, ghc_means, marker="o", linewidth=2, label="GHC (greedy+MC, your IC)")
    plt.plot(Ks, deg_means, marker="o", linewidth=2, label="Top out-degree")
    plt.plot(Ks, rnd_means, marker="o", linewidth=2, label="Random")
    plt.xlabel("Seed budget K")
    plt.ylabel("Expected spread  E[|infected|]")
    plt.title("Influence Maximization on email-Eu-core (IC)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_one_run_growth(G, seeds, edge_p=0.05, run_seed=7):
    ic = IndependentCascade(G, default_edge_prob=edge_p)
    np.random.seed(run_seed)
    infected = ic.bfs_traversal(list(seeds))

    # build time series from ic.idx_to_infected_time
    node_time = {ic.idx_to_node[i]: ic.idx_to_infected_time[i] for i in ic.idx_to_infected_time}
    times = [t for t in node_time.values() if t >= 0]
    if not times:
        print("No infections happened.")
        return

    T = max(times)
    new = np.zeros(T + 1, dtype=int)
    for t in times:
        new[t] += 1
    cum = np.cumsum(new)

    plt.figure()
    plt.plot(np.arange(T + 1), cum, marker="o", linewidth=2)
    plt.xlabel("time step")
    plt.ylabel("cumulative infected")
    plt.title(f"One cascade growth curve | seeds={list(seeds)} | p={edge_p}")
    plt.grid(True)
    plt.show()


# -------------------------
# 6) Main
# -------------------------
def main():
    G = load_email_eu_core()
    print(f"Loaded email-Eu-core: n={G.number_of_nodes()}  m={G.number_of_edges()}")

    # knobs (tune for speed)
    EDGE_P = 0.05
    K = 8

    # IMPORTANT speed knobs:
    TRIALS_SELECT = 5      # used inside greedy (small!)
    TOP_M = 200            # smaller candidate pool
    RANDOM_M = 30          # optional exploration
    LOG_EVERY = 25

    # evaluation knobs
    TRIALS_EVAL = 25       # used only for plotting curves

    print("\n=== Running Greedy Hill Climbing (small candidate pool) ===")
    greedy_seeds = greedy_hill_climb_ic(
        G,
        edge_p=EDGE_P,
        trials_select=TRIALS_SELECT,
        K=K,
        top_m=TOP_M,
        random_m=RANDOM_M,
        candidate_seed=0,
        log_every=LOG_EVERY
    )

    print("\nSelected seeds (ordered):", greedy_seeds)

    print("\n=== Effectiveness: spread vs K (GHC vs Degree vs Random) ===")
    eval_spread_curve(G, greedy_seeds, edge_p=EDGE_P, trials_eval=TRIALS_EVAL)

    print("\n=== One story run: how the cascade grows over time (your seeds) ===")
    plot_one_run_growth(G, seeds=set(greedy_seeds), edge_p=EDGE_P, run_seed=7)


if __name__ == "__main__":
    main()
