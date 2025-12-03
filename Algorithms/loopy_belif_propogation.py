import torch
import torch.nn as nn
import networkx as nx


class LoopyBP(nn.Module):
    '''Loopy belif propogation algorithm : Current implementation is backprop unsafe'''
    def __init__(self, G: nx.Graph, classes: int, eps=1e-12):
        super().__init__()
        self.G = G
        self.k = int(classes)
        self.eps = float(eps)

        self.device = torch.device('cpu')
        self.dtype = torch.float64

        self.nodes = list(self.G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}

        self.get_labelled_and_unlabelled_nodes()

        prior = torch.full((self.n, self.k), 1.0 / self.k, device=self.device, dtype=self.dtype)

        p_main = 0.8
        p_rest = (1.0 - p_main) / (self.k - 1) if self.k > 1 else 0.0

        for label_node in self.labeled_nodes:
            i = self.node_to_idx[label_node]
            y = int(self.G.nodes[label_node]["label"])

            prior[i].fill_(p_rest)      # set all labels
            prior[i, y] = p_main        #overwrite the true label to have largest probability
        
        prior = prior / (prior.sum(dim=1, keepdim=True).clamp_min(self.eps))
        self.prior = prior  

        self.label_to_label_potential = nn.Parameter(
            torch.zeros(self.k, self.k, device=self.device, dtype=self.dtype)
        )

        # This is Potts model, we can also use a learnable parameter, but that is out of scope for this algorithm.
        beta = 3.0 
        with torch.no_grad():
            self.label_to_label_potential.fill_(0.0)
            idx = torch.arange(self.k, device=self.device)
            self.label_to_label_potential[idx, idx] = torch.log(torch.tensor(beta, device=self.device, dtype=self.dtype))


        messages = torch.full((self.n, self.n, self.k), 1.0 / self.k, device=self.device, dtype=self.dtype)
        self.messages = messages


        self.idx_to_label_vector = self.prior.clone()

    def get_labelled_and_unlabelled_nodes(self):
        labeled_nodes = set()
        unlabeled_nodes = set()

        for node in self.G.nodes():
            label = self.G.nodes[node].get("label", None)
            if label is not None:
                labeled_nodes.add(node)
            else:
                unlabeled_nodes.add(node)

        self.labeled_nodes = labeled_nodes
        self.un_labeled_nodes = unlabeled_nodes

    def get_neighbour_messages(self, node, exclude_node):
        node_idx = self.node_to_idx[node]
        neighbors = [nbr for nbr in self.G.neighbors(node) if nbr != exclude_node]

        if len(neighbors) == 0:
            return torch.ones(self.k, device=self.device, dtype=self.dtype)

        prod = torch.ones(self.k, device=self.device, dtype=self.dtype)
        for nbr in neighbors:
            nbr_idx = self.node_to_idx[nbr]
            prod = prod * self.messages[nbr_idx, node_idx]  # (k,)

        return prod
    
    def psi_matrix(self):
        return torch.exp(self.label_to_label_potential)
    
    def update_message(self, i, j):
        i_idx = self.node_to_idx[i]

        psi_matrix = self.psi_matrix()
        neighbour_messages = self.get_neighbour_messages(i, j)

        b = (self.prior[i_idx]  * neighbour_messages).clamp_min(self.eps) # (k,) * (k, ) => Element wise multiplication
        m_new = (b @ psi_matrix)
        m_new = m_new / m_new.sum().clamp_min(self.eps)

        return m_new

    def belifs(self):

        belif_matrix = torch.zeros_like(self.prior)

        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            neighbour_messages = self.get_neighbour_messages(node, None)

            b = (self.prior[node_idx]  * neighbour_messages).clamp_min(self.eps)
            b = b / b.sum().clamp_min(self.eps)

            belif_matrix[node_idx] = b

        return belif_matrix

    def run_bp(self, iterations):

        for iter in range(iterations):
            updated_messages = self.messages.clone()

            for edge in self.G.edges():
                i, j = edge    
                i_idx, j_idx = self.node_to_idx[i], self.node_to_idx[j]
                updated_messages[i_idx, j_idx] = self.update_message(i, j)
                if not self.G.is_directed():
                    updated_messages[j_idx, i_idx] = self.update_message(j, i)

            self.messages = updated_messages
    
    def forward(self, iterations):
        self.run_bp(iterations)
        return self.belifs()

    def nll_loss(self):
        belifs = self.belifs()

        labeled_nodes = list(self.labeled_nodes)
        labeled_idx = torch.tensor([self.node_to_idx[node] for node in labeled_nodes], dtype=torch.long)

        y_true = torch.tensor([int(self.G.nodes[node]["label"]) for node in labeled_nodes], dtype=torch.long)

        p_true = belifs[labeled_idx, y_true].clamp_min(self.eps)

        return - torch.log(p_true).mean()
    

#### LLM Generated driver code



# ============================== DRIVER / __main__ ==============================
import os
import random
import gzip
import urllib.request
import argparse
from collections import Counter, defaultdict

import torch
import networkx as nx


# ------------------------- data utils (SNAP email-Eu-core) -------------------------

def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, path)


def load_email_eu_core(data_dir="data/email_eu_core"):
    """
    SNAP:
      - email-Eu-core.txt.gz (edges)
      - email-Eu-core-department-labels.txt.gz (node->department label)
    Returns:
      G: nx.Graph
      labels: dict[int,int]
    """
    edges_gz = os.path.join(data_dir, "email-Eu-core.txt.gz")
    labels_gz = os.path.join(data_dir, "email-Eu-core-department-labels.txt.gz")

    _download("https://snap.stanford.edu/data/email-Eu-core.txt.gz", edges_gz)
    _download("https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz", labels_gz)

    G = nx.Graph()
    with gzip.open(edges_gz, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = line.split()
            G.add_edge(int(u), int(v))

    labels = {}
    with gzip.open(labels_gz, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            node, lab = line.split()
            labels[int(node)] = int(lab)

    # Drop nodes without labels (rare, but safer)
    for u in list(G.nodes()):
        if u not in labels:
            G.remove_node(u)

    return G, labels


def keep_topK_classes(G: nx.Graph, labels: dict, K: int):
    """
    Keep nodes in top-K most frequent department labels, remap labels to 0..K-1.
    """
    # attach labels
    for u in G.nodes():
        G.nodes[u]["label"] = int(labels[u])

    counts = Counter(int(G.nodes[u]["label"]) for u in G.nodes())
    top = [lab for lab, _ in counts.most_common(K)]
    top_set = set(top)

    keep_nodes = [u for u in G.nodes() if int(G.nodes[u]["label"]) in top_set]
    H = G.subgraph(keep_nodes).copy()

    remap = {old: new for new, old in enumerate(top)}
    for u in H.nodes():
        H.nodes[u]["label"] = remap[int(H.nodes[u]["label"])]

    return H


def bfs_subgraph(G: nx.Graph, max_nodes: int, seed: int):
    """
    Induced BFS subgraph to keep runtime manageable.
    """
    if max_nodes is None or G.number_of_nodes() <= max_nodes:
        return G

    rng = random.Random(seed)
    start = rng.choice(list(G.nodes()))
    visited = {start}
    q = [start]

    while q and len(visited) < max_nodes:
        u = q.pop(0)
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                q.append(v)
            if len(visited) >= max_nodes:
                break

    return G.subgraph(list(visited)).copy()


def make_semi_supervised_by_unlabel_rate(G: nx.Graph, unlabel_rate: float, min_seeds_per_class: int, seed: int):
    """
    unlabel_rate:
      fraction of nodes (per class) to hide labels for (eval set).
    Keeps at least min_seeds_per_class labeled nodes per class.
    Returns:
      true_labels: dict[node->y]
      seed_nodes: set[node]  (labels kept in graph)
      eval_nodes: list[node] (labels removed from graph)
    """
    rng = random.Random(seed)

    # true labels snapshot before we delete anything
    true_labels = {u: int(G.nodes[u]["label"]) for u in G.nodes()}

    by_class = defaultdict(list)
    for u, y in true_labels.items():
        by_class[y].append(u)

    seed_nodes = set()
    eval_nodes = []

    for y, nodes in by_class.items():
        rng.shuffle(nodes)
        n = len(nodes)

        # how many to keep labeled
        n_keep = int(round((1.0 - unlabel_rate) * n))
        n_keep = max(min_seeds_per_class, n_keep)
        n_keep = min(n_keep, n)  # can't exceed

        keep = nodes[:n_keep]
        hide = nodes[n_keep:]

        seed_nodes.update(keep)
        eval_nodes.extend(hide)

    # hide labels for eval nodes
    for u in eval_nodes:
        if "label" in G.nodes[u]:
            del G.nodes[u]["label"]

    return true_labels, seed_nodes, eval_nodes


# ------------------------- metrics -------------------------

def metrics_multiclass(y_true, y_pred, K: int):
    n = len(y_true)
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / n if n else 0.0

    per_class = []
    tp_total = fp_total = fn_total = 0

    for c in range(K):
        tp = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))

        tp_total += tp
        fp_total += fp
        fn_total += fn

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_class.append((prec, rec, tp, fp, fn))

    macro_p = sum(p for p, _, *_ in per_class) / K if K else 0.0
    macro_r = sum(r for _, r, *_ in per_class) / K if K else 0.0

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0

    return acc, macro_p, macro_r, micro_p, micro_r, per_class


# ------------------------- main -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=7, help="Number of classes (top-K departments).")
    parser.add_argument("--max_nodes", type=int, default=400, help="BFS subgraph size for speed.")
    parser.add_argument("--bp_iters", type=int, default=10, help="Loopy BP iterations.")
    parser.add_argument("--unlabel_rate", type=float, default=0.050,
                        help="Proportion of nodes (per class) to REMOVE labels from (0..1).")
    parser.add_argument("--min_seeds_per_class", type=int, default=2,
                        help="Minimum labeled seeds per class kept in graph.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    if not (0.0 <= args.unlabel_rate <= 1.0):
        raise ValueError("--unlabel_rate must be in [0,1].")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Load real dataset
    G_raw, labels = load_email_eu_core()

    # 2) Keep top-K classes and remap labels to 0..K-1
    G = keep_topK_classes(G_raw, labels, K=args.K)

    # 3) Smaller subgraph
    G = bfs_subgraph(G, max_nodes=args.max_nodes, seed=args.seed)

    # 4) Semi-supervised: hide labels for a proportion of nodes
    true_labels, seed_nodes, eval_nodes = make_semi_supervised_by_unlabel_rate(
        G, unlabel_rate=args.unlabel_rate, min_seeds_per_class=args.min_seeds_per_class, seed=args.seed
    )

    print(f"Graph: n={G.number_of_nodes()}  m={G.number_of_edges()}  classes={args.K}")
    print(f"unlabel_rate={args.unlabel_rate:.2f} | Seeds(labeled)={len(seed_nodes)} | Eval(hidden)={len(eval_nodes)}")

    # 5) Run BP
    model = LoopyBP(G, classes=args.K)
    model.eval()
    with torch.no_grad():
        beliefs = model(args.bp_iters)  # (n,k)

    # 6) Evaluate on hidden-label nodes
    eval_idx = torch.tensor([model.node_to_idx[u] for u in eval_nodes], dtype=torch.long)
    y_true = [true_labels[u] for u in eval_nodes]
    y_pred = beliefs[eval_idx].argmax(dim=1).cpu().tolist()

    acc, macro_p, macro_r, micro_p, micro_r, per_class = metrics_multiclass(y_true, y_pred, args.K)

    # Prediction histogram (helps detect "collapse to one class")
    pred_hist = Counter(y_pred)

    print("\n=== Metrics on hidden-label nodes ===")
    print(f"Accuracy:        {acc:.4f}  (N={len(y_true)})")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Micro Precision: {micro_p:.4f}")
    print(f"Micro Recall:    {micro_r:.4f}")

    print("\nPredicted class histogram:")
    for c in range(args.K):
        print(f"  class {c}: {pred_hist.get(c, 0)}")

    print("\nPer-class (precision, recall, tp, fp, fn):")
    for c, (p, r, tp, fp, fn) in enumerate(per_class):
        print(f"  class {c:2d}: P={p:.4f}  R={r:.4f}   tp={tp:4d} fp={fp:4d} fn={fn:4d}")
