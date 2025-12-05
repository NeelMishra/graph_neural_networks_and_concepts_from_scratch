import torch
import torch.nn as nn
import networkx as nx


class LoopyBP(nn.Module):
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

        # ----- prior (same as yours) -----
        prior = torch.full((self.n, self.k), 1.0 / self.k, device=self.device, dtype=self.dtype)

        p_main = 0.8
        p_rest = (1.0 - p_main) / (self.k - 1) if self.k > 1 else 0.0

        for label_node in self.labeled_nodes:
            i = self.node_to_idx[label_node]
            y = int(self.G.nodes[label_node]["label"])
            prior[i].fill_(p_rest)
            prior[i, y] = p_main

        prior = prior / (prior.sum(dim=1, keepdim=True).clamp_min(self.eps))
        self.prior = prior

        # ----- potentials (learnable) -----
        self.label_to_label_potential = nn.Parameter(
            torch.zeros(self.k, self.k, device=self.device, dtype=self.dtype)
        )

        # Potts init (same spirit as yours)
        beta = 3.0
        with torch.no_grad():
            self.label_to_label_potential.fill_(0.0)
            idx = torch.arange(self.k, device=self.device)
            self.label_to_label_potential[idx, idx] = torch.log(
                torch.tensor(beta, device=self.device, dtype=self.dtype)
            )

        # =====================================================================
        # CHANGE 1: store messages on directed edges only: messages shape (E, k)
        # =====================================================================
        self.dir_edges = []          # list of (src_idx, dst_idx)
        self.edge_to_eidx = {}       # map (src_idx, dst_idx) -> e_idx

        for (u, v) in self.G.edges():
            ui, vi = self.node_to_idx[u], self.node_to_idx[v]

            if (ui, vi) not in self.edge_to_eidx:
                self.edge_to_eidx[(ui, vi)] = len(self.dir_edges)
                self.dir_edges.append((ui, vi))

            if not self.G.is_directed():
                if (vi, ui) not in self.edge_to_eidx:
                    self.edge_to_eidx[(vi, ui)] = len(self.dir_edges)
                    self.dir_edges.append((vi, ui))

        self.E = len(self.dir_edges)

        # incoming edge indices for each node: list of e_idx such that (src -> node)
        self.in_edges = [[] for _ in range(self.n)]
        for e_idx, (a, b) in enumerate(self.dir_edges):
            self.in_edges[b].append(e_idx)

        # reverse edge pointer: e_idx for (b->a), needed for excluding j from N(i)\{j}
        self.rev = [None] * self.E
        for e_idx, (a, b) in enumerate(self.dir_edges):
            self.rev[e_idx] = self.edge_to_eidx.get((b, a), None)

        # uniform init messages (E,k)
        self.reset_messages()

        self.idx_to_label_vector = self.prior.clone()

    def reset_messages(self):
        self.messages = torch.full(
            (self.E, self.k),
            1.0 / self.k,
            device=self.device,
            dtype=self.dtype
        )

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

    # -------------------------------------------------------------------------
    # CHANGE 2: neighbor message product computed from (E,k) messages
    # -------------------------------------------------------------------------
    def incoming_product(self, node_idx: int) -> torch.Tensor:
        prod = torch.ones(self.k, device=self.device, dtype=self.dtype)
        for e_in in self.in_edges[node_idx]:
            prod = prod * self.messages[e_in]
        return prod

    def psi_matrix(self):
        # (optional but recommended) clamp to avoid exp overflow during training
        W = self.label_to_label_potential.clamp(-10.0, 10.0)
        return torch.exp(W)

    # -------------------------------------------------------------------------
    # CHANGE 3: update a message by edge index (no assignment into big tensor)
    # -------------------------------------------------------------------------
    def update_message_eidx(self, e_idx: int) -> torch.Tensor:
        i_idx, j_idx = self.dir_edges[e_idx]

        psi = self.psi_matrix()

        # product of incoming messages to i
        prod_in_i = self.incoming_product(i_idx)

        # exclude j by dividing out reverse message (j->i) if it exists
        r = self.rev[e_idx]
        if r is not None:
            prod_in_i = prod_in_i / self.messages[r].clamp_min(self.eps)

        b = (self.prior[i_idx] * prod_in_i).clamp_min(self.eps)   # (k,)
        m_new = (b @ psi).clamp_min(self.eps)                     # (k,)
        m_new = m_new / m_new.sum().clamp_min(self.eps)
        return m_new

    def belifs(self):
        belief_matrix = torch.zeros_like(self.prior)
        for node in self.nodes:
            i = self.node_to_idx[node]
            prod_in = self.incoming_product(i)
            b = (self.prior[i] * prod_in).clamp_min(self.eps)
            b = b / b.sum().clamp_min(self.eps)
            belief_matrix[i] = b
        return belief_matrix

    # -------------------------------------------------------------------------
    # CHANGE 4: rebuild messages via stack (keeps autograd graph intact)
    # -------------------------------------------------------------------------
    def run_bp(self, iterations):
        for _ in range(iterations):
            new_msgs = [self.update_message_eidx(e) for e in range(self.E)]
            self.messages = torch.stack(new_msgs, dim=0)

    def forward(self, iterations):
        # important for training loops: reset message state each forward
        self.reset_messages()
        self.run_bp(iterations)
        return self.belifs()

    def nll_loss(self):
        beliefs = self.belifs()

        labeled_nodes = list(self.labeled_nodes)
        labeled_idx = torch.tensor([self.node_to_idx[node] for node in labeled_nodes],
                                   dtype=torch.long, device=self.device)

        y_true = torch.tensor([int(self.G.nodes[node]["label"]) for node in labeled_nodes],
                              dtype=torch.long, device=self.device)

        p_true = beliefs[labeled_idx, y_true].clamp_min(self.eps)
        return -torch.log(p_true).mean()


### LLM Generated driver code


# ============================== DRIVER / __main__ ==============================
# ============================== JUPYTER DRIVER CELL ==============================

import os
import random
import gzip
import urllib.request
from collections import Counter, defaultdict

import torch
import networkx as nx


# ------------------------- hyperparams (edit these) -------------------------
SEED = 0
K = 7
MAX_NODES = 400          # BFS subgraph size for speed
BP_ITERS = 10

UNLABEL_RATE = 0.95      # proportion of nodes (per-class) to hide labels for
MIN_SEEDS_PER_CLASS = 2  # guarantee at least these many labeled seeds per class

TRAIN = True            # set True to learn label_to_label_potential with backprop
EPOCHS = 50
LR = 1e-2
WEIGHT_DECAY = 0.0

TRAIN_FRAC_FROM_HIDDEN = 0.10  # fraction of hidden nodes used for loss (rest used for test)
# --------------------------------------------------------------------------


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, path)


def load_email_eu_core(data_dir="data/email_eu_core"):
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

    # safer: drop unlabeled nodes
    for u in list(G.nodes()):
        if u not in labels:
            G.remove_node(u)

    return G, labels


def keep_topK_classes(G: nx.Graph, labels: dict, K: int):
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
    rng = random.Random(seed)

    true_labels = {u: int(G.nodes[u]["label"]) for u in G.nodes()}
    by_class = defaultdict(list)
    for u, y in true_labels.items():
        by_class[y].append(u)

    seed_nodes = set()
    eval_nodes = []

    for y, nodes in by_class.items():
        rng.shuffle(nodes)
        n = len(nodes)

        n_keep = int(round((1.0 - unlabel_rate) * n))
        n_keep = max(min_seeds_per_class, n_keep)
        n_keep = min(n_keep, n)

        keep = nodes[:n_keep]
        hide = nodes[n_keep:]

        seed_nodes.update(keep)
        eval_nodes.extend(hide)

    for u in eval_nodes:
        if "label" in G.nodes[u]:
            del G.nodes[u]["label"]

    return true_labels, seed_nodes, eval_nodes


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


# ----------------------------- run pipeline -----------------------------

random.seed(SEED)
torch.manual_seed(SEED)

G_raw, labels = load_email_eu_core()
G = keep_topK_classes(G_raw, labels, K=K)
G = bfs_subgraph(G, max_nodes=MAX_NODES, seed=SEED)

true_labels, seed_nodes, hidden_nodes = make_semi_supervised_by_unlabel_rate(
    G, unlabel_rate=UNLABEL_RATE, min_seeds_per_class=MIN_SEEDS_PER_CLASS, seed=SEED
)

print(f"Graph: n={G.number_of_nodes()}  m={G.number_of_edges()}  classes={K}")
print(f"Seeds(labeled): {len(seed_nodes)} | Hidden(unlabeled): {len(hidden_nodes)} | unlabel_rate={UNLABEL_RATE}")

# split hidden into train/test (only used if TRAIN=True)
rng = random.Random(SEED)
hidden_shuf = hidden_nodes[:]
rng.shuffle(hidden_shuf)

n_train = max(K * 2, int(round(TRAIN_FRAC_FROM_HIDDEN * len(hidden_shuf))))
train_nodes = hidden_shuf[:n_train]
test_nodes = hidden_shuf[n_train:]

print(f"Train(hidden labels used for loss): {len(train_nodes)} | Test(hidden eval): {len(test_nodes)}")

model = LoopyBP(G, classes=K)
model.train(TRAIN)

def loss_on_nodes(beliefs, nodes):
    idx = torch.tensor([model.node_to_idx[u] for u in nodes], dtype=torch.long)
    y = torch.tensor([true_labels[u] for u in nodes], dtype=torch.long)
    p = beliefs[idx, y].clamp_min(model.eps)
    return -torch.log(p).mean()

# ----------------------------- optional training -----------------------------
if TRAIN:
    optim = torch.optim.Adam([model.label_to_label_potential], lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        optim.zero_grad()
        beliefs = model(BP_ITERS)
        loss = loss_on_nodes(beliefs, train_nodes)
        loss.backward()
        optim.step()

        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                b = model(BP_ITERS)
                def acc_on(nodes):
                    ii = torch.tensor([model.node_to_idx[u] for u in nodes], dtype=torch.long)
                    yy = torch.tensor([true_labels[u] for u in nodes], dtype=torch.long)
                    pred = b[ii].argmax(dim=1)
                    return (pred == yy).double().mean().item()
                print(f"epoch {epoch:03d} | loss={loss.item():.4f} | train_acc={acc_on(train_nodes):.4f} | test_acc={acc_on(test_nodes):.4f}")

# ----------------------------- final evaluation -----------------------------
model.eval()
with torch.no_grad():
    beliefs = model(BP_ITERS)

test_idx = torch.tensor([model.node_to_idx[u] for u in test_nodes], dtype=torch.long)
y_true = [true_labels[u] for u in test_nodes]
y_pred = beliefs[test_idx].argmax(dim=1).cpu().tolist()

acc, macro_p, macro_r, micro_p, micro_r, per_class = metrics_multiclass(y_true, y_pred, K)
pred_hist = Counter(y_pred)

print("\n=== Metrics on TEST hidden-label nodes ===")
print(f"Accuracy:        {acc:.4f}  (N={len(y_true)})")
print(f"Macro Precision: {macro_p:.4f}")
print(f"Macro Recall:    {macro_r:.4f}")
print(f"Micro Precision: {micro_p:.4f}")
print(f"Micro Recall:    {micro_r:.4f}")

print("\nPredicted class histogram:")
for c in range(K):
    print(f"  class {c}: {pred_hist.get(c, 0)}")

print("\nPer-class (precision, recall, tp, fp, fn):")
for c, (p, r, tp, fp, fn) in enumerate(per_class):
    print(f"  class {c:2d}: P={p:.4f}  R={r:.4f}   tp={tp:4d} fp={fp:4d} fn={fn:4d}")
