import numpy as np
import networkx as nx

class IterativeClassifer:

    def __init__(self, G, node_to_flat_vector_func, node_emb_dim, meta_emb_dim, classifier_func, classes):

        # node_to_flat_meta_vector_func => vector of meta information, example I_a, I_b, O_a, O_b ex : (atleast one of the incoming pages is a, b), (atleast one of the outgoing pages is a,b)
        
        self.G = G
        self.node_to_flat_vector_func = node_to_flat_vector_func
        self.classifier_func = classifier_func
        self.k = classes

        self.node_emb_dim = node_emb_dim
        self.meta_emb_dim = meta_emb_dim

        self.nodes = list(self.G.nodes())
        self.node_to_idx = {node : idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx : node for idx, node in enumerate(self.nodes)}
        
        self.idx_to_label_vector = {}
        self.idx_to_node_vector = {}
        self.idx_to_neighbour_vector = {}
        self.idx_to_complete_vector = {}

        self.set_node_vectors()
        self.get_labeled_nodes()
        self.get_unlabelled_nodes()

        return
    
    def node_to_flat_meta_vector_func(self, node):
        '''
        We can set this function based on any metadata of the class labels, degrees, egonet? etc basically the idea being that after each updates the labels will change so these features will change
        '''
        incoming_edges_per_class = np.zeros(self.k, )
        outgoing_edges_per_class = np.zeros(self.k, )

        if self.G.is_directed():
            incoming_nodes = self.G.predecessors(node)
            outgoing_nodes = self.G.successors(node)
        else:
            incoming_nodes = self.G.neighbors(node)
            outgoing_nodes = self.G.neighbors(node)
    
        for incoming_node in incoming_nodes:
            incoming_node_idx = self.node_to_idx[incoming_node]
            label = self.idx_to_label_vector[incoming_node_idx]
            incoming_edges_per_class += label

        for outgoing_node in outgoing_nodes:
            outgoing_node_idx = self.node_to_idx[outgoing_node]
            label = self.idx_to_label_vector[outgoing_node_idx]
            outgoing_edges_per_class += label

        return np.concatenate([incoming_edges_per_class, outgoing_edges_per_class])

    def set_node_vectors(self):

        for node in self.nodes:
            idx = self.node_to_idx[node]
            self.idx_to_node_vector[idx] = self.node_to_flat_vector_func(node)
        
        return

    def concatenate_vectors(self):

        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            self.idx_to_complete_vector[node_idx] = np.concatenate([self.idx_to_node_vector[node_idx], self.idx_to_neighbour_vector[node_idx]]) 
        
        return

    def get_labeled_nodes(self):
        labeled_nodes = set()
 
        for node in self.G.nodes():
            idx = self.node_to_idx[node]
            label = self.G.nodes[node].get("label", None)
            if label is not None:
                labeled_nodes.add(node)
                self.idx_to_label_vector[idx] = np.zeros(self.k, )
                self.idx_to_label_vector[idx][int(label)] = 1
        
        self.labeled_nodes = labeled_nodes

    def get_unlabelled_nodes(self):
        unlabeled_nodes = set()

        for node in self.G.nodes():
            idx = self.node_to_idx[node]
            label = self.G.nodes[node].get("label", None)
            if label is None:
                unlabeled_nodes.add(node)
                self.idx_to_label_vector[idx] = np.random.uniform(low = 0, high=1, size=(self.k))
                self.idx_to_label_vector[idx] /= np.sum(self.idx_to_label_vector[idx])
        
        self.un_labeled_nodes = unlabeled_nodes
        return

    def calculate_all_node_vectors(self):
        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            self.idx_to_node_vector[node_idx] = self.node_to_flat_vector_func(node)
        return

    def aggregate_neighbour_information(self):

        for node in self.nodes:
            neighbour_info = []
            node_idx = self.node_to_idx[node]

            for neighbour in self.G[node]:
                neighbour_info.append(self.node_to_flat_meta_vector_func(neighbour))

            if neighbour_info:
                self.idx_to_neighbour_vector[node_idx] = np.mean(neighbour_info, axis=0)
            else:
                self.idx_to_neighbour_vector[node_idx] = np.zeros(self.meta_emb_dim, )
        return
    
    def update_label_vector(self, nodes):
        # self.idx_to_label_vector 

        for node in nodes:
            node_idx = self.node_to_idx[node]
            classification = self.classifier_func(self.idx_to_complete_vector[node_idx])

            self.idx_to_label_vector[node_idx] = np.zeros(self.k,)
            self.idx_to_label_vector[node_idx][classification] = 1.0
    
    def forward(self):

        self.calculate_all_node_vectors()

        self.aggregate_neighbour_information()

        self.concatenate_vectors()
        
        self.update_label_vector(self.un_labeled_nodes)


##### LLM Generated driver code


import os
import gzip
import argparse
import urllib.request
import numpy as np
import networkx as nx




# -------------------- SNAP download + load --------------------

def _download(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return
    urllib.request.urlretrieve(url, out_path)

def _gunzip_to(gz_path: str, out_path: str) -> None:
    if os.path.exists(out_path):
        return
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        f_out.write(f_in.read())

def load_email_eu_core(data_dir="data/email_eu_core"):
    """
    Real-world directed email network + department labels from SNAP.
    Returns:
      G: nx.DiGraph
      dept_label: dict[node] -> original department id (int)
    """
    base = "https://snap.stanford.edu/data/"
    edges_gz = os.path.join(data_dir, "email-Eu-core.txt.gz")
    labs_gz  = os.path.join(data_dir, "email-Eu-core-department-labels.txt.gz")
    edges_txt = edges_gz[:-3]
    labs_txt  = labs_gz[:-3]

    _download(base + "email-Eu-core.txt.gz", edges_gz)
    _download(base + "email-Eu-core-department-labels.txt.gz", labs_gz)
    _gunzip_to(edges_gz, edges_txt)
    _gunzip_to(labs_gz, labs_txt)

    dept_label = {}
    with open(labs_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n, dept = line.split()
            dept_label[int(n)] = int(dept)

    G = nx.DiGraph()
    G.add_nodes_from(dept_label.keys())
    with open(edges_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = line.split()
            G.add_edge(int(u), int(v))

    return G, dept_label


# -------------------- label processing (top-K + other) --------------------

def make_topk_multiclass(dept_label, top_k=10):
    """
    Convert original department ids into:
      - top_k most frequent departments (as distinct classes)
      - all remaining departments collapsed into class 'other'
    Returns:
      mapped_label: dict[node] -> class_id in [0..top_k] (size = top_k+1)
      class_names: list[str] length k
    """
    depts = np.array(list(dept_label.values()), dtype=int)
    uniq, cnt = np.unique(depts, return_counts=True)
    order = np.argsort(-cnt)
    top = uniq[order[:top_k]]
    top_set = set(int(x) for x in top)

    top_list = sorted(list(top_set))
    dept_to_new = {dept: i for i, dept in enumerate(top_list)}
    other_id = len(top_list)

    mapped = {}
    for n, d in dept_label.items():
        mapped[n] = dept_to_new[d] if d in top_set else other_id

    class_names = [f"dept_{dept}" for dept in top_list] + ["other"]
    return mapped, class_names


# -------------------- semi-supervised simulation: random unlabel % --------------------

def randomly_unlabel_nodes(G, mapped_label, k, unlabel_pct=0.9, seed=0, min_labeled_per_class=3):
    """
    Hide labels for unlabel_pct fraction of nodes (random).
    Ensures at least min_labeled_per_class labeled nodes per class.
    Sets G.nodes[n]["label"] for labeled nodes only.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())

    # wipe any previous "label"
    for n in nodes:
        if "label" in G.nodes[n]:
            del G.nodes[n]["label"]

    keep = {n: (rng.random() > unlabel_pct) for n in nodes}

    # enforce minimum labeled per class
    for c in range(k):
        class_nodes = [n for n in nodes if mapped_label[n] == c]
        kept = [n for n in class_nodes if keep[n]]
        if len(kept) < min_labeled_per_class:
            need = min_labeled_per_class - len(kept)
            pool = [n for n in class_nodes if not keep[n]]
            if pool:
                add = rng.choice(pool, size=min(need, len(pool)), replace=False)
                for n in add:
                    keep[int(n)] = True

    labeled, unlabeled = [], []
    for n in nodes:
        if keep[n]:
            G.nodes[n]["label"] = int(mapped_label[n])
            labeled.append(n)
        else:
            unlabeled.append(n)

    return labeled, unlabeled


# -------------------- classifier: Ridge one-vs-rest (pure NumPy) --------------------

class RidgeOVR:
    def __init__(self, lam=2.0):
        self.lam = lam
        self.W = None

    def fit(self, X, y, k):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape

        Xb = np.concatenate([X, np.ones((n, 1), dtype=float)], axis=1)  # bias
        D = d + 1
        A = Xb.T @ Xb + self.lam * np.eye(D, dtype=float)

        W = np.zeros((D, k), dtype=float)
        for c in range(k):
            yc = (y == c).astype(float)
            b = Xb.T @ yc
            W[:, c] = np.linalg.solve(A, b)

        self.W = W
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        xb = np.concatenate([x, np.ones((1,), dtype=float)], axis=0)
        scores = xb @ self.W
        return int(np.argmax(scores))


# -------------------- metrics --------------------

def metrics_per_class(y_true, y_pred, k):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    prec = np.zeros(k, dtype=float)
    rec  = np.zeros(k, dtype=float)

    for c in range(k):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec[c] = tp / (tp + fp) if (tp + fp) else 0.0
        rec[c]  = tp / (tp + fn) if (tp + fn) else 0.0

    return acc, prec, rec


# -------------------- main driver --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--unlabel_pct", type=float, default=0.90, help="fraction of nodes to hide labels for")
    ap.add_argument("--min_labeled_per_class", type=int, default=5)
    ap.add_argument("--topk", type=int, default=10, help="keep top-k depts as classes; rest => other")
    ap.add_argument("--lam", type=float, default=2.0, help="ridge regularization")
    args = ap.parse_args()

    # 1) Directed real-world graph + labels
    G, dept_label = load_email_eu_core()

    # 2) Map to manageable multiclass task (topK + other)
    mapped_label, class_names = make_topk_multiclass(dept_label, top_k=args.topk)
    k = len(class_names)

    # 3) Randomly unlabel percentage
    labeled_nodes, unlabeled_nodes = randomly_unlabel_nodes(
        G, mapped_label, k=k,
        unlabel_pct=args.unlabel_pct,
        seed=args.seed,
        min_labeled_per_class=args.min_labeled_per_class
    )

    # 4) Static node features (label-independent): in/out degree, pagerank, clustering
    nodes = list(G.nodes())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    pr = nx.pagerank(G, alpha=0.85)
    clust = nx.clustering(G.to_undirected())

    X = np.zeros((len(nodes), 4), dtype=float)
    for i, n in enumerate(nodes):
        X[i, 0] = np.log1p(in_deg[n])
        X[i, 1] = np.log1p(out_deg[n])
        X[i, 2] = pr[n]
        X[i, 3] = clust[n]

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    feat_map = {n: X[i] for i, n in enumerate(nodes)}

    def node_to_flat_vector_func(node):
        return feat_map[node]

    node_emb_dim = 4
    meta_emb_dim = 2 * k  # because your meta = concat(in_k, out_k)

    # 5) Instantiate YOUR model (it reads labeled/unlabeled from G.nodes[n]["label"])
    model = IterativeClassifer(
        G=G,
        node_to_flat_vector_func=node_to_flat_vector_func,
        node_emb_dim=node_emb_dim,
        meta_emb_dim=meta_emb_dim,
        classifier_func=(lambda vec: 0),  # overwritten each epoch after fitting
        classes=k
    )

    unlabeled = list(model.un_labeled_nodes)

    print(f"Graph: n={G.number_of_nodes()}  m={G.number_of_edges()}  classes={k}")
    print(f"Labeled seeds: {len(model.labeled_nodes)}  Unlabeled (eval): {len(unlabeled)}")
    print("-" * 80)

    # 6) Epoch loop: build features -> fit classifier on labeled -> 1 ICA update -> eval on unlabeled
    for ep in range(1, args.epochs + 1):
        # old preds for "changed"
        old_pred = {}
        for n in unlabeled:
            idx = model.node_to_idx[n]
            old_pred[n] = int(np.argmax(model.idx_to_label_vector[idx]))

        # build current complete vectors from current Y
        model.calculate_all_node_vectors()
        model.aggregate_neighbour_information()
        model.concatenate_vectors()

        # fit on labeled nodes ONLY (true seeds)
        Xtr, ytr = [], []
        for n in model.labeled_nodes:
            idx = model.node_to_idx[n]
            Xtr.append(model.idx_to_complete_vector[idx])
            ytr.append(int(G.nodes[n]["label"]))

        clf = RidgeOVR(lam=args.lam).fit(np.vstack(Xtr), np.array(ytr), k=k)
        model.classifier_func = clf.predict

        # one ICA step
        model.update_label_vector(model.un_labeled_nodes)

        # evaluate on unlabeled
        changed = 0
        y_true, y_pred = [], []
        for n in unlabeled:
            idx = model.node_to_idx[n]
            pred = int(np.argmax(model.idx_to_label_vector[idx]))
            changed += (pred != old_pred[n])
            y_true.append(mapped_label[n])
            y_pred.append(pred)

        acc, prec, rec = metrics_per_class(y_true, y_pred, k=k)

        print(f"Epoch {ep:02d} | changed={changed:4d}/{len(unlabeled)} | acc={acc:.4f}")
        for c in range(k):
            print(f"  {c:2d} ({class_names[c]:>8s}): precision={prec[c]:.4f}  recall={rec[c]:.4f}")
        print("-" * 80)

        # optional early stop on convergence
        if changed == 0:
            break


if __name__ == "__main__":
    main()
