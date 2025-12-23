import numpy as np

class TransR:

    def __init__(self, G, gamma=1e-1, emb_dim=128, rel_dim=64):
        self.G = G
        self.default_relation = "connected_to"
        self.emb_dim = emb_dim
        self.gamma = gamma
        self.rel_dim = rel_dim

        self.entities = list(self.G.nodes())
        self.entity_to_idx = {entity : i for i, entity in enumerate(self.entities)}
        self.idx_to_entity = {i : entity for i, entity in enumerate(self.entities)}

        self.triples = []
        relations_set = set()

        for h, t, attr in self.G.edges(data=True):
            r = attr.get("relation", self.default_relation)
            self.triples.append((h, r, t))
            relations_set.add(r)

        self.relations = sorted(relations_set)
        self.relation_to_idx = {r : i for i, r in enumerate(self.relations)}
        self.idx_to_relation = {i : r for i, r in enumerate(self.relations)}

        self.initialize_random_embeddings()

    def initialize_random_embeddings(self, scale = 1e-2):
        
        n = len(self.entities)
        m = len(self.relations)

        self.node_embeddings = (np.random.standard_normal((n, self.emb_dim)) * scale).astype(float)
        self.relation_embeddings = (np.random.standard_normal((m, self.rel_dim)) * scale).astype(float)
        self.proj_mats = (np.random.standard_normal((m, self.rel_dim, self.emb_dim)) * scale).astype(float)

        self.project_entities()

    def sample(self, batch_size):
        rand_idxs = np.random.randint(0, len(self.triples), size=batch_size)
        pos_batch = [self.triples[i] for i in rand_idxs]
        
        neg_batch =  []

        for h,r,t in pos_batch:
            dice = np.random.rand()

            if dice < 0.5:
                # Corrupt head
                h_neg  = h
                while h_neg == h:
                    h_neg = np.random.choice(self.entities)
                neg_batch.append((h_neg, r, t))
            else:
                t_neg = t
                while t_neg == t:
                    t_neg = np.random.choice(self.entities)
                neg_batch.append((h, r, t_neg))

        return pos_batch, neg_batch
        

    def triplet_loss(self, pos_batch, neg_batch):
        total_sum = 0.0

        for (h,r,t), (h2,r2,t2) in zip(pos_batch, neg_batch):
            d_pos = self.distance_calculation(h, r, t)
            d_neg = self.distance_calculation(h2, r2, t2)
            total_sum += max(0, self.gamma + d_pos - d_neg)
        
        return total_sum


    def gradients(self, pos_batch, neg_batch, eps=1e-9):

        entities_grad = np.zeros((len(self.entities), self.emb_dim), dtype=float)
        relations_grad = np.zeros((len(self.relations), self.rel_dim), dtype=float)
        M_grad = np.zeros((len(self.relations), self.rel_dim, self.emb_dim), dtype=float)

        for (h,r,t), (h2,r2,t2) in zip(pos_batch, neg_batch):

            h_idx = self.entity_to_idx[h]
            t_idx = self.entity_to_idx[t]
            r_idx = self.relation_to_idx[r]

            M_r = self.proj_mats[r_idx]

            h2_idx = self.entity_to_idx[h2]
            t2_idx = self.entity_to_idx[t2]

            delta_pos = M_r @ self.node_embeddings[h_idx] + self.relation_embeddings[r_idx] - M_r @ self.node_embeddings[t_idx]
            delta_neg = M_r @ self.node_embeddings[h2_idx] + self.relation_embeddings[r_idx] - M_r @ self.node_embeddings[t2_idx]

            d_pos = np.linalg.norm(delta_pos)
            d_neg = np.linalg.norm(delta_neg)

            s = self.gamma + d_pos - d_neg

            if s <= 0:
                continue
            else:
                
                g_pos = delta_pos/(eps + d_pos)
                g_neg = delta_neg/(eps + d_neg)

                eh_minus_et = self.node_embeddings[h_idx] - self.node_embeddings[t_idx]
                eh2_minus_et2 = self.node_embeddings[h2_idx] - self.node_embeddings[t2_idx]

                g_pos_projected = M_r.T @ g_pos
                g_neg_projected = M_r.T @ g_neg

                entities_grad[h_idx] += g_pos_projected
                entities_grad[t_idx] -= g_pos_projected

                entities_grad[h2_idx] -= g_neg_projected
                entities_grad[t2_idx] += g_neg_projected

                relations_grad[r_idx] += g_pos - g_neg

                M_grad[r_idx] += np.outer(g_pos, eh_minus_et)
                M_grad[r_idx] -= np.outer(g_neg, eh2_minus_et2)



        return entities_grad, relations_grad, M_grad

    def distance_calculation(self, h, r, t):
        # return || e_h + e_r - e_t ||_p

        h_idx = self.entity_to_idx[h]
        t_idx = self.entity_to_idx[t]
        r_idx = self.relation_to_idx[r]
        M_r = self.proj_mats[r_idx]

        delta_pos = M_r @ self.node_embeddings[h_idx] + self.relation_embeddings[r_idx] - M_r @ self.node_embeddings[t_idx]

        return np.linalg.norm(delta_pos)

    def project_entities(self):
        self.node_embeddings = self.node_embeddings/(np.maximum(1.0, np.linalg.norm(self.node_embeddings, axis=1, keepdims=True)))
        return
    
    def project_relations(self):
        self.relation_embeddings = self.relation_embeddings/(np.maximum(1.0, np.linalg.norm(self.relation_embeddings, axis=1, keepdims=True)))
        
        # Row normalization
        row_norm = np.linalg.norm(self.proj_mats, axis=2, keepdims=True)  # (m, rel_dim, 1)
        self.proj_mats /= np.maximum(1.0, row_norm)
        
        return

    def fit(self, lr, epochs, steps_per_epoch, batch_size):

        for epoch in range(epochs):
            loss = 0.0
            for step in range(steps_per_epoch):
                pos_batch, neg_batch = self.sample(batch_size)
                loss += self.triplet_loss(pos_batch, neg_batch)
                
                entities_grad, relations_grad, M_grad = self.gradients(pos_batch, neg_batch)

                self.node_embeddings -= lr * entities_grad
                self.relation_embeddings -= lr * relations_grad
                self.proj_mats -= lr * M_grad

                self.project_entities()
                self.project_relations()

            print(f"Avg Loss in the epoch {epoch} is {loss/steps_per_epoch}")



##### LLM GENERATED DRIVER CODE
# ===== Driver code for your TransR class (ignore TensorBoard if you want) =====
if __name__ == "__main__":
    import argparse
    import time
    import networkx as nx

    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--rel_dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    # ----------------------------
    # 1) Build a directed multi-relation-ish graph
    # ----------------------------
    G_und = nx.karate_club_graph()

    G = nx.DiGraph()
    G.add_nodes_from(G_und.nodes(data=True))

    # Add edges both directions and set a relation attribute
    for u, v in G_und.edges():
        G.add_edge(u, v, relation="connected_to")
        G.add_edge(v, u, relation="connected_to")

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # ----------------------------
    # 2) Train TransR
    # ----------------------------
    model = TransR(G, gamma=args.gamma, emb_dim=args.emb_dim, rel_dim=args.rel_dim)

    # If steps_per_epoch <= 0, do ~one pass worth of mini-batches
    if args.steps_per_epoch <= 0:
        args.steps_per_epoch = max(1, len(model.triples) // args.batch_size)

    t0 = time.time()
    model.fit(
        lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )
    print(f"Training done in {time.time() - t0:.2f}s")

    # ----------------------------
    # 3) Quick sanity: show top neighbors by TransR score for one node
    # ----------------------------
    # Lower distance => "more plausible" triple (h, r, t)
    r = model.relations[0]
    h = model.entities[0]

    scores = []
    for t in model.entities:
        scores.append((t, model.distance_calculation(h, r, t)))

    scores.sort(key=lambda x: x[1])
    print(f"\nHead={h}, relation={r}. Top-10 closest tails:")
    for t, d in scores[:10]:
        print(f"  tail={t:>2}  dist={d:.4f}")
