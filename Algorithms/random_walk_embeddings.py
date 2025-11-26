import numpy as np
import networkx as nx

class RandomWalkEmbeddings:

    def __init__(self, graph):
        self.G = graph

        self.embeddings = None
        self.context_embeddings = None

        self.p = None # Return parameter
        self.q = None # in-out parameter

        self.nodes = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for i, n in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)
        self.embedding_dim = 128 

        self.initialize_random_vectors()
    

    def _sample_next_node(self, prev_node, current_node):
        """
        Use the appropriate transition probs (first step vs later steps)
        to sample and return the next node in the walk.
        """
        if prev_node == None:
            neighbours, probs = self._compute_transition_probs_first_step(current_node)
        else:
            neighbours, probs = self._compute_transition_probs_second_order(prev_node, current_node)

        if len(neighbours) == 0 or len(probs) == 0:
            return None
        
        return np.random.choice(neighbours, p=probs)

    def _compute_transition_probs_first_step(self, start_node):
        neighbours = list(self.G.adj[start_node])

        if not neighbours:
            return [], np.array([], dtype=float)
        
        weights = []
        for neighbour in neighbours:
            edge_data = self.G[start_node][neighbour]
            weights.append(edge_data.get('weight', 1))
        
        weights = np.array(weights, dtype=float)
        total = weights.sum()

        if total == 0:
            return [], np.array([], dtype=float)

        return neighbours, weights/total
            

    def _compute_transition_probs_second_order(self, prev_node, curr_node):
        nodes = [[prev_node, 0]] # node, distance
        seen = {prev_node, curr_node}

        # distance 1 nodes
        for adj_node in self.G.adj[prev_node]:
            if adj_node in self.G.adj[curr_node]:
                nodes.append([adj_node, 1])
                seen.add(adj_node)

        # distance 2 nodes
        for adj_node in self.G.adj[curr_node]:
            if adj_node not in seen:
                nodes.append([adj_node, 2])
                seen.add(adj_node)

        un_normalized_weights = []
        neighbours = []
        for node, dist in nodes:
            neighbours.append(node)
            wt = self.G[curr_node][node].get('weight', 1.0)
            if dist == 0:
                un_normalized_weights.append(wt * 1/self.p)
            elif dist == 1:
                un_normalized_weights.append(wt * 1)
            else:
                un_normalized_weights.append(wt * 1/self.q )

        un_normalized_weights = np.array(un_normalized_weights, dtype=float)
        total_sum = un_normalized_weights.sum()

        if total_sum == 0:
            return [], np.array([], dtype=float)
        
        return neighbours, un_normalized_weights/total_sum

    def random_walk(self, start_node, walk_length):
        walk = [start_node]

        prev = None
        current = start_node

        for i in range(walk_length - 1):
            next_node = self._sample_next_node(prev, current)

            if next_node is None:
                break

            walk.append(next_node)
            prev, current = current, next_node

        return walk


    def initialize_random_vectors(self):
        scale = 0.01
        self.embeddings = np.random.randn(self.num_nodes, self.embedding_dim) * scale
        self.context_embeddings = np.random.randn(self.num_nodes, self.embedding_dim) * scale


    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def feedforward(self, center_idx, context_idx, negative_indices, eps=1e-12):
        """
        Compute loss for one (center, context, negatives) example.
        """
        center_embedding = self.embeddings[center_idx]
        context_embedding = self.context_embeddings[context_idx]
        negative_context_embeddings = []
        for idx in negative_indices:
            negative_context_embeddings.append(self.context_embeddings[idx])
        
        negative_context_embeddings = np.array(negative_context_embeddings, dtype=float)

        pos_dot = np.dot(center_embedding, context_embedding)
        neg_dot = np.dot(negative_context_embeddings, center_embedding)

        return np.log(self.sigmoid(pos_dot) + eps) + np.sum(np.log(self.sigmoid(-neg_dot) + eps), axis=0)

    def backprop(self, center_idx, context_idx, negative_indices):
        """
        Do one SGD step
        """
        
        center_embedding = self.embeddings[center_idx]
        context_embedding = self.context_embeddings[context_idx]
        negative_context_embeddings = []
        for idx in negative_indices:
            negative_context_embeddings.append(self.context_embeddings[idx])
    
        negative_context_embeddings = np.array(negative_context_embeddings, dtype=float)

        x = self.sigmoid(np.dot(-center_embedding.T, context_embedding))
        y = self.sigmoid(np.dot(negative_context_embeddings,center_embedding))

        grad_u = np.dot(x, context_embedding) - np.sum(y[:, None] * negative_context_embeddings, axis=0) 

        grad_v = np.dot(x, center_embedding)
        grad_n = -y[:, None] * center_embedding

        self.embeddings[center_idx] += self.lr * grad_u
        self.context_embeddings[context_idx] += self.lr * grad_v
        
        self.context_embeddings[negative_indices] += self.lr * grad_n

        return




    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x/np.sum(exp_x) 

    def generate_walks(self, length):
        walks = []

        for node in self.G.nodes():
            current_walk = [node]

            prev_node = None
            curr_node = node
            next_node = None

            for i in range(length - 1):
                next_node = self._sample_next_node(prev_node, curr_node)

                if next_node is None:
                    break

                prev_node = curr_node
                curr_node = next_node
                
                current_walk.append(next_node)
            walks.append(current_walk)
        
        return walks

    def fit(self, epochs, walk_length=20, random_sample_size=5):
        total_nodes = list(self.G.nodes())
        all_ids = np.arange(len(total_nodes))

        for i in range(epochs):
            total_loss = 0
            counter = 0
            for center_node in total_nodes:
                walk = self.random_walk(center_node, walk_length=walk_length)
                center_id = self.node_to_idx[center_node]
                walk_ids = np.array([self.node_to_idx[n] for n in walk], dtype=int)
                allowed_neg = np.setdiff1d(all_ids, walk_ids, assume_unique=False)

                for node in walk[1:]:
                    node_id = self.node_to_idx[node]
                    random_sample_ids = np.random.choice(allowed_neg, size=random_sample_size, replace=True)
                    loss = self.feedforward(center_id, node_id, random_sample_ids)
                    self.backprop(center_id, node_id, random_sample_ids)
                    total_loss += loss
                    counter += 1

                
            print(f"The average loss in the epoch {i} is {-total_loss/counter}")



##### LLM GENERATED DRIVER CODE => Main goal is to implement the algorithms from scratch not the driver code
import io
import os
import zipfile
import urllib.request

import numpy as np
import networkx as nx


# ---- paste your RandomWalkEmbeddings class ABOVE this main block ----
# ---- Paste your RandomWalkEmbeddings class ABOVE this main block ----
if __name__ == "__main__":
    import numpy as np
    import networkx as nx
    from tensorboardX import SummaryWriter

    np.random.seed(42)

    # =========================
    # 1) Real-world graph with exactly 2 groups
    # =========================
    G = nx.karate_club_graph()  # node attr "club": "Mr. Hi" or "Officer"
    print(f"Karate graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # =========================
    # 2) Train with YOUR implementation
    # =========================
    model = RandomWalkEmbeddings(G)

    # MUST set these for your code
    model.lr = 0.005
    model.p = 1.0
    model.q = 1.0

    # IMPORTANT for small graphs:
    # Keep walk_length small so allowed_neg doesn't become empty.
    EPOCHS = 1000
    WALK_LENGTH = 30          # small to avoid walk covering all nodes
    NEG_K = 5                # small negatives for stability on small graph

    model.fit(epochs=EPOCHS, walk_length=WALK_LENGTH, random_sample_size=NEG_K)

    # =========================
    # 3) TensorBoard Embedding Projector
    # =========================
    emb = ((model.embeddings + model.context_embeddings) / 2.0).astype(np.float32)

    metadata = []
    for n in model.nodes:
        club = G.nodes[n]["club"]  # "Mr. Hi" or "Officer"
        metadata.append([str(n), club])

    LOGDIR = "runs/karate_two_groups"
    writer = SummaryWriter(LOGDIR)
    writer.add_embedding(
        mat=emb,
        metadata=metadata,
        metadata_header=["node", "club"],
        tag="karate_embeddings"
    )
    writer.close()

    print("\n✅ Wrote TensorBoard logs:", LOGDIR)
    print("Run:")
    print("  tensorboard --logdir runs")
    print("\nThen TensorBoard → Projector → select 'karate_embeddings' → Color by: 'club'")
