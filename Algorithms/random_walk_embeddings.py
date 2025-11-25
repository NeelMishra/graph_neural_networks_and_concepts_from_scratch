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


        return nodes
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
        self.embeddings = np.random.rand(self.num_nodes, self.embedding_dim)
        self.context_embeddings = np.random.rand(self.num_nodes, self.embedding_dim)

    def feedforward(self, center_idx, context_idx, negative_indices):
        """
        Compute loss for one (center, context, negatives) example.
        """
        pass

    def backprop(self, center_idx, context_idx, negative_indices):
        """
        Do one SGD step and return loss.
        """
        pass

    def similarity(self, node_id_1, node_id_2):
        pass

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

    def fit(self, epochs):
        pass
