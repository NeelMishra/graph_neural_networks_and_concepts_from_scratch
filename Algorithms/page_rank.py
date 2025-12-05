import numpy as np
import networkx as nx

class PageRank:
    ''' This implementation is for directed graph only, check using self.G.is_directed()'''
    def __init__(self, G: nx.DiGraph, beta: float = 0.9):
        
        if not G.is_directed():
            raise NotImplementedError("The current implementation is for directed graphs only")

        self.G = G
        self.beta = beta

        self.nodes = sorted(list(self.G.nodes()))
        self.num_of_nodes = len(self.nodes)

        self.idx_to_node = {idx : node for idx, node in enumerate(self.nodes)}
        self.node_to_idx = {node : idx for idx, node in enumerate(self.nodes)}

        self.page_rank = np.ones((self.num_of_nodes, 1), dtype=np.float64)/self.num_of_nodes # N x 1, N => Number of nodes
        self.M = np.zeros((self.num_of_nodes, self.num_of_nodes), dtype=np.float64) # Weigted Adjacency/Transistion Matrix, N x N

        self.out_degree = {node : self.G.out_degree(node) for node in self.nodes}
        self.build_transition_matrix()

    def build_transition_matrix(self):
        
        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            node_outdegree = self.out_degree[node]
            if node_outdegree == 0:
                continue
            for neighbor in self.G.neighbors(node):
                neighbor_idx = self.node_to_idx[neighbor]
                self.M[neighbor_idx, node_idx] += 1/node_outdegree # i->j => rj += 1/d_i

    def update_r(self):
        r_old = self.page_rank
        r_new = np.zeros((self.num_of_nodes, 1), dtype=np.float64)

        for node in self.nodes: # i
            node_idx = self.node_to_idx[node] 
            if self.out_degree[node] == 0:
                continue
            for neighbor in self.G.neighbors(node): # j
                # i -> j
                neighbor_idx = self.node_to_idx[neighbor]
                r_new[neighbor_idx] += self.beta * r_old[node_idx] / self.out_degree[node]

        
        # Teleportation exists in real life
        S = r_new.sum()
        r_new = r_new + (1-S)/self.num_of_nodes

        self.page_rank = r_new 
        return np.linalg.norm(r_new - r_old)

    def fit(self, epochs, tol=1e-8):

        for epoch in range(epochs):
            diff = self.update_r()
            if diff < tol:
                break
        return self.page_rank

class PageRankByRandomWalk:
    ''' This implementation is for directed graph only, check using self.G.is_directed()'''
    def __init__(self, G: nx.DiGraph, beta: float = 0.9):
        
        if not G.is_directed():
            raise NotImplementedError("The current implementation is for directed graphs only")

        self.G = G
        self.beta = beta

        self.nodes = sorted(list(self.G.nodes()))
        self.num_of_nodes = len(self.nodes)

        self.idx_to_node = {idx : node for idx, node in enumerate(self.nodes)}
        self.node_to_idx = {node : idx for idx, node in enumerate(self.nodes)}

        self.page_rank = np.ones((self.num_of_nodes, 1), dtype=np.float64)/self.num_of_nodes # N x 1, N => Number of nodes
        self.M = np.zeros((self.num_of_nodes, self.num_of_nodes), dtype=np.float64) # Weigted Adjacency/Transistion Matrix, N x N

        self.out_degree = {node : self.G.out_degree(node) for node in self.nodes}
        self.build_transition_matrix()

    def build_transition_matrix(self):
        
        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            node_outdegree = self.out_degree[node]
            if node_outdegree == 0:
                continue
            for neighbor in self.G.neighbors(node):
                neighbor_idx = self.node_to_idx[neighbor]
                self.M[neighbor_idx, node_idx] += 1/node_outdegree # i->j => rj += 1/d_i

    def update_r(self):
        r_old = self.page_rank
        r_new = np.zeros((self.num_of_nodes, 1), dtype=np.float64)

        for node in self.nodes: # i
            node_idx = self.node_to_idx[node] 
            if self.out_degree[node] == 0:
                continue
            for neighbor in self.G.neighbors(node): # j
                # i -> j
                neighbor_idx = self.node_to_idx[neighbor]
                r_new[neighbor_idx] += self.beta * r_old[node_idx] / self.out_degree[node]

        
        # Teleportation exists in real life
        S = r_new.sum()
        r_new = r_new + (1-S)/self.num_of_nodes

        self.page_rank = r_new 
        return np.linalg.norm(r_new - r_old)

    def fit(self, epochs, tol=1e-8):

        for epoch in range(epochs):
            diff = self.update_r()
            if diff < tol:
                break
        return self.page_rank
    

        

