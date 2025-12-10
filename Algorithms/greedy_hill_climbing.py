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