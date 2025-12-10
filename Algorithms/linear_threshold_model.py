import networkx as nx
import numpy as np
from collections import deque

class LTM:

    def __init__(self, G: nx.DiGraph, threshold: float):
        self.G = G
        self.threshold = threshold

        self.nodes = list(self.G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        self.idx_to_label = {idx: 0 for idx in self.idx_to_node}

    def activate_nodes(self, nodes, labels):
        
        for node, label in zip(nodes, labels):
            node_idx = self.node_to_idx[node]
            self.idx_to_label[node_idx] = label


    def propogate(self):
        changed = []
        new_labels = self.idx_to_label.copy()

        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            if self.idx_to_label[node_idx] == 1:
                continue
            weighted_sum = 0

            for neighbor in self.G.predecessors(node):
                neighbor_idx = self.node_to_idx[neighbor]
                if self.idx_to_label[neighbor_idx] == 1:
                    weighted_sum += self.G[neighbor][node].get("weight", 0.0)
                

            if weighted_sum >= self.threshold:
                changed.append(node)
                new_labels[node_idx] = 1

        self.idx_to_label = new_labels
        return changed
    

    def fit(self, active_nodes, labels, epochs):
        self.activate_nodes(active_nodes, labels)
        for epoch in range(epochs):
            changed_nodes = self.propogate()
            if not changed_nodes:
                break
        return
