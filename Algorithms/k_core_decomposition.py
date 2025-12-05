import networkx as nx
import numpy as np
from heapq import heappush, heappop

class KCoreDecomposition:

    def __init__(self, G : nx.Graph, K : int = 2):
        self.K = K
        self.G = G

        self.nodes = sorted(list(self.G.nodes()))
        self.num_of_nodes = len(self.nodes)

        self.node_to_idx = {node : idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx : node for idx, node in enumerate(self.nodes)}

        self.idx_to_curr_degree = {idx : self.G.degree(node) for idx, node in enumerate(self.nodes)}
        self.idx_to_core = {idx : 0 for idx in self.idx_to_node}

        self.idx_to_alive = {idx : True for idx in self.idx_to_node}

    def compute_core_numbers(self):
        heap = []
        
        for node in self.nodes:
            curr_idx = self.node_to_idx[node]
            curr_degree = self.idx_to_curr_degree[curr_idx]
            heappush(heap, (curr_degree, curr_idx))


        while heap:
            curr_degree, curr_idx = heappop(heap)
            if not self.idx_to_alive[curr_idx] or   :
                # Stale entry or irrelevant entry
                continue
            if curr_degree > self.K:
                break
            
            self.idx_to_alive[curr_idx] = False
            self.idx_to_core[curr_idx] = curr_degree


            curr_node = self.idx_to_node[curr_idx]

            for neighbor in self.G.neighbors(curr_node):
                neighbor_idx = self.node_to_idx[neighbor]
                if self.idx_to_alive[neighbor_idx] and self.idx_to_curr_degree[neighbor_idx] > curr_degree:
                    # self.idx_to_curr_degree[neighbor_idx] > curr_degree ensures that we only reduce the degree of the dest if the degree is greater then the current, example to check : (0-1), (0,3), (3, 2), (1, 2) => when we remove 0 the degree of 1 and 3 will become 1 and when we remove 1 and mark the core we will get core of 1 = 1, which is incorrect, hence this check is required.
                    self.idx_to_curr_degree[neighbor_idx] -= 1
                    heappush(heap, (self.idx_to_curr_degree[neighbor_idx], neighbor_idx))

        for idx in self.idx_to_node:
            if self.idx_to_alive[idx]:
                self.idx_to_core[idx] = self.K + 1


        