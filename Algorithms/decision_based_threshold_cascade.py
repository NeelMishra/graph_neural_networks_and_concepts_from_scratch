import networkx as nx
import numpy as np

class ThresholdCascade:
    ''' 2 set of labels A and B
        In real world this could be an example of 
            * Polarization vs non Polarization
            * Adoption of product A vs product B
    '''
    def __init__(self, G : nx.Graph, threshold : float = 0.5):
        self.threshold = threshold
        self.G = G

        self.nodes = sorted(list(self.G.nodes()))
        self.num_of_nodes = len(self.nodes)

        self.node_to_idx = {node : idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx : node for idx, node in enumerate(self.nodes)}

        self.idx_to_label = {idx : 'B' for idx in range(self.num_of_nodes)} # Initially all the nodes are have label B (non-adoption at start)

    
    def get_neighbor_labels(self, node):
        count_A = 0
        count_B = 0

        for neighbor in self.G.neighbors(node):
            neighbor_idx = self.node_to_idx[neighbor]
            neighbor_label = self.idx_to_label[neighbor_idx]

            if neighbor_label == 'A':
                count_A += 1
            else:
                count_B += 1

        return count_A, count_B
    
    def iterate(self):
        total_nodes_changed = 0
        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            count_A, count_B = self.get_neighbor_labels(node)
            if count_A == count_B and count_A == 0:
                continue
            
            ratio = count_A / (count_A + count_B)
            if ratio >= self.threshold and self.idx_to_label[node_idx] == 'B':
                self.idx_to_label[node_idx] = 'A'
                total_nodes_changed += 1

        return total_nodes_changed
    


class ABCascadeBestResponse:

    def __init__(self, G : nx.Graph, cost, reward_A, reward_B):
        self.cost = cost
        self.G = G
        self.reward_A = reward_A
        self.reward_B = reward_B

        self.nodes = sorted(list(self.G.nodes()))
        self.num_of_nodes = len(self.nodes)

        self.node_to_idx = {node : idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx : node for idx, node in enumerate(self.nodes)}

        self.idx_to_label = {idx : 'B' for idx in range(self.num_of_nodes)} # Initially all the nodes are have label B (non-adoption at start)

    def get_neighbor_labels(self, node):
        count_A = 0
        count_B = 0
        count_AB = 0

        for neighbor in self.G.neighbors(node):
            neighbor_idx = self.node_to_idx[neighbor]
            neighbor_label = self.idx_to_label[neighbor_idx]

            if neighbor_label == 'A':
                count_A += 1
            elif neighbor_label == 'B':
                count_B += 1
            else:
                count_AB += 1

        return count_A, count_B, count_AB
    
    def iterate(self):
        total_nodes_changed = 0
        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            count_A, count_B, count_AB = self.get_neighbor_labels(node)
            if count_A == count_B and count_A == count_AB and count_A == 0:
                continue
            
            total = count_A + count_B + count_AB
            UA = self.reward_A * (count_A + count_AB)
            UB = self.reward_B * (count_B + count_AB)
            UAB = self.reward_A * (count_A) + self.reward_B * (count_B) + max(self.reward_A, self.reward_B) * count_AB - self.cost 
            
            if UA > UB:

                if UA > UAB:
                    self.idx_to_label[node_idx] = 'A'
                    total_nodes_changed += 1
                else:
                    self.idx_to_label[node_idx] = 'AB'
                    total_nodes_changed += 1

            else:
                if UB > UAB:
                    self.idx_to_label[node_idx] = 'B'
                    total_nodes_changed += 1
                else:
                    self.idx_to_label[node_idx] = 'AB'
                    total_nodes_changed += 1

        return total_nodes_changed