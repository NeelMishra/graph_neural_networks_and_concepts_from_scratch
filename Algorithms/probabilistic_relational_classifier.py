import numpy as np
import networkx as nx

class ProbabilisticRelationalClassifier:

    def __init__(self, G, classes):
        self.G = G
        self.k = classes

        self.node_to_label_vector = {}
        
    
    def get_labeled_nodes(self):
        labeled_nodes = set()

        for node in self.G.nodes():
            label = self.G.nodes[node].get("label", None)
            if label is not None:
                labeled_nodes.add(node)
        
        return labeled_nodes

    def get_unlabelled_nodes(self):
        unlabeled_nodes = set()

        for node in self.G.nodes():
            label = self.G.nodes[node].get("label", None)
            if label is None:
                unlabeled_nodes.add(node)
        
        return unlabeled_nodes

    def uniform_initialize_unlabeled_node(self, unlabeled_nodes):
        
        for node in unlabeled_nodes:
            self.G.nodes[node]["label"] = np.random.uniform(low = 0, high=1, size=(self.k))
        

    def get_neighbour_labels(self):
        pass

    def calculate_neighbour_aggregation(self):
        pass

    def forward(self):
        pass

    def train(self, epochs):
        pass


