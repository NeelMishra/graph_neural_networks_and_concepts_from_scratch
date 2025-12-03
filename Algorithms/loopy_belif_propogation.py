import torch
import torch.nn as nn
import networkx as nx


class LoopyBP(nn.Module):
    def __init__(self, G: nx.Graph, classes: int, alpha=1e-1, eps=1e-12):
        super().__init__()
        self.G = G
        self.k = int(classes)
        self.eps = float(eps)

        self.device = torch.device('cpu')
        self.dtype = torch.float64

        self.nodes = list(self.G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}

        self.get_labelled_and_unlabelled_nodes()

        prior = torch.full((self.n, self.k), 1.0 / self.k, device=self.device, dtype=self.dtype)

        p_main = 0.8
        p_rest = (1.0 - p_main) / (self.k - 1) if self.k > 1 else 0.0

        for label_node in self.labeled_nodes:
            i = self.node_to_idx[label_node]
            y = int(self.G.nodes[label_node]["label"])

            prior[i].fill_(p_rest)      # set all labels
            prior[i, y] = p_main        #overwrite the true label to have largest probability
        
        prior = prior / (prior.sum(dim=1, keepdim=True).clamp_min(self.eps))
        self.prior = prior  

        self.label_to_label_potential = nn.Parameter(
            torch.rand(self.k, self.k, device=self.device, dtype=self.dtype)
        )

        messages = torch.rand(self.n, self.n, self.k, device=self.device, dtype=self.dtype)
        messages = messages / (messages.sum(dim=2, keepdim=True).clamp_min(self.eps))
        self.messages = messages 

        self.idx_to_label_vector = self.prior.clone()

    def get_labelled_and_unlabelled_nodes(self):
        labeled_nodes = set()
        unlabeled_nodes = set()

        for node in self.G.nodes():
            label = self.G.nodes[node].get("label", None)
            if label is not None:
                labeled_nodes.add(node)
            else:
                unlabeled_nodes.add(node)

        self.labeled_nodes = labeled_nodes
        self.un_labeled_nodes = unlabeled_nodes

    def get_neighbour_messages(self, node, exclude_node):
        node_idx = self.node_to_idx[node]
        neighbors = [nbr for nbr in self.G.neighbors(node) if nbr != exclude_node]

        if len(neighbors) == 0:
            return torch.ones(self.k, device=self.device, dtype=self.dtype)

        prod = torch.ones(self.k, device=self.device, dtype=self.dtype)
        for nbr in neighbors:
            nbr_idx = self.node_to_idx[nbr]
            prod = prod * self.messages[nbr_idx, node_idx]  # (k,)

        return prod
    
    def psi_matrix(self):
        return torch.exp(self.label_to_label_potential)
    
    def update_message(self, i, j):
        i_idx = self.node_to_idx[i]

        psi_matrix = self.psi_matrix()
        neighbour_messages = self.get_neighbour_messages(i, j)

        b = (self.prior[i_idx]  * neighbour_messages).clamp_min(self.eps) # (k,) * (k, ) => Element wise multiplication
        m_new = (b @ psi_matrix)
        m_new /= m_new.sum().clamp_min(self.eps)

        return m_new

    def belifs(self):

        belif_matrix = torch.zeros_like(self.prior)

        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            neighbour_messages = self.get_neighbour_messages(node, None)

            b = (self.prior[node_idx]  * neighbour_messages).clamp_min(self.eps)
            b /= b.sum().clamp_min(self.eps)

            belif_matrix[node_idx] = b

        return belif_matrix

    def run_bp(self, iterations):

        for iter in range(iterations):
            updated_messages = self.messages.clone()

            for edge in self.G.edges():
                i, j = edge    
                i_idx, j_idx = self.node_to_idx[i], self.node_to_idx[j]
                updated_messages[i_idx, j_idx] = self.update_message(i, j)
                if not self.G.is_directed():
                    updated_messages[j_idx, i_idx] = self.update_message(j, i)

            self.messages = updated_messages
    
    def forward(self, iterations):
        self.run_bp(iterations)
        return self.belifs()

    def nll_loss(self):
        belifs = self.belifs()

        labeled_nodes = list(self.labeled_nodes)
        labeled_idx = torch.tensor([self.node_to_idx[node] for node in labeled_nodes], dtype=torch.long)

        y_true = torch.tensor([int(self.G.nodes[node]["label"]) for node in labeled_nodes], dtype=torch.long)

        p_true = belifs[labeled_idx, y_true].clamp_min(self.eps)

        return - torch.log(p_true).mean()
    

#### LLM Generated driver code
