import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

'''
LLM Generated documentation

We work with citation networks, where each paper is a node and each citation is
a directed edge. Concretely, we use the Cora citation dataset:

- Nodes: research papers.
- Edges: a directed edge u → v means paper u cites paper v.
- Node features: a bag-of-words vector over the paper’s abstract/title
  (in Cora this is a 1433-dimensional binary feature vector).
- Node labels: each paper belongs to exactly one subject area
  (e.g., "Neural_Networks", "Reinforcement_Learning", etc.).
  We map these subject strings to integer class IDs 0..(num_classes-1).

In code, the raw data is first loaded into a NetworkX graph:

    G: nx.DiGraph
        - G.nodes[i]["x"] : np.ndarray of shape (num_features,)
        - G.nodes[i]["y"] : int class label

We then convert this graph into PyTorch tensors:

    adj : torch.FloatTensor of shape [N, N]
        Adjacency matrix of the graph (1 if there is an edge, 0 otherwise).
    x   : torch.FloatTensor of shape [N, F]
        Node feature matrix, one row per node.
    y   : torch.LongTensor of shape [N]
        Node labels.

Finally, we create a train/test split over the N nodes (e.g., 80% train,
20% test). The GNN is trained to predict the label y[i] for each node i
from its features x[i] and the graph structure encoded in adj.
'''


class GNN:
    '''
    Implemeneted from scratch using the stanford 2019 falls graph neural network course as reference, 224W
    '''
    def __init__(self, data, layer_config):
        '''
        layer_config : 
        [
        {'W': W0, 'B': B0}), # Layer_Idx, layer_configurations as a dict, W0 and B0 are the shape of the W and B
        {'W': W1, 'B': B1}),
        ]
        '''
        self.data = data
        self.layers = []
        for param_shape in layer_config: # Intialize the weights of each layer
            self.layers.append({
                'W' : torch.zeros(param_shape['W'], dtype=torch.float64, requires_grad=True),
                'B' : torch.zeros(param_shape['B'], dtype=torch.float64, requires_grad=True)
            })
    
    def aggregate_information_from_neighbour(self, node, layer_k_node_features, op_type='mean'):

        #layer_k_node_features =>  nodes x feature_dimension => We need mean vector as 1 x feature_dimension?
        result = layer_k_node_features[node].clone()

        count = 1 # self + neighbour couns
        for neighbour in self.data.adj[node]:
            if neighbour == node:
                continue
            result += layer_k_node_features[neighbour]
            count += 1
        if op_type=='mean':
            result /= count 

        return result # F
    

    def feed_forward(self, node_features):
        h = node_features
        for idx, layer in enumerate(self.layers):
            aggregate_information_all_nodes = torch.stack([self.aggregate_information_from_neighbour(node, h) for node in self.data.nodes],
                                                        dim=0) # Every aggregate is of shape F => so in total if we have N nodes, we have NxF
            # W0, B0 => F x W1.shape[0], F x B1.shape[0]

            # W => layer_last_w_shape_1 x layer_w_shape_1
            # B => layer_last_b_shape_1 x layer_b_shape_1
            if idx != len(self.layers) - 1:
                h = torch.sigmoid(
                    aggregate_information_all_nodes @ layer['W']  + h @ layer['B']
                )
            else:
                h = aggregate_information_all_nodes @ layer['W'] + h @ layer['B']
        return h

    def compute_node_features(self):
        """
        Build a [num_nodes, 4] feature matrix X.

        For each node v, features are:
            [ degree(v),
            ego_net_size(v),            # number of neighbors (ego-graph nodes - 1)
            outgoing_edges_from_ego(v), # edges from ego-net to outside
            edges_inside_ego(v) ]       # internal edges in ego-net
        """
        G = self.data
        num_nodes = G.number_of_nodes()
        X = torch.zeros((num_nodes, 4), dtype=torch.float64)
        
        degrees = dict(G.degree())

        for node in G.nodes():
            idx = node  # assuming nodes are 0..N-1

            # Degree of the node
            deg = degrees[node]

            # Egonet around the node (radius 1)
            egonet = nx.ego_graph(G, node, radius=1)
            ego_nodes = set(egonet.nodes())
            ego_size = len(ego_nodes) - 1  # neighbors only

            # Edges inside egonet
            edges_inside = egonet.number_of_edges()

            # Edges going from egonet to outside
            outgoing = 0
            for u in ego_nodes:
                for v in G.neighbors(u):
                    if v not in ego_nodes:
                        outgoing += 1

            X[idx] = torch.tensor([deg, ego_size, outgoing, edges_inside],
                                dtype=torch.float64)

        return X
        
    def train(self, Y, epochs, criterion, optimizer):

        node_features = self.compute_node_features() # X_0

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.feed_forward(node_features=node_features) 
            loss = criterion(logits, Y) # N, N_classes

            print(f"epoch : {epoch}, loss : {loss.detach().cpu().item()}")

            loss.backward()
            optimizer.step()
        

    def predict(self):
        node_features = self.compute_node_features()
        return self.feed_forward(node_features=node_features)


## LLM Generated Driver Code
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import networkx as nx

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"


def download_cora(data_dir: str = "data/cora"):
    """
    Download and extract the Cora citation dataset if not already present.

    Returns
    -------
    cites_path : str
        Path to 'cora.cites'.
    content_path : str
        Path to 'cora.content'.
    """
    os.makedirs(data_dir, exist_ok=True)
    cites_path = os.path.join(data_dir, "cora.cites")
    content_path = os.path.join(data_dir, "cora.content")

    # If files already exist, skip download
    if os.path.exists(cites_path) and os.path.exists(content_path):
        return cites_path, content_path

    archive_path = os.path.join(data_dir, "cora.tgz")
    print("Downloading Cora dataset...")
    urllib.request.urlretrieve(CORA_URL, archive_path)

    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    # After extraction, files are in data_dir/cora/
    src_dir = os.path.join(data_dir, "cora")
    for fname in ["cora.cites", "cora.content", "README"]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            os.replace(src, os.path.join(data_dir, fname))

    # Optionally remove the intermediate folder
    try:
        os.rmdir(src_dir)
    except OSError:
        pass

    return cites_path, content_path


def load_cora_as_networkx(data_dir: str = "data/cora", relabel_to_int: bool = True):
    """
    Load the Cora citation network as a NetworkX DiGraph.

    Node attributes:
        - 'x' : np.ndarray, shape (F,)   (F is num_features, typically 1433)
        - 'y' : int                      class index in [0, num_classes)

    Graph attributes:
        - G.graph['classes']     : list of class names (strings)
        - G.graph['label_map']   : dict {class_name: class_index}
        - G.graph['num_features']: int, number of input features
        - (optional) G.graph['idx_to_paper'] : {new_id: original_paper_id}
        - (optional) G.graph['paper_to_idx'] : {original_paper_id: new_id}

    Parameters
    ----------
    data_dir : str
        Directory to store/download Cora files.
    relabel_to_int : bool
        If True, relabel nodes to 0..N-1 (good for tensor indexing).

    Returns
    -------
    G : nx.DiGraph
    """
    cites_path, content_path = download_cora(data_dir)

    # --- Load citation edges ---
    # File format: each line "cited_paper_id  citing_paper_id"
    citations = pd.read_csv(
        cites_path,
        sep="\t",
        header=None,
        names=["target", "source"],
    )

    # Build a directed citation graph: source -> target (paper cites another paper)
    G = nx.from_pandas_edgelist(
        citations,
        source="source",
        target="target",
        create_using=nx.DiGraph(),
    )

    # --- Load paper features and labels ---
    # cora.content: paper_id, term_0, term_1, ..., term_1432, subject
    num_features = 1433
    column_names = ["paper_id"] + [f"term_{i}" for i in range(num_features)] + ["subject"]

    papers = pd.read_csv(
        content_path,
        sep="\t",
        header=None,
        names=column_names,
    )

    paper_ids = papers["paper_id"].to_numpy()
    features = papers.iloc[:, 1:-1].to_numpy(dtype=np.float32)  # [N, 1433]
    subjects = papers["subject"].to_numpy()                     # [N]

    # Map subject strings to integer labels
    classes, y_int = np.unique(subjects, return_inverse=True)
    label_map = {cls: i for i, cls in enumerate(classes)}

    # Ensure all nodes exist and attach attributes
    for i, pid in enumerate(paper_ids):
        if pid not in G:
            G.add_node(pid)
        G.nodes[pid]["x"] = features[i]
        G.nodes[pid]["y"] = int(y_int[i])

    # Store metadata
    G.graph["classes"] = classes.tolist()
    G.graph["label_map"] = label_map
    G.graph["num_features"] = num_features

    print(
        f"Loaded Cora: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, "
        f"{len(classes)} classes, "
        f"{num_features} features."
    )

    # Optionally relabel nodes to 0..N-1 for easier tensor indexing
    if relabel_to_int:
        old_nodes = list(G.nodes())
        mapping = {old_id: new_id for new_id, old_id in enumerate(old_nodes)}
        G_relabelled = nx.relabel_nodes(G, mapping, copy=True)

        # Save mappings in graph attributes
        G_relabelled.graph["idx_to_paper"] = {v: k for k, v in mapping.items()}
        G_relabelled.graph["paper_to_idx"] = mapping

        return G_relabelled

    return G



if __name__ == "__main__":
    lr = 1e-1
    G = load_cora_as_networkx()
    criterion = CrossEntropyLoss()
    layer_config = [
        {
            "W": (4, 10),
            "B" : (4, 10),
        },
        {
            "W" : (10, 4),
            "B" : (10, 4)
        },
        {
            "W" : (4, 7),
            "B" : (4, 7)
        }
    ]
    gnn_obj = GNN(G, layer_config)
    params_list = []
    for layer in gnn_obj.layers:
        params_list.append(layer['W'])
        params_list.append(layer['B'])

    optimizer = Adam(params_list, lr = lr)

    Y = torch.tensor(
        [G.nodes[i]["y"] for i in range(G.number_of_nodes())],
        dtype=torch.long
    )

    gnn_obj.train(Y, 100, criterion, optimizer)
