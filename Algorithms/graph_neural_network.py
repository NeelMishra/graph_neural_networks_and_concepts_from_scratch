import torch

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
                'W' : torch.tensor(param_shape['W'], dtype=torch.float64, requires_grad=True),
                'B' : torch.tensor(param_shape['B'], dtype=torch.float64, requires_grad=True)
            })
    
    def aggregate_information_from_neighbour(self, node, layer_k_node_features, type='mean'):

        #layer_k_node_features =>  nodes x feature_dimension => We need mean vector as 1 x feature_dimension?
        result = layer_k_node_features[node].clone()

        count = 1 # self + neighbour couns
        for neighbour in self.data.adj[node]:
            if neighbour == node:
                continue
            result += layer_k_node_features[neighbour]
            count += 1
        if type=='mean':
            result /= count 

        return result # 1 x F
    

    def feed_forward(self, node_features):
        h = node_features
        for layer in self.layers:
            aggregate_information_all_nodes = torch.stack([self.aggregate_information_from_neighbour(node, h) for node in self.data.nodes],
                                                        dim=0) # Every aggregate is of shape F => so in total if we have N nodes, we have NxF
            # W0, B0 => F x W1.shape[0], F x B1.shape[0]

            # W => layer_last_w_shape_1 x layer_w_shape_1
            # B => layer_last_b_shape_1 x layer_b_shape_1
            h = torch.sigmoid(
                aggregate_information_all_nodes @ layer['W']  + h @ layer['B']
            )

        return h

    def compute_node_features(self):
        pass
        
    def train(self, epochs):
        X = self.data['node_features'] # X_0

    def predict(self, data_point):
        pass


## LLM Generated Driver Code
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import networkx as nx


CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"


def download_cora(data_dir="data/cora"):
    """
    Download and extract the Cora dataset if not already present.
    Returns paths to cora.cites and cora.content.
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
    # optional: remove the intermediate folder
    try:
        os.rmdir(src_dir)
    except OSError:
        pass

    return cites_path, content_path


def load_cora_as_networkx(data_dir="data/cora"):
    """
    Load Cora as a NetworkX DiGraph.

    Node attributes:
        - 'x' : np.ndarray, shape (1433,)  feature vector
        - 'y' : int,                       class index 0..6

    Graph attribute:
        - G.graph['classes'] : list of class names, in index order
    """
    cites_path, content_path = download_cora(data_dir)

    # --- Load edges (citations) ---
    # columns: cited_paper_id (target), citing_paper_id (source)
    citations = pd.read_csv(
        cites_path,
        sep="\t",
        header=None,
        names=["target", "source"],
    )

    # Directed graph: source -> target (paper cites another paper)
    G = nx.from_pandas_edgelist(
        citations,
        source="source",
        target="target",
        create_using=nx.DiGraph(),
    )

    # --- Load node features & labels ---
    # cora.content columns: paper_id, subject, 1433 features (or paper_id, 1433 features, subject,
    # depending on variant; most common is: paper_id, 1433 features, subject) :contentReference[oaicite:0]{index=0}
    # We'll follow the Keras/tutorial ordering: paper_id, 1433 term_i, subject.
    column_names = ["paper_id"] + [f"term_{i}" for i in range(1433)] + ["subject"]
    papers = pd.read_csv(
        content_path,
        sep="\t",
        header=None,
        names=column_names,
    )

    paper_ids = papers["paper_id"].to_numpy()
    features = papers.iloc[:, 1:-1].to_numpy(dtype=np.float32)  # shape: [N, 1433]
    subjects = papers["subject"].to_numpy()                     # shape: [N]

    # Map subject strings to integer labels 0..num_classes-1
    classes, y_int = np.unique(subjects, return_inverse=True)
    label_map = {cls: i for i, cls in enumerate(classes)}

    # Ensure all nodes exist, then attach attrs
    for i, pid in enumerate(paper_ids):
        if pid not in G:
            G.add_node(pid)
        G.nodes[pid]["x"] = features[i]
        G.nodes[pid]["y"] = int(y_int[i])

    # Save class names on the graph
    G.graph["classes"] = classes.tolist()
    G.graph["label_map"] = label_map

    print(
        f"Loaded Cora: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, "
        f"{len(classes)} classes."
    )
    return G


if __name__ == "__main__":
    G = load_cora_as_networkx()
    # Quick sanity check
    node0 = next(iter(G.nodes))
    print("Example node id:", node0)
    print("Feature dim:", G.nodes[node0]["x"].shape)
    print("Label:", G.nodes[node0]["y"])
    print("Classes:", G.graph["classes"])
