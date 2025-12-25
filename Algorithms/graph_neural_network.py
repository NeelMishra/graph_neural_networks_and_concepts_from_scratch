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

class GraphSage:
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
                'W' : torch.zeros(param_shape['W'], dtype=torch.float64, requires_grad=True)
            })
    
    def aggregate_information_from_neighbour(self, node, layer_k_node_features, op_type='mean'):

        #layer_k_node_features =>  nodes x feature_dimension => We need mean vector as 1 x feature_dimension?
        result = torch.zeros_like(layer_k_node_features[node])

        count = 0 # neighbour counts
        for neighbour in self.data.adj[node]:
            if neighbour == node:
                continue
            result += layer_k_node_features[neighbour]
            count += 1
        if count == 0:
            return result
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

            z = torch.cat([aggregate_information_all_nodes, h], dim=1)
            if idx != len(self.layers) - 1:
                h = torch.sigmoid(
                    z @ layer['W']
                )
            else:
                h = z @ layer['W']
        return h

    def compute_node_features(self):

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
    

class GCN:
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
    
    def aggregate_information_from_neighbour(self, node, layer_k_node_features):

        #layer_k_node_features =>  nodes x feature_dimension => We need mean vector as 1 x feature_dimension?
        result = layer_k_node_features[node].clone() * (1/ (self.data.out_degree[node] + 1) )

        count = 1 # self + neighbour couns
        for neighbour in self.data.adj[node]:
            if neighbour == node:
                continue
            du = self.data.out_degree[node] + 1
            dv = self.data.out_degree[neighbour] + 1
            if du and dv:
                result += ( (1/np.sqrt(du)) * (1/np.sqrt(dv) ) * layer_k_node_features[neighbour] )


        return result # F
    

    def feed_forward(self, node_features):
        h = node_features
        for idx, layer in enumerate(self.layers):
            aggregate_information_all_nodes = torch.stack([self.aggregate_information_from_neighbour(node, h) for node in self.data.nodes],
                                                        dim=0) # Every aggregate is of shape F => so in total if we have N nodes, we have NxF
            # W0, B0 => F x W1.shape[0], F x B1.shape[0]

            # W => layer_last_w_shape_1 x layer_w_shape_1
            # B => 1 x out_dim
            if idx != len(self.layers) - 1:
                h = torch.sigmoid(
                    aggregate_information_all_nodes @ layer['W']  + layer['B']
                )
            else:
                h = aggregate_information_all_nodes @ layer['W'] +  layer['B']
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
