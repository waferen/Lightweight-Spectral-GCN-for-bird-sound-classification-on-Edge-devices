
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, degree
from torch_geometric.nn import global_mean_pool

class SpectralGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes):
        super(SpectralGCNLayer, self).__init__()
        self.output_dim = output_dim
        self.num_nodes = num_nodes

        # Learnable weight matrix for spectral convolution
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, x, edge_index):
        # Compute normalized Laplacian matrix in dense format
        laplacian = self.compute_laplacian(edge_index)

        # Spectral convolution: L * X * W
        x = torch.matmul(laplacian, x)  # Matrix multiplication (dense format)
        x = torch.matmul(x, self.weight)  # Apply weight matrix
        return x

    def compute_laplacian(self, edge_index):
        # Number of nodes in the graph
        num_nodes = self.num_nodes
        
        # Get the degree of each node
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float)

        # Create the adjacency matrix (symmetric normalized)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        adj[row, col] = 1
        adj[col, row] = 1  # Ensure symmetry for undirected graph

        # Normalize the adjacency matrix by degree matrix
        D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        laplacian = torch.eye(num_nodes) - torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)

        return laplacian

class Base_SpectralGCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(Base_SpectralGCNModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Define spectral GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(SpectralGCNLayer(input_dim, hidden_dim, num_nodes=100))  # First layer, assume 100 nodes
        for _ in range(num_layers - 1):
            self.convs.append(SpectralGCNLayer(hidden_dim, hidden_dim, num_nodes=100))  # Subsequent layers
        
        # Fully connected layers between GCN layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Final classification layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print('============================',edge_index.shape) 

        # Pass through spectral GCN + fully connected layers + activation + dropout
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)  # Apply spectral GCN
            x = F.relu(self.fc_layers[i](x))  # Fully connected layer
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        # Pooling over all nodes
        x = global_mean_pool(x, data.batch)

        # Fully connected layers for classification
        x = F.relu(self.fc1(x))  # Activation
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        x = self.fc2(x)  # Output layer

        return F.log_softmax(x, dim=-1)  # Log softmax for classification
