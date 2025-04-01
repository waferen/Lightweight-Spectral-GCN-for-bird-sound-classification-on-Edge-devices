import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from numpy.linalg import eig
import sys
from torch.nn import init
from scipy.linalg import fractional_matrix_power

from GCN.mlp import MLP
sys.path.append("models/")  # 添加模型目录到路径中


# 归一化有向图
def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

# 将密集张量转换为稀疏格式
def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # 如果所有元素都是零
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

# 计算图的度矩阵
def Comp_degree(A):
    device = A.device
    out_degree = torch.sum(A, dim=0).to(device)
    in_degree = torch.sum(A, dim=1).to(device)
    diag = torch.eye(A.size()[0], device=device)
    degree_matrix = diag*in_degree + diag*out_degree - torch.diagflat(torch.diagonal(A))
    return degree_matrix

# 图卷积层定义
class GraphConv_Ortega(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128):
        super(GraphConv_Ortega, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)

        for i in range(num_layers):
            init.xavier_uniform_(self.MLP.linears[i].weight)
            init.constant_(self.MLP.linears[i].bias, 0)

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        if(len(A.shape) == 2):
            A_norm = A
            deg_mat = Comp_degree(A_norm)
            frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.detach().cpu(), -0.5)).cuda()
            Laplacian = deg_mat - A_norm
            Laplacian_norm = frac_degree.matmul(Laplacian.matmul(frac_degree))
            landa, U = torch.linalg.eig(Laplacian_norm)

            # 只保留实数部分
            U_real = U.real

            repeated_U_t = U_real.t().repeat(b, 1, 1)
            repeated_U = U_real.repeat(b, 1, 1)
        else:
            repeated_U_t = []
            repeated_U = []
            for i in range(A.shape[0]):
                A_norm = A[i]
                deg_mat = Comp_degree(A_norm)
                frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.detach().cpu(), -0.5)).cuda()
                Laplacian = deg_mat - A_norm
                Laplacian_norm = frac_degree.matmul(Laplacian.matmul(frac_degree))
                landa, U = torch.linalg.eig(Laplacian_norm)

                # 只保留实数部分
                U_real = U.real

                repeated_U_t.append(U_real.t().view(1, U_real.shape[0], U_real.shape[1]))
                repeated_U.append(U_real.view(1, U_real.shape[0], U_real.shape[1]))
            repeated_U_t = torch.cat(repeated_U_t)
            repeated_U = torch.cat(repeated_U)

        agg_feats = torch.bmm(repeated_U_t, features)
        out = self.MLP(agg_feats.view(-1, d)).view(b, -1, self.out_dim)
        out = torch.bmm(repeated_U, out)
        return out

# 定义图卷积神经网络模型
class Graph_CNN_ortega(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, final_dropout,
                 graph_pooling_type, device, adj):
        super(Graph_CNN_ortega, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Adj = adj

        self.GCNs = torch.nn.ModuleList()
        self.GCNs.append(GraphConv_Ortega(self.input_dim, self.hidden_dim))
        for i in range(self.num_layers-1):
            self.GCNs.append(GraphConv_Ortega(self.hidden_dim, self.hidden_dim))

        self.classifier = nn.Sequential(
                            nn.Linear(self.hidden_dim, 128),
                            nn.Dropout(p=self.final_dropout),
                            nn.PReLU(128),
                            nn.Linear(128, output_dim))

    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features.view(1,-1,self.input_dim) for graph in batch_graph], 0).to(self.device)
        A = F.relu(self.Adj)

        h = X_concat
        for layer in self.GCNs:
            h = F.relu(layer(h, A))

        if(self.graph_pooling_type == 'mean'):
            pooled = torch.mean(h, dim=1)
        elif (self.graph_pooling_type == 'max'):
            pooled = torch.max(h, dim=1)[0]
        elif (self.graph_pooling_type == 'sum'):
            pooled = torch.sum(h, dim=1)

        score = self.classifier(pooled)

        return score
print('complete')