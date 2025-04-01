import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,global_add_pool,global_max_pool
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.linalg import eig
import sys
from torch.nn import init
from scipy.linalg import fractional_matrix_power
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data
from utils.mlp import MLP
import math
# 归一化有向图邻接矩阵的函数
def normalize_digraph(A):
    """
    归一化有向图的邻接矩阵
    :param A: 邻接矩阵，形状为 (num_nodes, num_nodes)
    :return: 归一化后的邻接矩阵
    """
    Dl = np.sum(A, 0)  # 每一列的度
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))  # 生成度矩阵
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)  # 归一化度矩阵
    AD = np.dot(A, Dn)  # 归一化后的邻接矩阵
    return AD

# 将密集张量转换为稀疏格式
def to_sparse(x):
    """
    将密集张量转换为稀疏格式
    :param x: 输入的密集张量
    :return: 稀疏张量
    """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)  # 获取非零元素的索引
    if len(indices.shape) == 0:  # 如果所有元素都是零
        return sparse_tensortype(*x.shape)
    indices = indices.t()  # 转置
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]  # 获取对应的值
    return sparse_tensortype(indices, values, x.size())
# 计算图的度矩阵
def Comp_degree(A):
    # Compute in-degree and out-degree for sparse adjacency matrix A
    in_degree = A.sum(dim=1)  # Row sum gives in-degree
    out_degree = A.sum(dim=0)  # Column sum gives out-degree

    # For the degree matrix, ensure it's in dense format before using operations like diagonal
    in_degree = in_degree.to_dense()  # Convert to dense to use diagonal
    out_degree = out_degree.to_dense()

    # Create degree matrix
    diag = torch.diag(in_degree + out_degree)
    
    # Use dense operations like diagonal extraction safely here
    degree_matrix = diag*in_degree + diag*out_degree - torch.diagflat(torch.diagonal(A.to_dense()))

    return degree_matrix
def Comp_laplacian(A):
    """
    计算环形图的拉普拉斯矩阵 L = D - A
    :param A: 邻接矩阵
    :return: 拉普拉斯矩阵
    """
    device = A.device
    n = A.size(0)

    # 对于环形图，度矩阵 D 是单位矩阵（每个节点的度为1）
    degree_matrix = torch.eye(n, device=device)

    # 邻接矩阵 A 对应的是环形单向连接
    # 直接构造一个稀疏的邻接矩阵
    adj = torch.zeros(n, n, device=device)
    for i in range(n-1):
        adj[i, i+1] = 1  # 单向连接
    adj[n-1, 0] = 1  # 最后一个节点与第一个节点相连，形成环形结构
    
    # 拉普拉斯矩阵 L = D - A
    laplacian = degree_matrix - adj
    return laplacian
def Comp_normalized_laplacian(A):
    """
    计算标准化拉普拉斯矩阵 L_norm = D^(-1/2) * (D - A) * D^(-1/2)
    :param A: 邻接矩阵
    :return: 标准化拉普拉斯矩阵
    """
    device = A.device
    n = A.size(0)

    # 计算度矩阵 D（每个节点的度为 1）
    degree_matrix = torch.eye(n, device=device)

    # 计算标准化的拉普拉斯矩阵
    laplacian = Comp_laplacian(A)  # 使用上述计算的 L = D - A
    frac_degree = torch.sqrt(torch.inverse(degree_matrix))  # D^(-1/2)
    
    # 标准化拉普拉斯矩阵
    norm_laplacian = frac_degree @ laplacian @ frac_degree
    return norm_laplacian


def compute_laplacian_eigen(A):
    """
    计算环形图的拉普拉斯矩阵特征值和特征向量
    :param A: 邻接矩阵
    :return: 特征值和特征向量
    """
    device = A.device
    n = A.size(0)  # 节点数量

    # 计算特征值：λ_k = 2 * (1 - cos(2πk / n)) for k = 0, 1, ..., n-1
    lambdas = [2 * (1 - math.cos(2 * math.pi * k / n)) for k in range(n)]
    lambdas = torch.tensor(lambdas, device=device)

    # 计算特征向量：u_k(i) = (1/sqrt(n)) * e^(2πki/n) for i = 0, 1, ..., n-1
    eigenvectors = torch.zeros(n, n, dtype=torch.complex64, device=device)
    for k in range(n):
        for i in range(n):
            # 确保所有计算的常量都是Tensor类型
            real_part = (1 / math.sqrt(n)) * torch.cos(torch.tensor(2 * math.pi * k * i / n, device=device))
            imag_part = (1 / math.sqrt(n)) * torch.sin(torch.tensor(2 * math.pi * k * i / n, device=device))
            eigenvectors[k, i] = torch.complex(real_part, imag_part)
    
    return lambdas, eigenvectors.real  # 返回特征值和特征向量的实部
# 图卷积层定义
class GraphConv_Ortega(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dim):
        super(GraphConv_Ortega, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)

        # 初始化MLP网络
        for i in range(num_layers):
            init.xavier_uniform_(self.MLP.linears[i].weight)
            init.constant_(self.MLP.linears[i].bias, 0)

    def forward(self, features, A):
        """
        特征输入为 (num_nodes, num_features)，
        邻接矩阵 A 为 (num_nodes, num_nodes)。
        """
        n, d = features.shape  # 获取节点数和特征维度
        assert (d == self.in_dim)  # 确保输入特征维度正确

        # 计算度矩阵和拉普拉斯矩阵
        A_norm = A  # 邻接矩阵 A（稀疏矩阵）
        
        # 计算度矩阵
        deg_mat = Comp_degree(A_norm)  # 假设这里返回的是稀疏矩阵
        
        # 使用fractional_matrix_power计算度矩阵的 -0.5 次幂
        deg_mat_cpu = deg_mat.detach().cpu().to_dense().numpy()  # 将稀疏矩阵转为稠密矩阵计算
        frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat_cpu, -0.5)).cuda()

        # 计算拉普拉斯矩阵 L = D^(-0.5) * (D - A) * D^(-0.5)
        Laplacian = deg_mat - A_norm
        Laplacian_norm = frac_degree.matmul(Laplacian.matmul(frac_degree))  # 标准化拉普拉斯矩阵

        # 由于稀疏矩阵的特征值分解通常需要将稀疏矩阵转为稠密矩阵，这里采用标准的eig
        landa, U = torch.linalg.eig(Laplacian_norm)  # 计算特征值和特征向量
        U_real = U.real  # 获取特征向量的实部

        repeated_U_t = U_real.t()  # 特征向量的转置
        repeated_U = U_real  # 特征向量

        # 聚合节点特征
        agg_feats = torch.sparse.mm(A_norm, features)  # 使用稀疏矩阵乘法
        agg_feats = torch.matmul(repeated_U_t, agg_feats)  # 特征向量的转置与聚合特征做点乘

        # 通过MLP进行进一步的特征映射
        out = self.MLP(agg_feats.view(-1, d)).view(n, self.out_dim)
        
        # 最终通过特征向量矩阵做点乘得到输出
        out = torch.matmul(repeated_U, out)
        return out

# 定义图卷积神经网络模型
class GCNModel(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, final_dropout,
                 graph_pooling_type, device):
        """
        定义图卷积神经网络模型
        :param num_layers: GCN层的数量
        :param input_dim: 输入特征的维度
        :param hidden_dim: 隐藏层的维度
        :param output_dim: 输出的类别数
        :param final_dropout: 最后全连接层的dropout概率
        :param graph_pooling_type: 池化方式 ('mean', 'max', 'sum')
        :param device: 使用的设备 ('cuda' 或 'cpu')
        """
        super(GCNModel, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 定义多个GCN层
        self.GCNs = torch.nn.ModuleList()
        self.GCNs.append(GraphConv_Ortega(self.input_dim, self.hidden_dim,self.num_layers,self.hidden_dim))
        for i in range(self.num_layers-1):
            self.GCNs.append(GraphConv_Ortega(self.hidden_dim, self.hidden_dim,self.num_layers,self.hidden_dim))

        # 定义分类器
        self.classifier = nn.Sequential(
                            nn.Linear(self.hidden_dim, 128),
                            nn.Dropout(p=self.final_dropout),
                            nn.PReLU(128),
                            nn.Linear(128, output_dim))

    def forward(self, data: Data):
        """
        :param data: torch_geometric.data.Data object
        """
        # 获取 Data 对象中的节点特征和邻接矩阵
        x = data.x  # 节点特征
        edge_index = data.edge_index  # 边连接信息

        # 构建稀疏邻接矩阵
        edge_index = edge_index.to(self.device)
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), device=self.device),
                                      (x.size(0), x.size(0)), dtype=torch.float32).to(self.device)

        adj = F.relu(adj)

        # 图卷积操作
        h = x
        for layer in self.GCNs:
            h = F.relu(layer(h, adj))

        # 图卷积操作
        h = x
        for layer in self.GCNs:
            h = F.relu(layer(h, adj))
        # 池化操作
        if self.graph_pooling_type == 'mean':
            pooled = global_mean_pool(h, data.batch)
        elif self.graph_pooling_type == 'max':
            pooled = global_max_pool(h, data.batch)
        elif self.graph_pooling_type == 'sum':
            pooled = global_add_pool(h, data.batch)
        print(pooled.shape) 
        # 分类
        score = self.classifier(pooled)
        return score


class Base_GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(Base_GCNModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # 定义GCN层
        self.convs = torch.nn.ModuleList()  # 用来存储多个 GCNConv 层
        self.convs.append(GCNConv(input_dim, hidden_dim))  # 第一层
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))  # 后续层
        
        # 全连接层进行分类
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 多层 GCN + 激活函数 + Dropout
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))  # 每层 GCN
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        
        # 池化操作，汇聚所有节点特征
        x = global_mean_pool(x, data.batch)  # 使用全局平均池化

        # 全连接层进行分类
        x = F.relu(self.fc1(x))  # 激活函数
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        x = self.fc2(x)  # 输出层

        return F.log_softmax(x, dim=-1)  # 使用log softmax进行分类