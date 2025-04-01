import networkx as nx  # 导入NetworkX库，用于图数据结构处理
import numpy as np  # 导入NumPy库，用于数值计算
import random  # 导入random库，用于随机数生成
import torch  # 导入PyTorch库，用于深度学习
from sklearn.model_selection import StratifiedKFold  # 导入StratifiedKFold类，用于分层K折交叉验证
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch常用的函数模块
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
# 定义一个图类，用于封装图数据
class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: 一个NetworkX图
            label: 图的标签，一个整数
            node_tags: 节点标签列表，每个元素都是一个整数
            node_features: 节点特征，一个torch浮点张量，通常是标签的一热编码形式，作为神经网络的输入
            edge_mat: 边矩阵，一个torch长整型张量，包含边列表，用于创建torch稀疏张量
            neighbors: 邻居列表（不包括自环）
        '''
        self.label = label  # 图的标签
        self.g = g  # 图本身
        self.node_tags = node_tags  # 节点标签列表
        self.neighbors = []  # 邻居列表
        self.node_features = 0  # 节点特征，默认为0
        self.edge_mat = 0  # 边矩阵，默认为0
        
        self.max_neighbor = 0  # 节点的最大邻居数



# 加载数据集
def load_data(dataset):
    '''
        dataset: 数据集名称
        Standardization: 是否对数据进行归一化处理
    '''
    print('正在加载数据...')
    g_list = []  # 图列表
    label_dict = {}  # 标签字典
    feat_dict = {}  # 特征字典

    # 读取数据集文件
    with open(dataset, 'r') as f:
        n_g = int(f.readline().strip())  # 图的数量
        for i in tqdm(range(n_g), desc="加载图"):  # 使用tqdm显示进度条
            row = f.readline().strip().split()  # 读取一行
            n, l = [int(w) for w in row]  # 节点数和标签
            if l not in label_dict:  # 如果标签不在字典中，则添加新的映射
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()  # 创建一个新的无向图
            node_tags = []  # 节点标签列表
            node_features = []  # 节点特征列表
            n_edges = 0  # 边的数量
            for j in range(n):
                g.add_node(j)  # 添加节点
                row = f.readline().strip().split()  # 读取一行
                tmp = int(row[1]) + 2
                if tmp == len(row):  # 如果没有节点属性
                    row = [int(w) for w in row]  # 转换为整数列表
                    attr = None  # 属性为空
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])  # 节点标签和属性
                    g.add_node(j, att=attr)  # 添加特征到图中
                if row[0] not in feat_dict:  # 如果节点标签不在特征字典中，则添加新的映射
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])  # 添加节点标签,这里的节点标签其实没有意义，因为它只是用来占位的，第一帧就标记为1，第二帧就标记为2，以此类推
                if attr is not None:  # 如果有节点特征，则添加
                    node_features.append(attr) # 添加特征到特征矩阵
                
                n_edges += row[1]  # 累加边数
                for k in range(2, len(row)):  # 添加边
                    g.add_edge(j, row[k])
            
            # 如果有节点特征，则堆叠成矩阵
            if node_features:
                node_features = np.stack(node_features)
            else:
                node_features = None
                
            # 确认图的节点数是否正确
            if not (dataset == "Mine_Graph" or dataset == "Mine_Graph_test"):
                assert len(g) == n
            
            # 将图及其相关信息封装成S2VGraph对象
            g_list.append(S2VGraph(g, l, node_tags))

    # 为每个图添加邻居列表和最大邻居数
    for g in tqdm(g_list, desc="处理邻居列表"):
        g.neighbors = [[] for _ in range(len(g.g))]  # 初始化邻居列表，有多少个节点就创建多少个空列表
        for i, j in g.g.edges():  # 遍历图的边
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = [len(neighbors) for neighbors in g.neighbors]  # 度列表
        g.max_neighbor = max(degree_list)  # 计算最大邻居数
        
        # 映射标签
        g.label = label_dict[g.label]

        # 创建边矩阵
        edges = [list(pair) for pair in g.g.edges()]  # 转换成列表形式
        edges.extend([[i, j] for j, i in edges])  # 包含反向边
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)  # 转换成PyTorch长整型张量

    # 提取唯一的标签集合
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))  # 合并标签集合
    
    tagset = list(tagset)  # 转换成列表
    tag2index = {tagset[i]: i for i in range(len(tagset))}  # 标签到索引的映射

    # 为每个图添加节点特征
    for g in tqdm(g_list, desc="处理节点特征"):
        if 'att' in g.g.nodes[0]:  # 确保至少有一个节点有 'att' 属性
            g.node_features = torch.zeros(len(g.node_tags), len(g.g.nodes[0]['att']))  # 初始化节点特征
            for i in range(len(g.node_tags)):
                g.node_features[i] = torch.FloatTensor(g.g.nodes[i]['att'])  # 赋值节点特征
        else:
            # 如果没有节点属性，则可以初始化一个默认的特征向量
            g.node_features = torch.zeros(len(g.node_tags), 1)  # 假设只有一个特征维度


    # 为每个图添加扩展节点特征
    for g in tqdm(g_list, desc="处理扩展节点特征"):
        g.node_features2 = torch.zeros(len(g.node_tags), 2 * len(g.g.nodes[0]['att']))  # 初始化扩展节点特征
        for i in range(len(g.node_tags)):
            if i == 0:
                g.node_features2[i] = torch.cat([g.node_features[i], g.node_features[i]])
            else:
                g.node_features2[i] = torch.cat([g.node_features[i], g.node_features[i] - g.node_features[i - 1]])

    print('# 类别数: %d' % len(label_dict))
    print('# 图节点数: %d' % len(tagset))
    print('# 图个数: %d' % len(g_list))

    return g_list, len(label_dict)  # 返回图列表和类别数量

# 分割数据集
def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx 必须是从 0 到 9。"
    skf = StratifiedKFold(n_splits=fold_idx, shuffle=True, random_state=seed)  # 分层K折交叉验证

    labels = [graph.label for graph in graph_list]  # 获取所有图的标签
    idx_list = []  # 索引列表
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)  # 收集每个折叠的索引
    
    train_portions = []  # 训练集列表
    test_portions = []  # 测试集列表
    for j in range(fold_idx):
        train_idx, test_idx = idx_list[j]  # 获取训练和测试索引
        train_portions.append([graph_list[i] for i in train_idx])  # 收集训练集
        test_portions.append([graph_list[i] for i in test_idx])  # 收集测试集

    return train_portions, test_portions  # 返回训练集和测试集