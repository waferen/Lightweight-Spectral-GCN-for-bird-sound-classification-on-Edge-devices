import torch
import torch.nn as nn
from thop import profile
from efficientnet_pytorch import EfficientNet
from GCN.graphcnn import Graph_CNN_ortega

# 定义类别数量和其他参数
num_classes = 20
num_nodes = 200  # 假设每个图有100个节点
input_dim = 34   # 假设节点特征维度为64
device = torch.device('cpu')

# EfficientNet模型定义
def create_efficientnet(num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

# 分析模型函数
def analyze_model(model, model_name, input_data):
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{model_name} 总参数量: {total_params:,}')

    # 计算FLOPs
    flops, _ = profile(model, inputs=input_data)
    print(f'{model_name} FLOPs: {flops:,}')

    # 计算模型大小（MB）
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f'{model_name} 模型大小: {model_size:.2f} MB')
    print('-' * 50)

# 分析EfficientNet模型
efficientnet = create_efficientnet(num_classes)
efficientnet.load_state_dict(torch.load('my_efficientnetb0_model_best.pth', map_location=torch.device('cpu')))
efficientnet_input = torch.randn(1, 3, 224, 224)
analyze_model(efficientnet, 'EfficientNet', (efficientnet_input,))

# 分析GCN模型
params = {
    'num_layers': 3,
    'hidden_dim': 128,
    'final_dropout': 0.5,
    'graph_pooling_type': 'mean'
}

# 创建一个虚拟的邻接矩阵A
A = torch.rand(num_nodes, num_nodes)

gcn_model = Graph_CNN_ortega(params['num_layers'], input_dim,
                             params['hidden_dim'], num_classes, params['final_dropout'], 
                             params['graph_pooling_type'], device, A).to(device)

gcn_model.load_state_dict(torch.load('checkpoint.pt', map_location=torch.device('cpu')))

# 准备GCN模型的输入
gcn_input = torch.randn(1, num_nodes, input_dim)  # 批次大小为1的输入

analyze_model(gcn_model, 'GCN', (gcn_input,))