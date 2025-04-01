# mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

### MLP with linear output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: 神经网络层数（不包括输入层）。如果 num_layers=1，则简化为线性模型。
            input_dim: 输入特征的维度。
            hidden_dim: 所有隐藏层的单元维度。
            output_dim: 预测的类别数。
        '''

        super(MLP, self).__init__()

        # 默认为线性模型
        self.linear_or_not = True 
        self.num_layers = num_layers

        # 检查层数是否为正数
        if num_layers < 1:
            raise ValueError("神经网络的层数应该为正数！")
        elif num_layers == 1:
            # 线性模型
            self.linear = nn.Linear(input_dim, output_dim, bias=True)
        else:
            # 多层模型
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()  # 用于存储每一层的线性变换
            self.batch_norms = torch.nn.ModuleList()  # 用于存储每一层的批量归一化
            
            # 第一层从输入维度到隐藏维度
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            
            # 中间层
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            
            # 最后一层从隐藏维度到输出维度
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            
            # 添加批量归一化层
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # 如果是线性模型
            return self.linear(x)
        else:
            # 如果是多层感知机
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))  # 通过线性层、批量归一化层和激活函数
            return self.linears[self.num_layers - 1](h)  # 最终的输出层，没有激活函数
print('complete')