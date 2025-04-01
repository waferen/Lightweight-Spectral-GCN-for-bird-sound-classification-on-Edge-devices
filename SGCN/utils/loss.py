import torch
import torch.nn as nn  # 神经网络模块
# 定义损失函数
num_classes = 20
weights = torch.ones(num_classes).cuda()  
criterion = nn.CrossEntropyLoss(weight=weights)
