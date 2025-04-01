import torch  # PyTorch 主要库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数
import torch.optim as optim  # 优化器模块
import numpy as np  # 数值计算
import time  # 时间操作
import networkx as nx  # 图数据处理
from tqdm import tqdm  # 进度条显示
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import sys
import os
from utils.visualize import visualize_results

from utils.earlystopping import EarlyStopping
from utils.radam import RAdam,AdamW
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建config文件的路径（假设config在父文件夹的父文件夹中）
config_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 将config目录添加到sys.path
sys.path.append(config_dir)
from config import*
# 定义损失函数
num_classes = 20
weights = torch.ones(num_classes).cuda()  
criterion = nn.CrossEntropyLoss(weight=weights)

def count_parameters(model):
    total_params = 0
    
    # 计算 GCNs 中的参数
    for gcn in model.GCNs:
        # MLP 中的参数
        for linear in gcn.MLP.linears:
            total_params += linear.weight.numel() + linear.bias.numel()
        
        # BatchNorm 层的参数（如果有的话）
        if hasattr(gcn.MLP, 'batch_norms'):
            for bn in gcn.MLP.batch_norms:
                total_params += 2 * bn.num_features  # weight 和 bias
    
    # 计算分类器中的参数
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            total_params += layer.weight.numel() + layer.bias.numel()
        elif isinstance(layer, nn.PReLU):
            total_params += layer.num_parameters
    
    return total_params


def train_one_step(args, model, device, train_graphs, optimizer, epoch, A):
    # 设置模型为训练模式
    model.train()
    

    # 总迭代次数
    total_iters = args['iters_per_epoch']
    # 创建进度条
    pbar = tqdm(range(total_iters), unit='batch')

    # 初始化损失累积
    loss_accum = 0
    # 开始计时
    start = time.time()
    for pos in pbar:
        # 随机选择一部分索引
        selected_idx = np.random.permutation(len(train_graphs))[:args['batch_size']]

        # 根据索引选取批量图数据
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        # 模型前向传播
        output = model(batch_graph)

        # 获取标签
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        
        # 计算损失
        loss = criterion(output, labels)

        # 反向传播
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 记录损失
        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # 更新进度条描述
        pbar.set_description('epoch: %d' % epoch)
    # 结束计时
    end = time.time()
    print('epoch time: %f' % (end - start))

    # 计算平均损失
    average_loss = loss_accum / total_iters
    print("loss training: %f" % average_loss)

    return average_loss

# 以mini-batch方式传递数据给模型，以避免内存溢出（不进行反向传播）
def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()  # 设置模型为评估模式
    output = []  # 存储输出
    idx = np.arange(len(graphs))  # 创建索引数组
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        # 将批量图数据输入模型
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)  # 拼接所有批次的输出
@timer
def test(args, model, device, train_graphs, test_graphs, num_class):
    model.eval()  # 设置模型为评估模式

    def calculate_accuracy_per_class(pred, labels, num_class):
        cm = confusion_matrix(labels.cpu(), pred.cpu(), labels=range(num_class))
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        return class_accuracies

    # # 测试训练集
    # output_train = pass_data_iteratively(model, train_graphs)
    # pred_train_ = output_train.max(1, keepdim=True)[1]
    # labels_train = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    # correct_train = pred_train_.eq(labels_train.view_as(pred_train_)).sum().cpu().item()
    # acc_train = correct_train / float(len(train_graphs))

    ## 计算每个类别的准确率（训练集）
    # acc_train_per_class = calculate_accuracy_per_class(pred_train_, labels_train, num_class)

    # 测试测试集
    output_test = pass_data_iteratively(model, test_graphs)
    pred_test = output_test.max(1, keepdim=True)[1]
    labels_test = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct_test = pred_test.eq(labels_test.view_as(pred_test)).sum().cpu().item()
    acc_test = correct_test / float(len(test_graphs))

    # 计算每个类别的准确率（测试集）
    acc_test_per_class = calculate_accuracy_per_class(pred_test, labels_test, num_class)

    return ( acc_test, acc_test_per_class.tolist())

@timer
def train(model, params, device, train_graphs, test_graphs, val_graphs, num_classes, A):
    train_data = train_graphs
    test_data = test_graphs
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    # optimizer = AdamW(model.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']),
    #                     weight_decay=params['weight_decay'])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 初始化提前停止机制
    early_stopping = EarlyStopping(patience=params['patience'], verbose=True)
    
    loss_list = []
    accuracy_test = []
    best_val_acc = 0  # 记录最佳验证集准确率

    # 训练循环
    for epoch in range(1, params['epochs'] + 1):
        # 训练模型
        avg_loss = train_one_step(params, model, device, train_graphs, optimizer, epoch, A)
        print(f'EPOCH: {epoch}, Loss: {avg_loss:.4f}')
        loss_list.append(avg_loss)
        scheduler.step()  # 调整学习率
        
        # 验证集评估
        if epoch > 1:
            with torch.no_grad():
                val_out = pass_data_iteratively(model, val_graphs)
                val_labels = torch.LongTensor([graph.label for graph in val_graphs]).to(device)
                val_loss = criterion(val_out, val_labels)
                val_loss = np.average(val_loss.detach().cpu().numpy())

                # 计算验证集准确率
                val_pred = torch.argmax(val_out, dim=1)
                val_acc = (val_pred == val_labels).float().mean().item()
            
            # 评估并打印结果
            if (epoch <= 100 and epoch % 10 == 0) or (epoch > 100 and epoch % 20 == 0):
                # 调用 test 函数获取准确率等信息
                _, (acc_test, acc_test_per_class) = test(params, model, device, train_data, test_data, num_classes)

                # 记录整体准确率
                accuracy_test.append(acc_test)

                # 打印当前epoch的整体准确率
                print(f'EPOCH: {epoch}, acc_test: {acc_test:.4f}')

                print("测试集每个类别的准确率:")
                for class_idx, acc in enumerate(acc_test_per_class):
                    print(f"类别 {class_idx}: {acc:.4f}")
            
            # 检查提前停止
            early_stopping(val_acc, model)

            # 如果提前停止，则跳出训练
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 根据验证集准确率决定是否保存模型
            if val_acc > best_val_acc:
                print(f"验证集准确率提高，从 {best_val_acc:.4f} 提高到 {val_acc:.4f}，保存模型...")
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(params['output_path'],'best_model.pt'))  # 保存模型

