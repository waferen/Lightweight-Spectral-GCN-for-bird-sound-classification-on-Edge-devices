import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 假设你已经有 GCNModel, AudioGraphDataset, get_audio_files_from_directory 的实现

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # 设置模型为训练模式
    train_loss = 0
    correct = 0
    total = 0
    for data in tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=100):  # 添加进度条
        data = data.to(device)  # 将整个数据对象移动到指定设备
        inputs, labels = data.x, data.y

        optimizer.zero_grad()  # 清零梯度
        outputs = model(data)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss += loss.item()  # 累积损失
        _, predicted = torch.max(outputs, 1)  # 获取预测结果
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 计算正确的预测数

    avg_loss = train_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():  # 在测试时不需要计算梯度
        for data in tqdm(test_loader, desc="Testing", ncols=100):  # 添加进度条
            data = data.to(device)  # 将整个数据对象移动到指定设备
            inputs, labels = data.x, data.y
            outputs = model(data)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            test_loss += loss.item()  # 累积损失
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 计算正确的预测数

            all_labels.append(labels.cpu())
            all_predictions.append(predicted.cpu())

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    # 计算每个类别的分类准确率
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    class_accuracies = []
    for class_idx in range(20):  # 假设类别数为20
        correct_class = (all_predictions == class_idx) & (all_labels == class_idx)
        class_accuracy = correct_class.sum().item() / (all_labels == class_idx).sum().item() * 100
        class_accuracies.append(class_accuracy)
        
    return avg_loss, accuracy, class_accuracies

