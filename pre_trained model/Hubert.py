import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import HubertModel, AutoFeatureExtractor
import librosa
import numpy as np
from tqdm import tqdm
import os
import sys
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建config文件的路径（假设config在父文件夹的父文件夹中）
config_dir = os.path.abspath(os.path.join(current_dir, '..'))
# 将config目录添加到sys.path
sys.path.append(config_dir)
from config import*


class Dataset(torch.utils.data.Dataset):
    """鸟鸣数据集类"""
    def __init__(self, root_dir, split='train', feature_extractor=None):
        self.input_values = []
        self.labels = []
        self.feature_extractor = feature_extractor
        
        data_dir = os.path.join(root_dir, split)
        files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        
        print(f"正在加载{split}数据集...")
        for file in tqdm(files, desc=f"处理{split}集"):
            wav_path = os.path.join(data_dir, file)
            
            try:
                # 加载音频
                audio, sr = librosa.load(wav_path, sr=16000)
                
                # 获取标签（鸟类ID）
                label_key = file[:4]
                if label_key in category_map:
                    label = category_map[label_key]
                    
                    # 直接使用整个音频
                    inputs = self.feature_extractor(
                        audio,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    self.input_values.append(inputs.input_values.squeeze())
                    self.labels.append(label)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
                continue
        
        print(f"{split}集加载完成，共有{len(self.labels)}个样本")
        
        # 打印每个鸟类的样本数量
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print("\n各鸟类样本数量:")
        
        # 获取标签到原始ID的反向映射
        reverse_category_map = {v: k for k, v in category_map.items()}
        for label, count in sorted(label_counts.items()):
            bird_id = reverse_category_map[label]
            print(f"鸟类ID {bird_id}: {count}个样本")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_values': self.input_values[idx],
            'labels': self.labels[idx]
        }

class HubertForBirdClassification(nn.Module):
    """用于鸟鸣分类的HuBERT模型"""
    def __init__(self, num_labels=20, model_name="facebook/hubert-base-ls960"):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hubert.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_values, attention_mask=None):
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        logits = self.classifier(pooled_output)
        
        return logits
@timer
def train_model(model, train_loader, val_loader, test_loader, device, outputdir, num_epochs=50, learning_rate=5e-5, patience=5):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 早停相关变量
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in progress_bar:
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证阶段（每个epoch都进行）
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                input_values = batch['input_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_values)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = np.mean(np.array(val_preds) == np.array(val_labels)) * 100
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Average Val Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
        # 每10个epoch进行一次测试集评估
        if (epoch + 1) % 10 == 0:
            print("\n进行测试集评估...")
            model.eval()
            test_preds = []
            test_labels = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing"):
                    input_values = batch['input_values'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_values)
                    _, predicted = torch.max(outputs.data, 1)
                    test_preds.extend(predicted.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            
            test_accuracy = np.mean(np.array(test_preds) == np.array(test_labels)) * 100
            print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            best_epoch = epoch
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(outputdir,'best_model.pt'))
            print("保存最佳模型")
        else:
            early_stopping_counter += 1
            print(f"验证损失未改善。早停计数器: {early_stopping_counter}/{patience}")
            
            if early_stopping_counter >= patience:
                print(f"\n早停触发！最佳模型出现在epoch {best_epoch + 1}")
                return

def evaluate_model(model, test_loader, device, outputdir, train_time):
    """详细评估模型性能"""
    model.eval()
    
    # 创建Tee对象用于同时输出到控制台和文件
    with open(os.path.join(outputdir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
        tee = Tee(sys.stdout, f)
        
        # 模型分析
        print("\n模型分析:", file=tee)
        print("=" * 50, file=tee)
        
        # 计算总参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f'总参数量: {total_params:,}', file=tee)
        
        # 获取模型大小
        model_size = get_model_size(model, os.path.join(outputdir, 'temp_model.pth'))
        print(f"模型大小: {model_size:.2f} MB", file=tee)
        
        # 打印训练时间
        print(f"训练时间: {train_time:.2f} seconds", file=tee)
        
        # 初始化每个鸟类的统计数据
        class_correct = {i: 0 for i in range(20)}
        class_total = {i: 0 for i in range(20)}
        
        # 用于计算混淆矩阵
        all_preds = []
        all_labels = []
        
        # 进行测试集评估
        print("\n进行测试集评估...")
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_values = batch['input_values'].to(device)
                labels = batch['labels'].to(device)
                
                # 模型输出
                outputs = model(input_values)
                _, predicted = torch.max(outputs.data, 1)
                
                # 更新预测结果和标签
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                
                # 统计每个鸟类的正确数和总数
                for label, pred in zip(labels, predicted):
                    label_idx = label.item()
                    class_total[label_idx] += 1
                    if label_idx == pred.item():
                        class_correct[label_idx] += 1

        # 计算总体准确率
        test_accuracy = np.mean(np.array(test_preds) == np.array(test_labels)) * 100
        print(f'Test Accuracy: {test_accuracy:.2f}%', file=tee)

        # 获取标签到原始ID的反向映射
        reverse_category_map = {v: k for k, v in category_map.items()}

        # 计算并打印每个类别的准确率
        print("\n各鸟类准确率:", file=tee)
        print("鸟类ID\t准确率\t样本数", file=tee)
        print("-" * 30, file=tee)

        total_correct = 0
        total_samples = 0
        
        for class_idx in range(20):
            if class_total[class_idx] > 0:
                class_acc = class_correct[class_idx] / class_total[class_idx] * 100
                bird_id = reverse_category_map[class_idx]
                print(f"{bird_id}\t{class_acc:.2f}%\t{class_total[class_idx]}", file=tee)
                
                total_correct += class_correct[class_idx]
                total_samples += class_total[class_idx]

        # 计算推理时间
        inference_start = time.time()
        with torch.no_grad():
            for batch in test_loader:
                input_values = batch['input_values'].to(device)
                outputs = model(input_values)
        inference_time = time.time() - inference_start
        print(f"测试集推理时间: {inference_time:.2f} seconds", file=tee)

    return test_accuracy, class_correct, class_total, inference_time

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置参数
    root_dir = "./data"
    batch_size = 16
    num_epochs = 1000
    patience = 20
    lr = 5e-5
    # 加载特征提取器
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    
    # 准备数据集
    train_dataset = Dataset(root_dir, 'train', feature_extractor)
    val_dataset = Dataset(root_dir, 'val', feature_extractor)
    test_dataset=Dataset(root_dir, 'test', feature_extractor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=4,  # 使用4个子进程加载数据
        pin_memory=True  # 将数据加载到固定内存中以加速GPU传输
    )
    
    # 创建输出目录
    outputdir='./code/pre_trained model/output/hubert'
    os.makedirs(outputdir, exist_ok=True)
    # 初始化模型
    model = HubertForBirdClassification(num_labels=20)
    
    # 训练模型
    print("开始训练...")
    training_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        outputdir=outputdir,
        num_epochs=num_epochs,
        learning_rate=lr,
        patience=patience
    )
    train_time = training_result[0]
    
    print(f"训练完成！训练时间: {train_time:.2f} seconds")
    
    # 训练完成后，加载最佳模型进行测试
    print("\n开始测试评估...")
    best_model = HubertForBirdClassification(num_labels=20)
    checkpoint = torch.load(os.path.join(outputdir,'best_model.pt'))
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    # 评估模型时传入训练时间
    evaluate_model(best_model, test_loader, device, outputdir, train_time)
    
    print("\n评估结果已保存到: ",os.path.join(outputdir,'evaluation_results.txt'))

if __name__ == "__main__":
    main()