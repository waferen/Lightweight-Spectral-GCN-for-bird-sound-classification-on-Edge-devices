import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建config文件的路径（假设config在父文件夹的父文件夹中）
config_dir = os.path.abspath(os.path.join(current_dir, '..'))
# 将config目录添加到sys.path
sys.path.append(config_dir)
from config import*

@timer
def process_audio(input_path, output_path):
    # 创建保存图像的目录
    os.makedirs(output_path, exist_ok=True)
    if 'train' in input_path:
        dataset_type = 'train'
    elif 'val' in input_path:
        dataset_type = 'val'
    else:
        dataset_type = 'test'
    # 获取所有wav文件
    wav_files = [f for f in os.listdir(input_path) if f.endswith('.wav')]

    if data_percent < 1.0 and dataset_type == 'train':
        # 提取类别标签
        labels = [file[:4] for file in wav_files]  # 假设类别在文件名的前四个字符
        wav_files, _ = train_test_split(wav_files, train_size=data_percent, random_state=seed, stratify=labels)
    # 使用tqdm创建进度条
    for file in tqdm(wav_files, desc="处理音频文件"):
        category = file[:4]  # 获取类别（前4个字符）
        label = category_map.get(category, -1)
        
        wav_path = os.path.join(input_path, file)
        
        # 使用 Librosa 加载音频文件
        y, sr = librosa.load(wav_path, sr=sample_rate)
        
        # 计算梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_bins, fmin=fmin, fmax=fmax)
        
        # 转换为分贝单位
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 归一化
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # 生成图像
        plt.figure(figsize=(10, 10))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax)
        plt.axis('off')
        plt.title('Mel Spectrogram')
        plt.tight_layout(pad=0)
        
        # 保存图像
        image_filename = os.path.join(output_path, f"{file[:-4]}.png")
        plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 读取保存的图像并调整大小
        img = cv2.imread(image_filename)
        img = cv2.resize(img, (224, 224))  # 调整为 EfficientNet-B0 的输入尺寸
        
        # 保存调整后的图像
        cv2.imwrite(image_filename, img)



# 设置随机种子以保证可重复性
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 读取图像文件路径和标签
def load_image_paths_and_labels(data_dir, dataset):
    image_paths = []
    labels = []
    dataset_dir = os.path.join(data_dir, dataset)
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(dataset_dir, filename)
            # 从文件名中提取类别
            category = filename[:4]
            if category in category_map:
                image_paths.append(image_path)
                labels.append(category_map[category])
    return image_paths, labels

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 创建数据加载器
def create_data_loaders(data_dir, batch_size=16, num_workers=4):
    train_paths, train_labels = load_image_paths_and_labels(data_dir, 'train')
    val_paths, val_labels = load_image_paths_and_labels(data_dir, 'val')  # 加载验证集
    test_paths, test_labels = load_image_paths_and_labels(data_dir, 'test')

    train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transform)
    test_dataset = CustomDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

# 加载预训练的EfficientNetB0模型并修改最后一层
def create_model(num_classes):
    model = EfficientNet.from_name('efficientnet-b0')
    # 修改模型的最后一层
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

# 训练函数
@timer
def train_model(model, dataloaders, criterion, optimizer, scheduler, save_dir, num_epochs=1000, patience=20, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 创建模型保存目录 
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model.to(device)

    best_model_wts = model.state_dict()  # 初始模型权重保存

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = model.state_dict()  # 更新最佳模型权重
                    # 更新保存路径
                    torch.save(best_model_wts, os.path.join(save_dir, 'model_best.pth'))

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        # No return, simply break
                        return

    # 直接在函数外部加载最佳模型
    torch.save(best_model_wts, os.path.join(save_dir, 'model_best.pth'))  # 最终保存最佳权重


# 评估函数
@timer
def evaluate_model(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

# 主函数
import torch

# 主函数
def main():
    # 主程序
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查GPU并设置设备

    # 数据根目录和输出目录
    data_root = "data"  # 数据根目录
    output_root = "Mel_spectrogram_images"  # 输出预处理数据根目录
    save_dir = 'code/EfficientNet/output'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'experiment_results.txt'), 'w', encoding='utf-8') as f:
        # 创建一个Tee对象，它会同时写入到控制台和文件
        original_stdout = sys.stdout
        tee = Tee(sys.stdout, f)

        # 处理训练、测试和验证集的数据并计算数据处理时间
        data_processing_time, _ = process_audio(os.path.join(data_root, 'train'), os.path.join(output_root, 'train'))
        print(f"Data processing time for train dataset: {data_processing_time:.2f} seconds", file=tee)
        
        data_processing_time, _ = process_audio(os.path.join(data_root, 'val'), os.path.join(output_root, 'val'))
        print(f"Data processing time for validation dataset: {data_processing_time:.2f} seconds", file=tee)
        
        data_processing_time, _ = process_audio(os.path.join(data_root, 'test'), os.path.join(output_root, 'test'))
        print(f"Data processing time for test dataset: {data_processing_time:.2f} seconds", file=tee)

        print("Mel_Spectrogram extraction and saving complete for both train and test datasets.")

        # 创建数据加载器
        dataloaders = create_data_loaders(output_root)

        # 模型定义和配置
        num_classes = len(category_map)
        model = create_model(num_classes).to(device)  # 将模型移动到设备

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=0.0001)

        # 训练模型并计算训练时间
        train_time,_ = train_model(model, dataloaders, criterion, optimizer, scheduler, save_dir, num_epochs=1000, patience=10)
        # 在外部加载最佳模型
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model_best.pth')))

        test_time, (test_pred, test_true) = evaluate_model(model, dataloaders['test'])
        print(f"Test Accuracy: {accuracy_score(test_true, test_pred):.4f}", file=tee)

        # 计算每个类别的准确率
        cm = confusion_matrix(test_true, test_pred)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)

        for i, acc in enumerate(class_accuracies):
            print(f"Class {reverse_category_map[i]} Accuracy: {acc:.4f}", file=tee)

        # 打印混淆矩阵
        print("Confusion Matrix:", file=tee)
        print(cm, file=tee)

        # model_analysis
        # 总参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f'总参数量: {total_params:,}', file=tee)  
        # 获取模型大小
        model_size = get_model_size(model, os.path.join(save_dir, 'model_best.pth'))
        print(f"Model Size: {model_size:.2f} MB", file=tee)
        # 打印训练时间
        print(f"Training time: {train_time:.2f} seconds", file=tee)
        # 在测试集上进行最终评估并计算推理时间
        print(f"Inference time on test set: {test_time:.2f} seconds", file=tee)

        # 重置stdout
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
