import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os,sys
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils.data import AudioGraphDataset
from utils.data_path import get_audio_files_from_directory
from utils.GCN import Base_GCNModel
from utils.train_test import train, test
import time  # 添加时间模块
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建config文件的路径（假设config在父文件夹的父文件夹中）
config_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 将config目录添加到sys.path
sys.path.append(config_dir)
from config import*
def main():
    params = {
        'data_of_audio': 'data',  # 音频数据路径
        'device': 0,
        'batch_size': 16,
        'iters_per_epoch': 50,
        'epochs': 1000,
        'lr': 0.005,
        'seed': 0,
        'fold_idx': 5,
        'num_layers': 2,
        'hidden_dim': 64,
        'final_dropout': 0.5,
        'graph_pooling_type': 'mean',
        'graph_type': "cycle",
        'Standardization': True,
        'patience': 20,
        'input_dim': 32,
        'hidden_dim': 128,
        'num_classes': 20,
        'save_path': 'code/baseGCN/output',
        'sample_rate': 44100,
        'fmin': 0,
        'fmax': 24000,
        'target_length': 200
    }

    # 标签映射
    category_map = {
        "0009": 0, "0017": 1, "0034": 2, "0036": 3, "0074": 4, "0077": 5,
        "0114": 6, "0121": 7, "0180": 8, "0202": 9, "0235": 10, "0257": 11,
        "0265": 12, "0281": 13, "0298": 14, "0300": 15, "0364": 16, "0368": 17,
        "0370": 18, "1331": 19
    }

    # 创建保存路径的文件夹
    os.makedirs(params['save_path'], exist_ok=True)

    # 初始化随机种子
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    device = torch.device("cuda:" + str(params['device'])) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params['seed'])
     # 创建或打开 experiment 文件以写入输出
    

    # 创建模型
    model = Base_GCNModel(params['input_dim'], params['hidden_dim'], params['num_classes'], params['num_layers'], params['final_dropout']).to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    with open(os.path.join(params['save_path'],'experiment.txt'), 'w') as output_file:
        # 创建 Tee 实例，将标准输出和文件输出结合
        tee = Tee(output_file, sys.stdout)
        # 记录数据处理时间
        data_start_time = time.time()
        train_audio_files = get_audio_files_from_directory(os.path.join(params['data_of_audio'], 'train'))
        val_audio_files = get_audio_files_from_directory(os.path.join(params['data_of_audio'], 'val'))
        test_audio_files = get_audio_files_from_directory(os.path.join(params['data_of_audio'], 'test'))
        

        # 打印路径数量，确保加载正确
        print(f"Number of training audio files: {len(train_audio_files)}")
        print(f"Number of validating audio files: {len(val_audio_files)}")
        print(f"Number of testing audio files: {len(test_audio_files)}")

        # 创建数据集
        train_dataset = AudioGraphDataset(train_audio_files, category_map, params['sample_rate'], params['input_dim'], params['fmin'], params['fmax'], params['target_length'])
        val_dataset = AudioGraphDataset(val_audio_files, category_map, params['sample_rate'], params['input_dim'], params['fmin'], params['fmax'], params['target_length'])
        test_dataset = AudioGraphDataset(test_audio_files, category_map, params['sample_rate'], params['input_dim'], params['fmin'], params['fmax'], params['target_length'])
        data_processing_time = time.time() - data_start_time
        tee.write(f"Data Processing Time: {data_processing_time:.2f} seconds\n")
        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

        best_val_accuracy = 0.0  # Initialize best validation accuracy
        epochs_since_improvement = 0  # Counter to track number of epochs since last improvement
        # 记录训练开始时间
        train_start_time = time.time()
            
        for epoch in range(params['epochs']):
            print(f"EPOCH: {epoch}")

            
            # Train the model
            train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch)
            
            
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

            # Evaluate on validation set
            val_loss, val_accuracy, _ = test(model, device, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            # Check if validation accuracy has improved
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_since_improvement = 0
                torch.save(model.state_dict(), os.path.join(params['save_path'], 'best_model.pth'))
                print(f"Model saved to {os.path.join(params['save_path'], 'best_model.pth')}")
            else:
                epochs_since_improvement += 1

            # If no improvement for 'patience' epochs, stop training early
            if epochs_since_improvement >= params['patience']:
                print("Training stopped early due to lack of improvement.")
                break

            # Every 10 epochs, evaluate on the test set
            if (epoch + 1) % 10 == 0:
                print("\nEvaluating on the test set...")
                test_loss, test_accuracy, _ = test(model, device, test_loader, criterion)
                print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

            # Update learning rate
            scheduler.step(val_loss)
        
        # 记录训练结束时间
        train_time = time.time() - train_start_time
        tee.write(f"Training Time: {train_time:.2f} seconds\n")

        # Load best model and evaluate
        model.load_state_dict(torch.load(os.path.join(params['save_path'], 'best_model.pth')))
        tee.write("Best Model Performance on Each Class:\n")
        # 记录测试开始时间
        test_start_time = time.time()
        test_loss, test_accuracy, class_accuracies = test(model, device, test_loader, criterion)
        tee.write(f'Final Test Accuracy: {test_accuracy:.2f}%\n')
        for idx, acc in enumerate(class_accuracies):
            tee.write(f"Class {idx} Accuracy: {acc:.2f}%\n")
        test_time = time.time() - test_start_time
        tee.write(f"Testing Time: {test_time:.2f} seconds\n")
        # 打印模型大小和参数数量
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tee.write(f"Model Parameter Count: {model_size} parameters\n")
        # 计算模型大小并输出
        model_size = get_model_size(model, save_path=os.path.join(params['save_path'], 'best_model.pth'))
        tee.write(f"Model Size: {model_size:.2f} MB\n")  # 输出模型大小

if __name__ == "__main__":
    main()
