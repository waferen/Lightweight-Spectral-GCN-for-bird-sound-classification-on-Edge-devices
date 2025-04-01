import argparse  # 命令行参数解析
import torch  # PyTorch 主要库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数
import torch.optim as optim  # 优化器模块
import numpy as np  # 数值计算
import time  # 时间操作
import networkx as nx  # 图数据处理
from tqdm import tqdm  # 进度条显示
from sklearn.model_selection import train_test_split  # 数据集划分
import sys,gc
import os

from utils.train_test import train,test
from utils.util import load_data,separate_data
from data_process.mfcc import MFCC
from data_process.lfcc import LFCC
from data_process.local_peaks import Local_Peaks
from data_process.Mel import MelSpectrogram
from data_process.mixed_features import extract_features
from GCN.graphcnn import Graph_CNN_ortega
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建config文件的路径（假设config在父文件夹的父文件夹中）
config_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 将config目录添加到sys.path
sys.path.append(config_dir)
from config import*

def main():
    # 直接定义参数
    params = {
        'data_of_audio' : 'data' ,# 音频数据路径
        'train_dataset':'processed_data\Mel_train.txt',
        'test_dataset':'processed_data\Mel_test.txt',
        'val_dataset':'processed_data\Mel_val.txt',
        'device': 0,
        'batch_size': 128,
        'iters_per_epoch': 50,
        'epochs': 1000,
        'lr': 0.001,
        'seed': 0,
        'fold_idx': 5,
        'num_layers': 2,
        'hidden_dim': 64,
        'final_dropout': 0.5,
        'graph_pooling_type': "max",
        'graph_type': "cycle",
        'Standardization': True,
        'patience': 20,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 1e-4,
        'cross_validation': False,
        'features':'Mel',
        'output_path':'code/SGCN/output'
    }
    # 新建保存路径
    os.makedirs(params['output_path'],exist_ok=True)

    # 打开一个文件用于写入
    with open(os.path.join(params['output_path'],'experiment_results.txt'), 'w',encoding='utf-8') as f:
        # 创建一个Tee对象，它会同时写入到控制台和文件
        original_stdout = sys.stdout
        tee = Tee(sys.stdout, f)
        # 测试一次
        # 恢复原始stdout以避免最后三行被写入文件
        sys.stdout = original_stdout
            
        # 设置随机种子和设备
        torch.manual_seed(params['seed'])
        np.random.seed(params['seed'])    
        device = torch.device("cuda:" + str(params['device'])) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(params['seed'])

        # 数据预处理
        if params['features'] == 'mfcc':
            data_processing_time,params['train_dataset'] = MFCC(os.path.join(params['data_of_audio'],'train'))
            print(f"Data processing time for train dataset: {data_processing_time:.2f} seconds",file=tee)
            print('data process for train data complete')

            data_processing_time,params['test_dataset'] = MFCC(os.path.join(params['data_of_audio'],'test'))
            print(f"Data processing time for test dataset: {data_processing_time:.2f} seconds",file=tee)
            print('data process for test data complete')

            data_processing_time,params['val_dataset'] = MFCC(os.path.join(params['data_of_audio'],'val'))
            print(f"Data processing time for val dataset: {data_processing_time:.2f} seconds",file=tee)
            print('data process for val data complete')

        if params['features'] == 'Mel':
            data_processing_time,params['train_dataset'] = MelSpectrogram(os.path.join(params['data_of_audio'],'train'))
            print(f"Data processing time for train dataset: {data_processing_time:.2f} seconds",file=tee)
            print('data process for train data complete')

            data_processing_time, params['test_dataset'] = MelSpectrogram(os.path.join(params['data_of_audio'],'test'))
            print(f"Data processing time for test dataset: {data_processing_time:.2f} seconds",file=tee)
            print('data process for test data complete')

            data_processing_time, params['val_dataset'] = MelSpectrogram(os.path.join(params['data_of_audio'],'val'))
            print(f"Data processing time for val dataset: {data_processing_time:.2f} seconds",file=tee)
            print('data process for val data complete')

        # 加载图数据，这里传入的是一个已经处理好的txt文件的路径
        train_graphs, num_classes = load_data(params['train_dataset'])
        test_graphs, _ = load_data(params['test_dataset'])
        val_graphs,_ =load_data(params['val_dataset'])
        # 线形图or环形图
        A = nx.to_numpy_array(train_graphs[0].g)
        if params['graph_type'] == 'cycle':
            A[0, -1] = 1
            A[-1, 0] = 1
        A = torch.Tensor(A).to(device)
        # 初始化模型
        model = Graph_CNN_ortega(params['num_layers'], train_graphs[0].node_features.shape[1],
                                params['hidden_dim'], num_classes, params['final_dropout'], params['graph_pooling_type'],
                                device, A).to(device)
        # 训练及测试
        train_time,_ = train(model,params, device, train_graphs, test_graphs,val_graphs, num_classes, A)
        # 加载最佳模型
        model.load_state_dict(torch.load(os.path.join(params['output_path'],'best_model.pt')))
        
        # 最终测试
        test_time,(acc_train, acc_test_per_class) = test(params, model, device, train_graphs, test_graphs, num_classes)
        # 保存数据
        print(f' Final test acc: {acc_train}', file=tee)

        print("测试集每个类别的准确率:", file=tee)
        for class_idx, acc in enumerate(acc_test_per_class):
            print(f"类别 {class_idx}: {acc:.4f}", file=tee)

        # model_analysis
        # 总参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f'总参数量: {total_params:,}',file=tee)  
        # 获取模型大小
        model_size = get_model_size(model,os.path.join(params['output_path'], 'model_best.pth'))
        print(f"Model Size: {model_size:.2f} MB",file=tee)
        # 打印训练时间
        print(f"Training time: {train_time:.2f} seconds",file=tee)
        # 在测试集上进行最终评估并计算推理时间
        print(f"Inference time on test set: {test_time:.2f} seconds",file=tee)
        # 重置stdout
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
