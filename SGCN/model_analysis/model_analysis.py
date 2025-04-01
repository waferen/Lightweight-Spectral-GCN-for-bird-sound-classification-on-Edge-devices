import torch
import torch.nn as nn
import librosa
import numpy as np
import cv2
import time
import os
from thop import profile
from efficientnet_pytorch import EfficientNet
from GCN.graphcnn import Graph_CNN_ortega
from data_process.mfcc import MFCC
from utils.util import load_data

# 定义参数
num_classes = 20

# 音频处理参数
sample_rate = 44100
n_fft = 2048
hop_length = 320
win_length = 1024
fmin = 0
fmax = 24000
n_mfccs = 16  # 请根据您的实际设置调整

def process_audio_for_efficientnet(file_path):
    # 加载音频文件
    y, sr = librosa.load(file_path, sr=sample_rate)
    
    # 计算 STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # 获取幅度谱，并转换为 dB 单位
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # 归一化
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    
    # 调整大小为 EfficientNet-B0 的输入尺寸
    img = cv2.resize(S_db, (224, 224))
    
    # 转换为 RGB 图像（EfficientNet 需要 3 通道输入）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # 转换为 PyTorch 张量
    efficientnet_input = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
    
    return efficientnet_input

def process_audio_for_gcn(file_path):
    # 使用MFCC函数处理音频文件
    temp_dir = 'temp_mfcc'
    os.makedirs(temp_dir, exist_ok=True)
    
    # 获取原始文件名
    original_filename = os.path.basename(file_path)
    
    # 在临时目录中使用原始文件名
    temp_file = os.path.join(temp_dir, original_filename)
    
    # 复制音频文件到临时目录
    import shutil
    shutil.copy(file_path, temp_file)
    
    # 处理音频文件
    processed_file = MFCC(temp_dir, n_mfccs)
    
    # 加载处理后的数据
    g_list, _ = load_data(processed_file)
    
    # 清理临时文件
    os.remove(temp_file)
    os.remove(processed_file)
    os.rmdir(temp_dir)
    
    return g_list[0]  # 返回第一个（也是唯一的）图

# 定义 EfficientNet 模型
def create_efficientnet(num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

# 分析模型函数
def analyze_model(model, model_name, input_data):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        _ = model(input_data)
        end_time = time.time()
    
    inference_time = end_time - start_time
    total_params = sum(p.numel() for p in model.parameters())
    flops, _ = profile(model, inputs=(input_data,))
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    print(f"{model_name} 结果:")
    print(f"推理时间: {inference_time:.4f} 秒")
    print(f"总参数量: {total_params:,}")
    print(f"FLOPs: {flops:,}")
    print(f"模型大小: {model_size:.2f} MB")
    print("-" * 50)

# 主程序
def main():
    # 在开始就定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载音频文件
    audio_path = "data\\test\\0009_111654_1.wav"
    
    # 处理 EfficientNet 输入
    efficientnet_input = process_audio_for_efficientnet(audio_path)

    # EfficientNet
    efficientnet = create_efficientnet(num_classes).to(device)
    efficientnet.load_state_dict(torch.load('my_efficientnetb0_model_best.pth', map_location=device))
    analyze_model(efficientnet, "EfficientNet", efficientnet_input.to(device))

    # 处理 GCN 输入
    gcn_input = process_audio_for_gcn(audio_path)

    # 确保 GCN 输入在正确的设备上
    gcn_input.node_features = gcn_input.node_features.to(device)
    gcn_input.edge_mat = gcn_input.edge_mat.to(device)

    # GCN
    params = {
        'num_layers': 2,
        'hidden_dim': 64,
        'final_dropout': 0.5,
        'graph_pooling_type': 'mean'
    }
    A = torch.rand(gcn_input.node_features.shape[0], gcn_input.node_features.shape[0], device=device)
    gcn_model = Graph_CNN_ortega(params['num_layers'], gcn_input.node_features.shape[1],
                                 params['hidden_dim'], num_classes, params['final_dropout'], 
                                 params['graph_pooling_type'], device, A).to(device)
    try:
        gcn_model.load_state_dict(torch.load('checkpoint.pt', map_location=device))
    except RuntimeError as e:
        print(f"加载GCN模型权重时出错: {e}")
        print("继续使用未加载权重的模型进行分析...")
    
    analyze_model(gcn_model, "GCN", [gcn_input])

    # 打印音频文件路径
    print(f"处理的音频文件: {audio_path}")

if __name__ == "__main__":
    main()
