import numpy as np
import time
import os
import torch
from thop import profile

# 参数
seed=0
sample_rate = 16000
n_mfccs = 14
mel_bins = 64
fmin = 50
fmax = 8000
window_time_ms = 20  # 窗口长度，单位：毫秒
hop_time_ms = 10     # 步长，单位：毫秒
data_percent=1
# 根据采样率计算 win_length 和 hop_length
win_length = int(window_time_ms * sample_rate / 1000) 
hop_length = int(hop_time_ms * sample_rate / 1000) 
n_fft = 2048
window = 'hann'
ref = 1.0
amin = 1e-10
top_db = 80.0
n_frames=200
# 预加重系数
pre_emph_coeff = 0.97

# 标签映射
category_map = {
    "0009": 0,
    "0017": 1,
    "0034": 2,
    "0036": 3,
    "0074": 4,
    "0077": 5,
    "0114": 6,
    "0121": 7,
    "0180": 8,
    "0202": 9,
    "0235": 10,
    "0257": 11,
    "0265": 12,
    "0281": 13,
    "0298": 14,
    "0300": 15,
    "0364": 16,
    "0368": 17,
    "0370": 18,
    "1331": 19
}

# 反转字典，用于从索引到类别的映射
reverse_category_map = {v: k for k, v in category_map.items()}


# 统计训练时间和推理时间
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result  # 返回时间和原函数的结果
    return wrapper

# 计算FLOPs
def get_flops(model, input_size=(1, 3, 224, 224)):
    if profile is not None:
        input_tensor = torch.randn(input_size)
        flops, _ = profile(model, inputs=(input_tensor,))
        return flops
    else:
        print("FLOPs calculation skipped due to missing thop package.")
        return 0

# 定义一个函数来创建一个同时写入文件和标准输出的类
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 确保数据立即写入文件

    def flush(self):
        for f in self.files:
            f.flush()

# 计算模型大小，传入保存路径
def get_model_size(model, save_path='model.pth'):
    # 保存模型到指定路径
    torch.save(model.state_dict(), save_path)  
    
    # 获取模型文件大小（MB）
    model_size = os.path.getsize(save_path) / (1024 * 1024)  # 获取文件大小（单位：MB）
    
    # 删除保存的模型文件
    os.remove(save_path)
    
    return model_size
#抽帧
def extract_high_energy_frames(S_db_mel, n_frames=220):
    """
    从Mel频谱图中抽取能量最大的前n帧。
    
    参数:
    S_db_mel (np.ndarray): Mel频谱图的分贝表示，形状为 (n_mels, n_frames)。
    n_frames (int): 要抽取的能量最大帧数，默认为220。
    
    返回:
    S_dB_high_energy_frames (np.ndarray): 抽取的能量最大的前n帧。
    high_energy_indices (np.ndarray): 布尔数组或索引数组，表示哪些帧是能量最大的帧。
    """
    
    # 计算每帧的能量
    frame_energy = np.sum(S_db_mel, axis=0)
    
    # 找出能量最大的n帧的索引
    high_energy_indices = np.argsort(frame_energy)[-n_frames:][::-1]
    
    # 根据这些索引从 S_db_mel 中提取帧
    S_dB_high_energy_frames = S_db_mel[:, high_energy_indices]
    
    return S_dB_high_energy_frames

# 计算一阶差分

def compute_delta(features, N=2):
    """
    从特征向量序列中计算 delta 特征。  
    参数:
        features (numpy.ndarray): 一个大小为 (帧数 x 特征数) 的 numpy 数组，包含特征。每一行表示一个特征向量。
        N (int): 对于每一帧，基于前后的 N 帧来计算 delta 特征。     
    返回:
        delta_features (numpy.ndarray): 一个大小为 (帧数 x 特征数) 的 numpy 数组，包含 delta 特征。每一行表示一个 delta 特征向量。
    异常:
        ValueError: 如果 N 小于 1，则抛出异常。
    """
    if N < 1:
        raise ValueError('N 必须是一个大于等于 1 的整数')
    
    # 获取帧数
    NUMFRAMES = len(features)
    
    # 计算分母
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    
    # 初始化 delta 特征数组
    delta_feat = np.empty_like(features)
    
    # 在边缘进行填充
    padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
    
    # 计算每一帧的 delta 特征
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t + 2*N + 1]) / denominator
    
    return delta_feat