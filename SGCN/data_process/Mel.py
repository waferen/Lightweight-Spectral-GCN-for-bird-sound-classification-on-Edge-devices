import numpy as np
import os
import sys
from tqdm import tqdm 
import librosa
import random
from collections import Counter
from scipy.signal import find_peaks
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建config文件的路径（假设config在父文件夹的父文件夹中）
config_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 将config目录添加到sys.path
sys.path.append(config_dir)

from config import*
from sklearn.model_selection import train_test_split

@timer
def MelSpectrogram(data_path):
    Label = []  # 存储标签
    Data = []  # 存储音频特征数据
    
    # 确定是训练集还是测试集还是验证集
    if 'train' in data_path:
        dataset_type = 'train'
    elif 'val' in data_path:
        dataset_type = 'val'
    else:
        dataset_type = 'test'
    
    # 获取文件列表
    files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    # 如果是训练集并且需要根据data_percent进行抽样
    if data_percent < 1.0 and dataset_type == 'train':
        # 提取类别标签
        labels = [file[:4] for file in files]  # 假设类别在文件名的前四个字符
        
        # 打印原始类别数量
        original_counts = Counter(labels)
        print("Original class distribution:", dict(original_counts))
        
        # 进行抽样
        files, _ = train_test_split(files, train_size=data_percent, random_state=seed, stratify=labels)
        
        # 打印抽样后的类别数量
        sampled_labels = [file[:4] for file in files]
        sampled_counts = Counter(sampled_labels)
        print("Sampled class distribution:", dict(sampled_counts))

    # 遍历每个WAV文件，并使用tqdm显示进度条
    for file in tqdm(files, desc=f"Processing {dataset_type} files", unit="file"):
        wav_path = os.path.join(data_path, file)
        
        # 使用librosa加载音频文件并指定采样率
        sig, fs = librosa.load(wav_path, sr=sample_rate)
        # 应用预加重
        sig_preemph = np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])

        # 计算Mel频谱图
        S = librosa.feature.melspectrogram(y=sig_preemph,
                                            sr=sample_rate,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            win_length=win_length,
                                            window=window,
                                            n_mels=mel_bins,
                                            fmin=fmin,
                                            fmax=fmax)
        # 形状 (n_mels, n_frames)n_mels 是 Mel 频率 bin 的数量。n_frames 是时间帧的数量。
        S_dB = librosa.power_to_db(S, ref=ref, amin=amin, top_db=top_db)
        
        # 抽帧
        S_dB = extract_high_energy_frames(S_dB, n_frames=n_frames)
        
        # Mel频谱图
        S_dB = S_dB.T
        
        # 将Mel频谱图作为特征添加到Data列表
        Data.append(S_dB)
        
        # 前四个字符作为标签并进行映射
        label_key = file[:4]
        label = category_map.get(label_key, -1)
        Label.append(label)

    # 转换为NumPy数组
    tr_feat = np.array(Data)
    tr_label = np.array(Label)

    # 创建输出目录
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)

    # 写入文件
    with open(os.path.join(output_dir, f'Mel_{dataset_type}.txt'), 'w') as f:
        # 记录样本的数量
        f.write(str(tr_feat.shape[0]) + '\n')

        # 使用tqdm显示写入进度
        for i in tqdm(range(tr_feat.shape[0]), desc="Writing samples to file", unit="sample"):
            for j in range(tr_feat.shape[1]):
                if j == 0:
                    f.write('%s %s\n' % (tr_feat.shape[1], int(tr_label[i])))
                if j == tr_feat.shape[1] - 1:
                    a = [e.astype('float16') for e in tr_feat[i, j]]
                    f.write('%s 1 %s ' % (j, j - 1) + ' '.join(str(e) + ' ' for e in a) + '\n')
                else:
                    a = [e.astype('float16') for e in tr_feat[i, j]]
                    f.write('%s 1 %s ' % (j, j + 1) + ' '.join(str(e) + ' ' for e in a) + '\n')

    return os.path.join(output_dir, f'Mel_{dataset_type}.txt')


