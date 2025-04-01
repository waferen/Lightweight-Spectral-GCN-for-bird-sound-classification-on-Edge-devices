import numpy as np
import os
from spafe.features.mfcc import mfcc
from tqdm import tqdm 
import librosa
import random
from collections import Counter
from scipy.signal import find_peaks
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建config文件的路径（假设config在父文件夹的父文件夹中）
config_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 将config目录添加到sys.path
sys.path.append(config_dir)
from config import*

Data = []  # 用于存储特征

def extract_features(data_path):
    Label = []  # 存储标签
    Data = []  # 存储音频特征数据
    
    # 确定是训练集还是测试集
    dataset_type = 'train' if 'train' in data_path else 'test'
    
    # 获取文件列表
    files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    
    # 遍历每个WAV文件，并使用tqdm显示进度条
    for file in tqdm(files, desc=f"Processing {dataset_type} files", unit="file"):
        wav_path = os.path.join(data_path, file)
        
         # 加载音频文件
        sig, fs = librosa.load(wav_path, sr=sample_rate)

        # 应用预加重
        sig_preemph = np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])

        # 计算Mel频谱图
        S = librosa.feature.melspectrogram(
            y=sig_preemph,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax
        )
        S_dB = librosa.power_to_db(S)
        # 1. 计算MFCC
        mfccs = librosa.feature.mfcc(S=S_dB, n_mfcc=n_mfccs)
        mfcc_deltas = librosa.feature.delta(mfccs)

        # 2. 计算零交叉率
        zcr = librosa.feature.zero_crossing_rate(y=sig_preemph, frame_length=win_length, hop_length=hop_length)
        zcr_deltas = librosa.feature.delta(zcr)

        # 3. 计算基频F0和语音概率
        f0, voiced_flags, voice_probs = librosa.pyin(sig, fmin=50, fmax=500, sr=sample_rate)
        # 填充缺失值
        f0 = np.nan_to_num(f0)
        voice_probs = np.nan_to_num(voice_probs)
        f0_deltas = librosa.feature.delta(f0.reshape(1, -1))
        voice_probs_deltas = librosa.feature.delta(voice_probs.reshape(1, -1))

        # 4. 计算对数能量
        rms_energy = librosa.feature.rms(y=sig_preemph, frame_length=n_fft, hop_length=hop_length)
        rms_energy_deltas = librosa.feature.delta(rms_energy)
        # 打印每个特征的形状
        # print("MFCC shape:", mfccs.shape)
        # print("MFCC Deltas shape:", mfcc_deltas.shape)
        # print("Zero Crossing Rate (ZCR) shape:", zcr.shape)
        # print("ZCR Deltas shape:", zcr_deltas.shape)
        # print("F0 shape:", f0.reshape(1, -1).shape)
        # print("Voice Probability shape:", voice_probs.reshape(1, -1).shape)
        # print("F0 Deltas shape:", f0_deltas.shape)
        # print("Voice Probability Deltas shape:", voice_probs_deltas.shape)
        # print("Log RMS Energy shape:", np.log1p(rms_energy).shape)
        # print("RMS Energy Deltas shape:", rms_energy_deltas.shape)
        # 特征拼接
        features = np.vstack((
            mfccs,
            mfcc_deltas,
            zcr,
            zcr_deltas,
            f0.reshape(1, -1),
            voice_probs.reshape(1, -1),
            f0_deltas,
            voice_probs_deltas,
            # np.log1p(rms_energy),
            # rms_energy_deltas
        )).T  # 转置以使每行是一个帧的特征

        Data.append(features)
            
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
    with open(os.path.join(output_dir, f'mix_{dataset_type}.txt'), 'w') as f:
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

    return os.path.join(output_dir, f'mix_{dataset_type}.txt')



