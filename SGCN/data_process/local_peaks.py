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


def smooth_spectrum(spectrum, window_size=5):
    """
    使用移动平均滤波器平滑频谱。
    
    :param spectrum: 一维数组，表示频率分量的幅度。
    :param window_size: 平滑窗口大小。
    :return: 平滑后的频谱。
    """
    # 使用np.convolve进行简单移动平均
    kernel = np.ones(window_size) / window_size
    smoothed_spectrum = np.convolve(spectrum, kernel, mode='same')
    return smoothed_spectrum

def find_local_peaks(spectrum, mode='value', num_peaks=13):
    """
    在给定的频谱中找到局部峰值。
    
    :param spectrum: 一维数组，表示频率分量的幅度。
    :param mode: 'value' 或 'distance'，决定是基于峰值值还是基于峰值间的距离来选择峰值。
    :param num_peaks: 要找的局部峰值的数量。
    :return: 一个元组列表，每个元素是一个 (频率索引, 幅度) 的元组。
    """
    # 找到所有的局部峰值
    peaks, _ = find_peaks(spectrum)
    
    if len(peaks) == 0:
        # 如果没有找到任何峰值，返回全零的峰值
        return [(0, 0)] * num_peaks
    
    if mode == 'value':
        # 按照峰值的大小降序排序并选取前num_peaks个
        peak_indices = peaks[np.argsort(-spectrum[peaks])[:num_peaks]]
    elif mode == 'distance':
        # 选择彼此之间距离最远的num_peaks个峰值
        selected_peaks = []
        for _ in range(num_peaks):
            if len(peaks) == 0:
                break
            # 选择当前距离其他峰值最远的一个峰值
            distances = np.abs(peaks[:, None] - peaks).sum(axis=1)
            farthest_peak_index = np.argmax(distances)
            selected_peaks.append(peaks[farthest_peak_index])
            # 移除已选的峰值
            peaks = np.delete(peaks, farthest_peak_index)
        
        peak_indices = selected_peaks
    else:
        raise ValueError("mode should be either 'value' or 'distance'")
    
    # 如果找到的峰值数量少于 num_peaks，用零填充
    if len(peak_indices) < num_peaks:
        # 计算需要填充的数量
        padding_size = num_peaks - len(peak_indices)
        # 创建填充的峰值 (频率索引, 幅度) 元组
        padding_peaks = [(0, 0)] * padding_size
        # 将实际找到的峰值和填充的峰值合并
        peak_indices = np.concatenate([peak_indices, [0] * padding_size])
        peak_info = [(index, spectrum[index] if index in peaks else 0) for index in peak_indices]
    else:
        # 返回峰值对应的频率索引和幅度
        peak_info = [(index, spectrum[index]) for index in peak_indices]
    
    return peak_info

def Local_Peaks(data_path,n_peaks):
    Label = []  # 存储标签
    
    Data=[]
    # 确定是训练集还是测试集
    dataset_type = 'train' if 'train' in data_path else 'test'
    
    # 获取文件列表
    files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    
    # 遍历每个WAV文件，并使用tqdm显示进度条
    for file in tqdm(files, desc=f"Processing {dataset_type} files", unit="file"):
        wav_path = os.path.join(data_path, file)
        # 使用librosa加载音频文件并指定采样率
        sig, fs = librosa.load(wav_path, sr=sample_rate)
        # 应用预加重
        sig_preemph = np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])

        # # 计算Mel频谱图
        # S = librosa.feature.melspectrogram(y=sig_preemph,
        #                                     sr=sample_rate,
        #                                     n_fft=n_fft,
        #                                     hop_length=hop_length,
        #                                     win_length=win_length,
        #                                     window=window,
        #                                     n_mels=mel_bins,
        #                                     fmin=fmin,
        #                                     fmax=fmax)
        # # 形状 (n_mels, n_frames)n_mels 是 Mel 频率 bin 的数量。n_frames 是时间帧的数量。
        # S_dB = librosa.power_to_db(S, ref=ref, amin=amin, top_db=top_db)
        
        # 计算STFT
        D = librosa.stft(y=sig_preemph,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=window)

        # 将复数的STFT结果转换为幅度谱
        magnitude = np.abs(D)

        # 将幅度谱转换为分贝单位
        S_dB = librosa.amplitude_to_db(magnitude, ref=ref, amin=amin, top_db=top_db)
        


        # 抽帧（统一去掉了百分之二十静音帧）
        S_dB = extract_high_energy_frames(S_dB, n_frames=n_frames)
        features = []
        # 对每一帧进行平滑处理，并找到local peaks
        # 对每一帧进行处理
        for frame in S_dB.T:  # 每一列是一个时间帧
            smoothed_frame = smooth_spectrum(frame, window_size=5)
            peaks_info = find_local_peaks(smoothed_frame, num_peaks=n_peaks)
            peak_freqs, peak_dbs = zip(*peaks_info)  # 解包成频率和dB值
            
            # 将峰值频率和对应的dB值合并成一个列表
            frame_features = list(peak_freqs) + list(peak_dbs)
            features.append(frame_features)
    
        Data.append(features)
        # 前四个字符作为标签并进行映射
        label_key = file[:4]
        label = category_map.get(label_key, -1)
        Label.append(label)
    
    # 创建输出目录
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)

    # 转换为NumPy数组
    tr_feat = np.array(Data)
    tr_label = np.array(Label)

   
    # 写入文件
    with open(os.path.join(output_dir, f'Local_peaks_{dataset_type}_{n_peaks}.txt'), 'w') as f:
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
    
    return os.path.join(output_dir, f'Local_peaks_{dataset_type}_{n_peaks}.txt')


