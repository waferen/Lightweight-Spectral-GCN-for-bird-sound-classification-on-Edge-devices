import numpy as np
import os
from tqdm import tqdm 
import librosa
from config import*
import scipy.fftpack

def LFCC(data_path, n_lfccs):
    Label = []  # 存储标签
    Data = []  # 存储音频特征数据
    
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

        # 计算线性频谱图（而不是Mel频谱图）
        S = np.abs(librosa.stft(sig_preemph,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               win_length=win_length,
                               window=window))
        
        # 转换为分贝刻度
        S_dB = librosa.amplitude_to_db(S, ref=ref, amin=amin, top_db=top_db)
        
        # 抽帧
        S_dB = extract_high_energy_frames(S_dB, n_frames=n_frames)
        
        # 计算LFCC（使用线性频率尺度和DCT变换）
        # 1. 创建线性频率滤波器组
        n_filters = 20  # 可以根据需要调整滤波器数量
        freq_bins = np.linspace(0, 1, n_fft // 2 + 1)
        filter_banks = np.zeros((n_filters, n_fft // 2 + 1))
        
        # 创建均匀分布的三角滤波器
        for i in range(n_filters):
            # 计算三角滤波器的左中右点
            left = int((n_fft // 2 + 1) * i / (n_filters + 1))
            center = int((n_fft // 2 + 1) * (i + 1) / (n_filters + 1))
            right = int((n_fft // 2 + 1) * (i + 2) / (n_filters + 1))
            
            # 创建三角滤波器
            for j in range(left, center):
                filter_banks[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filter_banks[i, j] = (right - j) / (right - center)
        
        # 2. 应用滤波器组
        filtered = np.dot(filter_banks, S_dB)
        
        # 3. 应用DCT
        lfccs = scipy.fftpack.dct(filtered, type=2, axis=0, norm='ortho')[:n_lfccs]
        
        # 计算delta特征
        lfcc_deltas = librosa.feature.delta(lfccs)
        
        # 拼接lfcc和它的delta
        features = np.vstack((lfccs, lfcc_deltas)).T
        
        # 将特征添加到Data列表
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
    with open(os.path.join(output_dir, f'lfcc_{dataset_type}_{n_lfccs}.txt'), 'w') as f:
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

    return os.path.join(output_dir, f'lfcc_{dataset_type}_{n_lfccs}.txt')