import torch
import torchaudio
import os
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F

class AudioGraphDataset(Dataset):
    def __init__(self, audio_files, category_map, sample_rate, n_mels, fmin, fmax, target_length):
        self.audio_files = audio_files  # 音频文件列表
        self.category_map = category_map  # 标签映射
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.target_length = target_length  # 目标长度，填充或裁剪到此长度

    def __len__(self):
        return len(self.audio_files)

    def _generate_feature_vectors(self, waveform):
        # 将音频信号转换为 Mel 频谱图
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax
        )(waveform)

        # 确保 Mel 频谱图的形状是 [num_frames, n_mels] (每帧一个特征)
        mel_spectrogram = mel_spectrogram.squeeze(0).T  # 转置得到 [num_frames, n_mels]
        
        # 返回的特征应该是 [num_frames, n_mels]
        return mel_spectrogram

    def __getitem__(self, idx):
        # 获取音频文件路径
        wav_file_path = self.audio_files[idx]
        
        # 获取文件名前四个字符以提取类别标签
        label_str = os.path.basename(wav_file_path)[:4]
        
        # 使用提供的 category_map 转换为数值标签
        label = self.category_map.get(label_str, -1)  # 若无匹配则返回 -1
        
        # 加载音频并构建图数据
         # 加载音频并确保它是单通道的
        waveform, _ = torchaudio.load(wav_file_path, normalize=True)
        
        # 如果是多通道音频，只取第一个通道
        if waveform.shape[0] > 1:
            waveform = waveform[0, :]  # 只取第一个通道

        feature_vectors = self._generate_feature_vectors(waveform)

        # 填充或裁剪特征向量到目标长度
        feature_vectors = self._pad_or_trim(feature_vectors)

        # 构建图数据对象
        edge_index = self._build_edge_index(feature_vectors.shape[0])  # 连接所有相邻的帧
        
        # 创建 Data 对象
        graph_data = Data(
            x=feature_vectors,  # 每帧的特征，应该是 [num_frames, n_mels]
            edge_index=edge_index,  # 图的边索引
            y=torch.tensor(label, dtype=torch.long)  # 类别标签
        )
            # 打印 graph_data 的各元素形状
        # print(f"Feature vectors shape: {graph_data.x.shape}")
        # print(f"Edge index shape: {graph_data.edge_index.shape}")
        # print(f"Label shape: {graph_data.y.shape}")
        return graph_data

    def _pad_or_trim(self, feature_vectors):
        """填充或裁剪特征向量至统一的目标长度"""
        current_length = feature_vectors.shape[0]
        if current_length < self.target_length:
            # 如果当前长度小于目标长度，填充零
            padding = self.target_length - current_length
            feature_vectors = F.pad(feature_vectors, (0, 0, 0, padding), mode='constant', value=0)
        elif current_length > self.target_length:
            # 如果当前长度大于目标长度，裁剪掉多余的部分
            feature_vectors = feature_vectors[:self.target_length, :]
        return feature_vectors

    def _build_edge_index(self, num_frames):
        """
        构建每帧之间有前后两个节点相连的环形图的边索引。
        每个节点与前后相邻节点有单向连接，形成一个环形结构。
        """
        row = torch.arange(num_frames)  # 从0到num_frames-1
        col = torch.roll(row, shifts=-1)  # 将行滚动，形成环形连接
        edge_index = torch.stack([row, col], dim=0)
        return edge_index
