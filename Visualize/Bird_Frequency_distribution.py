import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random
from spafe.frequencies.dominant_frequencies import get_dominant_frequencies

# 参数
seed = 0
sample_rate = 16000
n_mfccs = 14
mel_bins = 64
window_time_ms = 20  # 窗口长度，单位：毫秒
hop_time_ms = 10     # 步长，单位：毫秒
# 根据采样率计算 win_length 和 hop_length
win_length = int(window_time_ms * sample_rate / 1000)
hop_length = int(hop_time_ms * sample_rate / 1000) 
n_fft = 2048
window = 'hann'
ref = 1.0
amin = 1e-10
top_db = 80.0
n_frames = 200
# 预加重系数
pre_emph_coeff = 0.97
# 数据路径
data_path = "data/train"  # 数据路径

# 标签映射
category_map = {
    "0009": "Gray Goose",
    "0017": "Whooper Swan",
    "0034": "Mallard Duck",
    "0036": "Green-winged Teal",
    "0074": "Grey Partridge",
    "0077": "Common Quail",
    "0114": "Pheasant",
    "0121": "Red-throated Diver",
    "0180": "Grey Heron",
    "0202": "Great Cormorant",
    "0235": "Eurasian Sparrowhawk",
    "0257": "Eurasian Buzzard",
    "0265": "Western Water Rail",
    "0281": "Moorhen",
    "0298": "Black-winged Stilt",
    "0300": "Northern Lapwing",
    "0364": "White-rumped Sandpiper",
    "0368": "Redshank",
    "0370": "Wood Sandpiper",
    "1331": "House Sparrow"
}

# 存储每种类别的频率统计数据
class_frequencies = {category: [] for category in category_map.values()}

# 获取所有wav文件
wav_files = [f for f in os.listdir(data_path) if f.endswith('.wav')]

# 新增的函数：计算音频文件的主频率
def calculate_dominant_frequencies(y, sr, n_fft, hop_length):
    # 计算梅尔频谱图
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # 计算频谱质心
    spectral_centroids = librosa.feature.spectral_centroid(S=S, sr=sr)
    # 取平均作为主频率
    dominant_frequency = np.mean(spectral_centroids)
    return dominant_frequency

# 遍历所有wav文件，并使用tqdm显示进度
for file in tqdm(wav_files, desc="Processing files"):
    # 从文件名获取类别
    category_label = file[:4]  # 获取文件名前四个字符
    label = category_map.get(category_label, -1)
    
    if label == -1:
        print(f"Unknown category: {category_label}")
        continue
    
    wav_path = os.path.join(data_path, file)
    
    # 加载音频文件
    y, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    
    # 计算主频率
    dominant_frequencies = calculate_dominant_frequencies(y, sample_rate, n_fft, hop_length)
    
    # 存储频率统计数据
    class_frequencies[label].append(dominant_frequencies)

# 输出目录
output_dir = 'figs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 在绘图之前设置字体
plt.rcParams['font.sans-serif'] = ['Arial']  # 设置为英文字体
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 绘制箱型图
plt.figure(figsize=(15, 8))
data = [class_frequencies[category] for category in class_frequencies.keys()]
box = plt.boxplot(data, labels=class_frequencies.keys(), showmeans=True, patch_artist=True,showfliers=False,
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  whiskerprops=dict(color='green', linewidth=1.5),
                  flierprops=dict(markerfacecolor='red', marker='o', markersize=8),
                  medianprops=dict(color='black', linewidth=2),
                  meanprops=dict(marker='o', markerfacecolor='yellow', markersize=8))

# 美化箱型图
plt.title('Box Plot of Dominant Frequencies by Category', fontsize=18, weight='bold')
plt.ylabel('Dominant Frequency (Hz)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# 设置y轴范围，避免显示低于0的值
plt.ylim(bottom=0)

# 添加网格线
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# 保存为PDF，路径在figs文件夹下
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dominant_frequencies_boxplot.pdf'), format='pdf')

# 显示图形
plt.show()

# 绘制均值-标准差图
mean_std_data = [(np.mean(freqs), np.std(freqs)) for freqs in data]
means, stds = zip(*mean_std_data)

plt.figure(figsize=(15, 8))
positions = np.arange(len(class_frequencies))
plt.errorbar(positions, means, yerr=stds, fmt='o', ecolor='red', capsize=5, markersize=10,
             markerfacecolor='blue', markeredgecolor='black', linestyle='None')

# 美化均值-标准差图
plt.xticks(positions, class_frequencies.keys(), rotation=45, fontsize=12)
plt.title('Mean and Standard Deviation of Dominant Frequencies by Category', fontsize=18, weight='bold')
plt.ylabel('Frequency (Hz)', fontsize=14)

# 设置y轴范围，避免显示低于0的值
plt.ylim(bottom=0)

# 添加网格线
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# 保存为PDF，路径在figs文件夹下
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dominant_frequencies_mean_std.pdf'), format='pdf')

# 显示图形
plt.show()
