import torch
import torchaudio
import librosa
import numpy as np
import os
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from config import*
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from config import *
# 标签映射
category_map = {
    "0009": 0, "0017": 1, "0034": 2, "0036": 3, "0074": 4,
    "0077": 5, "0114": 6, "0121": 7, "0180": 8, "0202": 9,
    "0235": 10, "0257": 11, "0265": 12, "0281": 13, "0298": 14,
    "0300": 15, "0364": 16, "0368": 17, "0370": 18, "1331": 19
}

def process_audio(input_path, output_path):
    # 创建保存图像的目录
    os.makedirs(output_path, exist_ok=True)

    # 获取所有wav文件
    wav_files = [f for f in os.listdir(input_path) if f.endswith('.wav')]
    
    # 使用tqdm创建进度条
    for file in tqdm(wav_files, desc="处理音频文件"):
        category = file[:4]  # 获取类别（前4个字符）
        label = category_map.get(category, -1)
        
        wav_path = os.path.join(input_path, file)
        
        # 使用 Librosa 加载音频文件
        y, sr = librosa.load(wav_path, sr=sample_rate)
        
        # 计算梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, mel_bins=mel_bins, fmin=fmin, fmax=fmax)
        
        # 转换为分贝单位
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 归一化
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # 生成图像
        plt.figure(figsize=(10, 10))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax)
        plt.axis('off')
        plt.title('Mel Spectrogram')
        plt.tight_layout(pad=0)
        
        # 保存图像
        image_filename = os.path.join(output_path, f"{file[:-4]}.png")
        plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 读取保存的图像并调整大小
        img = cv2.imread(image_filename)
        img = cv2.resize(img, (224, 224))  # 调整为 EfficientNet-B0 的输入尺寸
        
        # 保存调整后的图像
        cv2.imwrite(image_filename, img)

# 主程序
data_root = "data"  # 数据根目录
output_root = "Mel_spectrogram_images"  # 输出根目录

for dataset in ['train', 'test']:
    input_path = os.path.join(data_root, dataset)
    output_path = os.path.join(output_root, dataset)
    print(f"处理 {dataset} 数据...")
    process_audio(input_path, output_path)

print("Mel_Spectrogram extraction and saving complete for both train and test datasets.")



def load_finetuned_efficientnet(checkpoint_path, device='cuda'):
    # 导入EfficientNet的原始实现
    from efficientnet_pytorch import EfficientNet
    
    # 创建模型
    model = EfficientNet.from_name('efficientnet-b0')
    
    # 移除分类头，替换为Identity层
    model._fc = torch.nn.Identity()
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 处理状态字典
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # 处理状态字典中的键名前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除可能的'module.'前缀
        if k.startswith('module.'):
            k = k[7:]
        # 移除可能的'_orig_mod.'前缀
        if k.startswith('_orig_mod.'):
            k = k[10:]
        new_state_dict[k] = v
    
    # 加载处理后的状态字典
    model.load_state_dict(new_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # 定义预处理转换
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return model, preprocess


def extract_features_with_mel(waveform, sample_rate, window_size, hop_size, model, preprocess, device):
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    features_list = []
    
    for start in range(0, waveform.shape[1] - window_samples + 1, hop_samples):
        end = start + window_samples
        window_waveform = waveform[:, start:end].squeeze().numpy()
        
        mel_spec = librosa.feature.melspectrogram(
            y=window_waveform, 
            sr=sample_rate,
            mel_bins=mel_bins,  # 使用全局参数
            fmin=fmin,      # 使用全局参数
            fmax=fmax       # 使用全局参数
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        mel_spec_rgb = np.stack([mel_spec_db] * 3, axis=-1)
        mel_spec_rgb = (mel_spec_rgb * 255).astype(np.uint8)
        
        image = Image.fromarray(mel_spec_rgb)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(image_tensor)
            
        features_list.append(embedding.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)


def EfficientNet_data(data_path, model_name, window_size, hop_size, category_map, output_dir, checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = load_finetuned_efficientnet(checkpoint_path, device)
    
    datasets = ['train', 'test']
    output_files = []
    
    for dataset_type in datasets:
        Label = []
        Data = []
        
        dataset_folder = os.path.join(data_path, dataset_type)
        files = [f for f in os.listdir(dataset_folder) if f.endswith('.wav')]
        
        for file in tqdm(files, desc=f"Processing {dataset_type} files", unit="file"):
            wav_path = os.path.join(dataset_folder, file)
            
            waveform, sr = torchaudio.load(wav_path)
            
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)
            
            features = extract_features_with_mel(
                waveform, 
                sample_rate, 
                window_size, 
                hop_size, 
                model, 
                preprocess,
                device
            )
            
            Data.append(features)
            
            label_key = file[:4]
            label = category_map.get(label_key, -1)
            Label.append(label)
        
        tr_feat = np.array(Data)
        tr_label = np.array(Label)
        
        output_path = os.path.join(output_dir, f'{model_name}_{dataset_type}.txt')
        with open(output_path, 'w') as f:
            f.write(str(tr_feat.shape[0]) + '\n')

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
        
        output_files.append(output_path)
    
    return output_files

window_size = 1.0  # 1秒窗口
hop_size = 0.1    # 0.1秒步长
category_map = {
    "0009": 0, "0017": 1, "0034": 2, "0036": 3, "0074": 4,
    "0077": 5, "0114": 6, "0121": 7, "0180": 8, "0202": 9,
    "0235": 10, "0257": 11, "0265": 12, "0281": 13, "0298": 14,
    "0300": 15, "0364": 16, "0368": 17, "0370": 18, "1331": 19
}
# 反转字典
reverse_category_map = {v: k for k, v in category_map.items()}
output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True)
checkpoint_path = 'model/EffcientNet_finetune_model_for_GNN/model_last.pth'
output_file = EfficientNet_data(
    data_path='data',
    model_name='efficientnet',
    window_size=window_size,
    hop_size=hop_size,
    category_map=category_map,
    output_dir=output_dir,
    checkpoint_path=checkpoint_path
)