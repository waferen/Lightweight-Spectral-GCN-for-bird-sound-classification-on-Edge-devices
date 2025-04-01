import os
    # 获取所有.wav文件路径的函数
def get_audio_files_from_directory(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):  # 只获取 .wav 文件
                audio_files.append(os.path.join(root, file))
    return audio_files