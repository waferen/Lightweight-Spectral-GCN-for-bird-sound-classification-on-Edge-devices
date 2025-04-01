import sys

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