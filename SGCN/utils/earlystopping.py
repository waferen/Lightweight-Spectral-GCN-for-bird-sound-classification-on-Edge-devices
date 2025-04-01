import numpy as np
class EarlyStopping:
    """如果验证损失在给定的patience内没有改善，则提前停止训练。"""
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        参数:
            patience (int): 自验证损失最后一次改进后等待的周期数。
                            默认值: 7
            verbose (bool): 如果为True，在每次验证损失改进时打印一条消息。
                            默认值: False
            delta (float): 监控量改进的最小变化量。
                            默认值: 0
        """
        self.patience = patience  # 设置耐心度
        self.verbose = verbose  # 是否打印信息
        self.counter = 0  # 计数器，用于记录连续未改进的次数
        self.best_score = None  # 最佳得分
        self.early_stop = False  # 是否提前停止标志
        self.val_loss_min = np.Inf  # 初始验证损失最小值设为无穷大
        self.delta = delta  # 改进阈值

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            # 如果是第一次调用，则更新最佳得分
            self.best_score = score
        elif score < self.best_score - self.delta:
            # 如果得分没有足够改进，则增加计数器
            self.counter += 1
            print(f'提前停止计数器: {self.counter} 次中的 {self.patience}')
            if self.counter >= self.patience:
                # 如果计数器达到耐心度，则设置提前停止标志
                self.early_stop = True
        else:
            # 如果得分有改进，则更新最佳得分，并重置计数器
            self.best_score = score
            self.counter = 0
