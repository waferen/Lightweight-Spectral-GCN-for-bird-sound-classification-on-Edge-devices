import matplotlib.pyplot as plt
# 可视化.py
def visualize_results(loss_list, accuracy_train, accuracy_test, i):
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))  # 设置图形大小
    
    # 绘制损失值变化曲线
    epochs = range(1, len(loss_list) + 1)
    plt.plot(epochs, loss_list, label='Loss', color='red')
    
    # 计算准确率对应的 epoch
    acc_epochs = range(10, len(loss_list) + 1, 10)
    
    # 绘制训练准确率
    plt.plot(acc_epochs, accuracy_train, label='Train Accuracy', color='blue', linestyle='--')
    
    # 绘制测试准确率
    plt.plot(acc_epochs, accuracy_test, label='Test Accuracy', color='green', linestyle='-.')

    # 添加标题和标签，包括第几次交叉验证的信息
    plt.title(f'Training Loss and Accuracies Over Time (Cross-Validation #{i+1})', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Value', fontsize=12)

    # 设置图例
    plt.legend(loc='best', fontsize=10)
    
    # 显示网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 美化坐标轴
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 展示图形
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()