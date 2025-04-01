import matplotlib.pyplot as plt
import pandas as pd
import os
# 数据准备
category_map = {
    "Class 0": "Gray Goose", "Class 1": "Whooper Swan", "Class 2": "Mallard Duck",
    "Class 3": "Green-winged Teal", "Class 4": "Grey Partridge", "Class 5": "Common Quail",
    "Class 6": "Pheasant", "Class 7": "Red-throated Diver", "Class 8": "Grey Heron",
    "Class 9": "Great Cormorant", "Class 10": "Eurasian Sparrowhawk", "Class 11": "Eurasian Buzzard",
    "Class 12": "Western Water Rail", "Class 13": "Moorhen", "Class 14": "Black-winged Stilt",
    "Class 15": "Northern Lapwing", "Class 16": "White-rumped Sandpiper", "Class 17": "Redshank",
    "Class 18": "Wood Sandpiper", "Class 19": "House Sparrow"
}

def create_accuracy_df(layer_data, layer_name):
    df = pd.DataFrame(layer_data, columns=['Accuracy'])
    df['Category'] = ['Class ' + str(i) for i in range(len(layer_data))]
    df['Bird'] = df['Category'].map(category_map)
    df['Layer'] = layer_name  # 添加图卷积层数信息
    return df[['Layer', 'Bird', 'Accuracy']]

# 第一组：一层图卷积
layer1_accuracy = [0.8882, 0.8625, 0.9221, 0.8017, 0.1667, 0.9730,
                   0.8812, 0.9102, 0.9353, 0.8187, 0.7347, 0.4828,
                   0.8088, 0.8261, 0.9177, 0.9080, 0.8239, 0.8354,
                   0.8424, 0.8703]
df_layer1 = create_accuracy_df(layer1_accuracy, 'One Layer GCN')

# 第二组：两层图卷积
layer2_accuracy = [0.8487, 0.9125, 0.9156, 0.8347, 0.3333, 0.9797,
                   0.9062, 0.9521, 0.9471, 0.8538, 0.7619, 0.6207,
                   0.8529, 0.8478, 0.8797, 0.9141, 0.9014, 0.8797,
                   0.8909, 0.8745]
df_layer2 = create_accuracy_df(layer2_accuracy, 'Two Layers GCN')

# 第三组：三层图卷积
layer3_accuracy = [0.8618, 0.9625, 0.9286, 0.7686, 0.1667, 0.9595,
                   0.8438, 0.9281, 0.8706, 0.7778, 0.7211, 0.6552,
                   0.7868, 0.8261, 0.8987, 0.9325, 0.7394, 0.8418,
                   0.8727, 0.8954]
df_layer3 = create_accuracy_df(layer3_accuracy, 'Three Layers GCN')

# 创建折线图进行对比
def plot_line_accuracy(*dfs):
    plt.figure(figsize=(14, 8))

    for df in dfs:
        plt.plot(df['Bird'], df['Accuracy'], label=df['Layer'].iloc[0], marker='o', linestyle='-', markersize=5)

    plt.xlabel('Bird Species')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison of Each Bird Category for Different GCN Layers')
    plt.xticks(rotation=45, ha="right")  # 旋转标签以便更好地显示
    plt.legend(title='GCN Layers')
    
    # 增加网格
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    # 保存为PDF文件，路径在figs文件夹下
    output_dir = 'figs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'ablation_layers_accuracy.pdf'), format='pdf')
    plt.show()

# 绘制折线图
plot_line_accuracy(df_layer1, df_layer2, df_layer3)
