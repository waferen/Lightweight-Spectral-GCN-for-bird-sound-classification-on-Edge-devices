import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
# 从CSV文件加载数据
df = pd.read_csv('code/acc.csv')

# 将"Model"列设置为索引
df.set_index('Model', inplace=True)

# 标签映射字典，添加 "Class " 前缀来匹配列头
category_map = {
    "Class 0009": "Gray Goose",
    "Class 0017": "Whooper Swan",
    "Class 0034": "Mallard Duck",
    "Class 0036": "Green-winged Teal",
    "Class 0074": "Grey Partridge",
    "Class 0077": "Common Quail",
    "Class 0114": "Pheasant",
    "Class 0121": "Red-throated Diver",
    "Class 0180": "Grey Heron",
    "Class 0202": "Great Cormorant",
    "Class 0235": "Eurasian Sparrowhawk",
    "Class 0257": "Eurasian Buzzard",
    "Class 0265": "Western Water Rail",
    "Class 0281": "Moorhen",
    "Class 0298": "Black-winged Stilt",
    "Class 0300": "Northern Lapwing",
    "Class 0364": "White-rumped Sandpiper",
    "Class 0368": "Redshank",
    "Class 0370": "Wood Sandpiper",
    "Class 1331": "House Sparrow"
}

# 替换列名称为鸟类名称
df.columns = [category_map.get(col, col) for col in df.columns]

# 创建折线图
plt.figure(figsize=(14, 8))

# 使用seaborn的颜色调色板来分配每个模型不同的颜色
sns.set_palette("tab10")

# 为每个模型绘制折线图
for model in df.index:
    plt.plot(df.columns, df.loc[model], label=model, marker='o', linestyle='-', markersize=6)

# 添加标题和标签
plt.title('Comparison of Accuracy Across Models and Bird Species', fontsize=16)
plt.xlabel('Bird Species', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图例
plt.legend(title='Models', fontsize=12)

# 美化x轴和y轴的刻度
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)

# 调整布局
plt.tight_layout()

# 保存为PDF文件，路径在figs文件夹下
output_dir = 'figs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'comparison_accuracy.pdf'), format='pdf')

# 显示图形
plt.show()
