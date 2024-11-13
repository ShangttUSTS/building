import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 方法名称
methods = [
    'DeepGO-SE', 'DeepGraphGO', 'Naive',
    'DeepGOCNN', 'MLP', 'PU-GO',
    'DeepGOZero', 'LifelongGO_MF',
    'LifelongGO_MF_CC', 'LifelongGO_MF_CC_BP'
]

# MF 数据
mf_data = np.array([
    [0.538, 0.535, 0.849],
    [0.411, 0.346, 0.659],
    [0.314, 0.178, 0.500],
    [0.365, 0.319, 0.629],
    [0.314, 0.197, 0.500],
    [0.537, 0.521, 0.848],
    [0.513, 0.482, 0.773],
    [0.539, 0.543, 0.870],
    [0.523, 0.511, 0.853],
    [0.545, 0.536, 0.849]
])

# CC 数据
cc_data = np.array([
    [0.705, 0.717, 0.912],
    [0.674, 0.682, 0.814],
    [0.618, 0.490, 0.500],
    [0.672, 0.692, 0.758],
    [0.611, 0.532, 0.500],
    [0.712, 0.727, 0.917],
    [0.642, 0.627, 0.719],
    [0.590, 0.648, 0.812],
    [0.710, 0.725, 0.929],
    [0.720, 0.731, 0.927]
])

# BP 数据
bp_data = np.array([
    [0.423, 0.401, 0.834],
    [0.322, 0.313, 0.714],
    [0.281, 0.172, 0.500],
    [0.339, 0.289, 0.639],
    [0.291, 0.238, 0.500],
    [0.457, 0.437, 0.868],
    [0.401, 0.355, 0.687],
    [0.412, 0.406, 0.813],
    [0.426, 0.402, 0.846],
    [0.473, 0.459, 0.876]
])

# 创建 3x3 的子图
fig, axs = plt.subplots(3, 3, figsize=(18, 12))

# 定义数据和指标
datasets = [mf_data, cc_data, bp_data]
metrics = ['Fmax', 'AUPR', 'AUC']
dataset_names = ['MF', 'CC', 'BP']

# 绘制每个子图
for i, (data, dataset_name) in enumerate(zip(datasets, dataset_names)):
    for j in range(3):
        # 计算颜色深度
        norm = plt.Normalize(data[:, j].min(), data[:, j].max())
        colors = sns.color_palette("Blues", 6)
        bar_colors = [colors[int(norm(value) * (len(colors) - 1))] for value in data[:, j]]

        # 绘制条形图
        bar_plot = sns.barplot(y=methods, x=data[:, j], ax=axs[i, j], palette=bar_colors, ci=None, alpha=0.7)

        # 添加数值标记
        for index, value in enumerate(data[:, j]):
            axs[i, j].text(value + 0.01, index, f'{value:.3f}', ha='left', va='center', fontsize=10)

        # 找到最大值索引并高亮
        max_index = np.argmax(data[:, j])
        axs[i, j].patches[max_index].set_facecolor('darkorange')  # 设置最大值的颜色为深橙色

        # 设置标题和标签
        axs[i, j].set_title(f'{dataset_name} - {metrics[j]}', fontsize=16)
        axs[i, j].set_xlim(0, 1.2)
        axs[i, j].set_yticklabels(methods, rotation=0)
        axs[i, j].set_xlabel('Value', fontsize=14)

# 设置 y 轴标签
for ax in axs[:, 0]:  # 最左列设置 y 轴标签
    ax.set_ylabel('Methods', fontsize=14)

# 设置整体标题
# plt.suptitle('Different Model Performance Across Different GO Sub-ontologies', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('table3.png', bbox_inches='tight', dpi=300)  # 设置 dpi 参数
plt.show()