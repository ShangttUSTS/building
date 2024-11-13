import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 方法名称
methods = [
    'LifeLongGoMf_CC_BP', 'LifeLongGoMf_BP_CC', 'LifeLongGoCc_MF_BP',
    'LifeLongGoCc_BP_MF', 'LifeLongGoBp_CC_MF', 'LifeLongGoBp_MF_CC'
]

# MF 数据
mf_data = np.array([
    [0.545, 0.536, 0.849],
    [0.545, 0.537, 0.851],
    [0.546, 0.541, 0.849],
    [0.538, 0.534, 0.857],
    [0.550, 0.550, 0.862],
    [0.547, 0.541, 0.855]
])

# CC 数据
cc_data = np.array([
    [0.720, 0.731, 0.927],
    [0.725, 0.735, 0.934],
    [0.720, 0.729, 0.928],
    [0.719, 0.730, 0.929],
    [0.724, 0.733, 0.933],
    [0.727, 0.736, 0.931]
])

# BP 数据
bp_data = np.array([
    [0.473, 0.459, 0.876],
    [0.472, 0.454, 0.874],
    [0.474, 0.457, 0.878],
    [0.465, 0.443, 0.875],
    [0.458, 0.436, 0.870],
    [0.458, 0.437, 0.870]
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
        # 绘制条形图
        bar_plot = sns.barplot(x=methods, y=data[:, j], ax=axs[i, j],
                               palette=sns.color_palette("Blues", 6), ci=None, alpha=0.7)

        # 添加数值标记
        for index, value in enumerate(data[:, j]):
            axs[i, j].text(index, value + 0.01, f'{value:.3f}',
                           ha='center', va='bottom', fontsize=10)

        # 绘制折线图
        # axs[i, j].plot(methods, data[:, j], marker='o', color='red', linewidth=2)

        # 设置标题和标签
        axs[i, j].set_title(f'{dataset_name} - {metrics[j]}', fontsize=16)
        axs[i, j].set_ylim(0, 1.2)
        axs[i, j].set_xticklabels(methods, rotation=45)
        axs[i, j].set_ylabel('Value', fontsize=14)

# 设置 x 轴标签
for ax in axs[-1]:  # 最后一行设置 x 轴标签
    ax.set_xlabel('Methods', fontsize=14)

# 设置整体标题
plt.suptitle('Model Performance Across Different GO Sub-ontologies', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('experiments_results.png', bbox_inches='tight')
plt.show()
