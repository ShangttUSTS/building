import numpy as np
import matplotlib.pyplot as plt

# 方法列表
methods = [
    'GVP', 'Esm2', 'EsmS',
    'EsmS-GVP(128)', 'Esm2-GVP(128)', 'Esm2-GVP(512)'
]

# 每个方法的指标值
mf_values = [
    [0.363, 0.315, 0.727],
    [0.513, 0.508, 0.841],
    [0.524, 0.520, 0.850],
    [0.516, 0.512, 0.848],
    [0.530, 0.527, 0.853],
    [0.545, 0.536, 0.849]
]

cc_values = [
    [0.627, 0.618, 0.782],
    [0.714, 0.728, 0.926],
    [0.696, 0.709, 0.909],
    [0.694, 0.708, 0.908],
    [0.715, 0.726, 0.925],
    [0.720, 0.731, 0.927]
]

bp_values = [
    [0.336, 0.280, 0.718],
    [0.458, 0.431, 0.878],
    [0.438, 0.412, 0.853],
    [0.437, 0.410, 0.850],
    [0.451, 0.423, 0.875],
    [0.473, 0.459, 0.876]
]

# 指标名称
labels = ['Fmax', 'AUPR', 'AUC']

# 高对比度颜色列表
colors = ['#FF5733', '#33FF57', '#3357FF', '#FFC300', '#DAF7A6', '#FF33FF']

# 创建雷达图
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 确保闭合

# 绘图
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

# MF 雷达图
for i, method in enumerate(methods):
    values = mf_values[i] + [mf_values[i][0]]  # 闭合
    axs[0].plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', label=method)

axs[0].set_title('Ablation Experiment - MF', fontsize=16)
axs[0].set_xticks(angles[:-1])
axs[0].set_xticklabels(labels)
axs[0].set_ylim(0, 1)  # 根据数据调整 y 轴范围
axs[0].legend(methods, bbox_to_anchor=(1.1, 1.1))

# CC 雷达图
for i, method in enumerate(methods):
    values = cc_values[i] + [cc_values[i][0]]  # 闭合
    axs[1].plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', label=method)

axs[1].set_title('Ablation Experiment - CC', fontsize=16)
axs[1].set_xticks(angles[:-1])
axs[1].set_xticklabels(labels)
axs[1].set_ylim(0, 1)  # 根据数据调整 y 轴范围
axs[1].legend(methods, bbox_to_anchor=(1.1, 1.1))

# BP 雷达图
for i, method in enumerate(methods):
    values = bp_values[i] + [bp_values[i][0]]  # 闭合
    axs[2].plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', label=method)

axs[2].set_title('Ablation Experiment - BP', fontsize=16)
axs[2].set_xticks(angles[:-1])
axs[2].set_xticklabels(labels)
axs[2].set_ylim(0, 1)  # 根据数据调整 y 轴范围
axs[2].legend(methods, bbox_to_anchor=(1.1, 1.1))

# 显示图形
plt.tight_layout()
plt.savefig('table6.png', bbox_inches='tight',dpi=300)
plt.show()
