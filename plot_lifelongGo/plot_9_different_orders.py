import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

# 数据准备
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

# 创建热图
plt.figure(figsize=(18, 5))

# Fmax 热图
fmax_data = [[mf_data[i, 0], cc_data[i, 0], bp_data[i, 0]] for i in range(len(methods))]
plt.subplot(1, 3, 1)
sns.heatmap(fmax_data, annot=True, fmt=".3f", cmap='Blues',
             norm=Normalize(vmin=min(min(row) for row in fmax_data),
                            vmax=max(max(row) for row in fmax_data)),
             xticklabels=['MF', 'CC', 'BP'], yticklabels=methods)
plt.title('Fmax Comparison', fontsize=16)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Methods', fontsize=12)

# AUPR 热图
aupr_data = [[mf_data[i, 1], cc_data[i, 1], bp_data[i, 1]] for i in range(len(methods))]
plt.subplot(1, 3, 2)
sns.heatmap(aupr_data, annot=True, fmt=".3f", cmap='Blues',
             norm=Normalize(vmin=min(min(row) for row in aupr_data),
                            vmax=max(max(row) for row in aupr_data)),
             xticklabels=['MF', 'CC', 'BP'], yticklabels=methods)
plt.title('AUPR Comparison', fontsize=16)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Methods', fontsize=12)

# AUC 热图
auc_data = [[mf_data[i, 2], cc_data[i, 2], bp_data[i, 2]] for i in range(len(methods))]
plt.subplot(1, 3, 3)
sns.heatmap(auc_data, annot=True, fmt=".3f", cmap='Blues',
             norm=Normalize(vmin=min(min(row) for row in auc_data),
                            vmax=max(max(row) for row in auc_data)),
             xticklabels=['MF', 'CC', 'BP'], yticklabels=methods)
plt.title('AUC Comparison', fontsize=16)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Methods', fontsize=12)

plt.tight_layout()
plt.savefig('table9_results.png', bbox_inches='tight',dpi=300)
plt.show()
