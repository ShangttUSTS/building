import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 初始化标准化器
scaler = StandardScaler()

# 选择1月份和 primary_use 为 0 的数据
train_rri = pd.read_pickle('train_rri.pkl')
january_data = train_rri[(train_rri['month_datetime'] == 1)]

january_data['meter_reading'] = scaler.fit_transform(january_data[['meter_reading']])

grouped_data = january_data.groupby('primary_use')

# 定义高斯核函数
def gaussian_kernel_matrix(x, y, sigma):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)

    result = torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / (2 * sigma))

    # 添加调试输出
    if torch.isnan(result).any():
        print("Found NaN in Gaussian Kernel Matrix computation.")

    return result

# 定义分批计算的 MMD 计算函数
def mmd_batch(x, y, sigma=1.0, batch_size=100):
    num_batches = len(x) // batch_size + (1 if len(x) % batch_size != 0 else 0)
    mmd_values = []
    for i in range(num_batches):
        for j in range(num_batches):
            x_batch = x[i * batch_size:(i + 1) * batch_size]
            y_batch = y[j * batch_size:(j + 1) * batch_size]
            xx = gaussian_kernel_matrix(x_batch, x_batch, sigma)
            yy = gaussian_kernel_matrix(y_batch, y_batch, sigma)
            xy = gaussian_kernel_matrix(x_batch, y_batch, sigma)

            # 添加调试输出
            # if torch.isnan(xx).any() or torch.isnan(yy).any() or torch.isnan(xy).any():
            #     print(f"NaN detected in Gaussian kernel matrices for batches i={i}, j={j}")
            #     print(f"xx: {xx}")
            #     print(f"yy: {yy}")
            #     print(f"xy: {xy}")

            mmd_value = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
            # 如果 mmd_value 为 NaN，则设置为 1
            if torch.isnan(mmd_value).any():
                mmd_value = torch.tensor(1.0)
            # if torch.isnan(mmd_value).any():
            #     print(f"NaN detected in MMD computation for batches i={i}, j={j}")
            #     print(f"x_batch: {x_batch}")
            #     print(f"y_batch: {y_batch}")

            mmd_values.append(mmd_value)
    return sum(mmd_values) / len(mmd_values)

building_id_vectors = {}

for name, group in grouped_data:
    first_building_data = group[group['building_id'] == group['building_id'].iloc[0]]
    features = first_building_data[['meter_reading']].values
    building_id_vectors[name] = torch.tensor(features, dtype=torch.float32)

building_id_keys = list(building_id_vectors.keys())
similarity_matrix = np.zeros((len(building_id_keys), len(building_id_keys)))

for i in range(len(building_id_keys)):
    for j in range(i, len(building_id_keys)):
        if i != j:
            x = building_id_vectors[building_id_keys[i]]
            y = building_id_vectors[building_id_keys[j]]
            sigma_value = 0.1  # 使用较小的 sigma 值
            similarity_matrix[i][j] = mmd_batch(x, y, sigma=sigma_value, batch_size=100).item()
            similarity_matrix[j][i] = similarity_matrix[i][j]
        else:
            similarity_matrix[i][j] = 0  # 自相似度为0
# 教育、娱乐/公众集会、医疗保健、住宿/住宅、制造业/工业、办公室、其他、公共服务
# 生成热图
building_id_keys = ["教育", "娱乐/公众集会", "医疗保健", "住宿/住宅", "制造业/工业", "办公室", "其他", "公共服务"]
# 设置字体为 SimHei 以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
from plot_building.hotmat import heatmap,annotate_heatmap
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

im, cbar = heatmap(similarity_matrix, building_id_keys, building_id_keys, ax=ax, cbar_kw={"shrink": 0.7},
                   cmap="coolwarm", cbarlabel="")
texts = annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
# plt.title('Month2 With different Primary Use MMD')
# plt.xlabel('Primary Use')
# plt.ylabel('Primary Use')
plt.savefig('Primary_Use_Feature_Heatmap.png')
plt.show()
# plt.figure(figsize=(10, 10),dpi=100)
# sns.heatmap(similarity_matrix, xticklabels=building_id_keys, yticklabels=building_id_keys, cmap='viridis', annot=True, fmt=".4f")
# plt.title('Building20 With different Month MMD')
# plt.xlabel('Month')
# plt.ylabel('Month')
# plt.savefig('building_id_20_Feature_Heatmap')
# plt.show()
