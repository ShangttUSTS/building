import pandas as pd
import numpy as np
import torch

# 定义高斯核函数
def gaussian_kernel_matrix(x, y, sigma):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)

    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / (2 * sigma))

# 定义分批计算的 MMD 计算函数
def mmd_batch(x, y, sigma=1.0, batch_size=100):
    num_batches = len(x) // batch_size + (1 if len(x) % batch_size != 0 else 0)
    mmd_values = []
    for i in range(num_batches):
        for j in range(num_batches):
            x_batch = x[i*batch_size:(i+1)*batch_size]
            y_batch = y[j*batch_size:(j+1)*batch_size]
            xx = gaussian_kernel_matrix(x_batch, x_batch, sigma)
            yy = gaussian_kernel_matrix(y_batch, y_batch, sigma)
            xy = gaussian_kernel_matrix(x_batch, y_batch, sigma)
            mmd_values.append(torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy))
    return sum(mmd_values) / len(mmd_values)

# 加载数据
train_rri = pd.read_pickle('train_rri.pkl')

# 将 primary_use 进行编码
primary_use_encoder = {use: idx for idx, use in enumerate(train_rri['primary_use'].unique())}
train_rri['primary_use_encoded'] = train_rri['primary_use'].map(primary_use_encoder)

# 按 primary_use 分组，获取特征向量
grouped_data = train_rri.groupby('primary_use')

# 计算每个 primary_use 的特征向量
primary_use_vectors = {}
for name, group in grouped_data:
    # features = group.drop(['primary_use', 'building_id', 'primary_use_encoded'], axis=1).values
    features = group[['meter_reading']].values  # 只使用 meter_reading 作为特征
    primary_use_vectors[name] = torch.tensor(features, dtype=torch.float32)

# 计算不同 primary_use 之间的相似度
primary_use_keys = list(primary_use_vectors.keys())
similarity_matrix = np.zeros((len(primary_use_keys), len(primary_use_keys)))

for i in range(len(primary_use_keys)):
    for j in range(i, len(primary_use_keys)):
        if i != j:
            x = primary_use_vectors[primary_use_keys[i]]
            y = primary_use_vectors[primary_use_keys[j]]
            similarity_matrix[i][j] = mmd_batch(x, y, sigma=1.0, batch_size=100).item()
            similarity_matrix[j][i] = similarity_matrix[i][j]
        else:
            similarity_matrix[i][j] = 0  # 自相似度为0

# 打印相似度矩阵
print("不同 primary_use 之间的相似度（MMD）：")
for i, key1 in enumerate(primary_use_keys):
    for j, key2 in enumerate(primary_use_keys):
        print(f"{key1} 和 {key2} 的相似度: {similarity_matrix[i][j]:.4f}")
