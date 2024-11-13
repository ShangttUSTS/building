import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 根据primary_use进行迁移的
# 效果不好

# 加载数据
data = pd.read_pickle('../src/train_df.pkl')

# 标准化特征
scaler = StandardScaler()
features = ['site_id', 'square_feet', 'air_temperature', 'cloud_coverage', 'dew_temperature','meter_reading',
            'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'month_datetime', 'hour_datetime', 'day_week', 'day_month_datetime']
data[features] = scaler.fit_transform(data[features])

features = ['site_id', 'primary_use', 'square_feet', 'air_temperature', 'cloud_coverage', 'dew_temperature',
            'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'month_datetime', 'hour_datetime', 'day_week', 'day_month_datetime']


# 准备LSTM数据
def create_sequences(df, target, sequence_length=21):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length][features].values)
        y.append(df.iloc[i+sequence_length][target])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

sequence_length = 21

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_dim = len(features)
hidden_dim = 50
output_dim = 1
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 针对每个 primary_use 训练基础模型
primary_uses = list(data['primary_use'].unique())

primary_use_building_counts = data.groupby('primary_use')['building_id'].nunique()
# print("每个 primary_use 中不同 building_id 的数量：")
print(primary_use_building_counts)
primary_use_counts = data.groupby('primary_use')['building_id'].nunique()
PRIMARY_USE_min = primary_use_counts.idxmin()
PRIMARY_USE_max = primary_use_counts.idxmax()
print(f"最少的primary_use是: {PRIMARY_USE_min}, 数量是: {primary_use_counts[PRIMARY_USE_min]}")
print(f"最多的primary_use是: {PRIMARY_USE_max}, 数量是: {primary_use_counts[PRIMARY_USE_max]}")
orgin = PRIMARY_USE_max
target = PRIMARY_USE_min
# print(primary_uses)
base_models = {}

orgin_data = data[data['primary_use'] == orgin]
X, y = create_sequences(orgin_data, 'meter_reading', sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_values = []
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())  # 记录当前损失值
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


import matplotlib.pyplot as plt

# 初始化一个字典来存储每个模型的测试损失和预测结果

# 迁移学习新的 primary_use
new_pu_data = data[data['primary_use'] == target]
X, y = create_sequences(new_pu_data, 'meter_reading', sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# 初始化迁移模型
# transfer_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
transfer_model = model
# 加载相关 primary_use 的模型参数
# transfer_model.load_state_dict(base_models[orgin])

# 微调
criterion = nn.MSELoss()
optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)

transfer_model.train()
for epoch in range(10):  # 微调轮数
    optimizer.zero_grad()
    outputs = transfer_model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()

# 评估模型
transfer_model.eval()
with torch.no_grad():
    outputs = transfer_model(X_test.to(device))
    loss = criterion(outputs, y_test.to(device))
    print(f' Loss: {loss.item():.4f}')

    # 保存预测结果和真实标签
    outputs.cpu().numpy().flatten()
    y_test_values = y_test.numpy()

    # 绘制预测结果和真实值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_values, label='Real', color='blue')
    plt.plot(outputs.cpu().numpy().flatten(), label='Predictions', color='orange')
    plt.title(f'Model {orgin} Predictions Vs Real')
    plt.xlabel('Index')
    plt.ylabel('Electricity')
    plt.legend()
    plt.grid()
    plt.savefig('1trans_building_primary_use.png')
    plt.show()
