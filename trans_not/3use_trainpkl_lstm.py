import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
# 使用train区分训练集和预测集
# 使用直接使用lstm
# 效果还行

# 设置设备为GPU（如果可用），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设你的数据已经加载到DataFrame中
data = pd.read_pickle('../src/with_primary_decode_train_rri.pkl')

# 生成历史用电量的滑动平均值
# data['meter_reading_3hr_avg'] = data['meter_reading'].rolling(window=3).mean()
# data['meter_reading_24hr_avg'] = data['meter_reading'].rolling(window=24).mean()

# 生成季节性特征
data['month_sin'] = np.sin(2 * np.pi * data['month_datetime'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month_datetime'] / 12)

data['hour_sin'] = np.sin(2 * np.pi * data['hour_datetime'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour_datetime'] / 24)

data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_week'] / 7)
data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_week'] / 7)

# # 添加滞后特征
# data['meter_reading_lag_1'] = data['meter_reading'].shift(1)
# data['meter_reading_lag_24'] = data['meter_reading'].shift(24)

# 特征缩放（标准化）
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['meter_reading', 'air_temperature', 'cloud_coverage', 'wind_speed',
                                             'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
                                             'day_of_week_sin', 'day_of_week_cos']])

# 构建目标变量（标签），假设 meter_reading 是要预测的目标
target = scaler.fit_transform(data[['meter_reading']])


# 创建时间序列数据（输入特征和标签）
def create_sequences(features, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(features) - seq_length):
        seq = features[i:i + seq_length]
        label = target[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences), torch.tensor(labels)


seq_length = 24  # 使用过去24小时的数据来预测
X, y = create_sequences(scaled_features, target, seq_length)

# 划分训练集和验证集
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)


# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 输出单个预测值

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 使用LSTM的最后一层输出
        return out


# 初始化模型
input_size = X.shape[2]  # 特征数
hidden_size = 64  # LSTM隐层大小
num_layers = 2  # LSTM层数
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
model.to(device)  # 将模型移动到GPU

# 训练设置
criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 更小的学习率


num_epochs = 20  # 训练轮数

train_losses = []
val_losses = []

# 训练过程
import torch.nn.utils as nn_utils

# 训练过程
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for inputs, targets in train_loader:
        # 将数据移动到GPU
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.float())

        # 防止损失为NaN
        if torch.any(torch.isnan(outputs)):
            print(f"NaN detected in outputs at epoch {epoch}")
            continue

        loss = criterion(outputs, targets.float())

        # 检查损失是否为NaN
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch}")
            continue

        loss.backward()

        # 梯度裁剪
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()
        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / len(train_loader))  # 记录训练损失

    # 验证过程
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            # 将数据移动到GPU
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())

            if torch.any(torch.isnan(outputs)):
                print(f"NaN detected in outputs during validation at epoch {epoch}")
                continue

            epoch_val_loss += criterion(outputs, targets.float()).item()

    val_losses.append(epoch_val_loss / len(val_loader))  # 记录验证损失

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_train_loss / len(train_loader):.4f}, Val Loss: {epoch_val_loss / len(val_loader):.4f}')

# 绘制训练损失和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 使用模型进行预测
model.eval()
test_input = X_val.to(device)  # 将验证集数据移动到GPU
predictions = model(test_input.float())

# 将预测结果与实际值进行比较
predictions = predictions.detach().cpu().numpy()  # 转换为numpy数组，并移回CPU
actual = y_val.numpy()

# 计算MSE和RMSE
mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)

print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}')

# 假设 `actual` 和 `predictions` 是从训练集或验证集的输出中获取的
# 在这里我们选择从索引 2300 到 3700 的数据进行绘制

start_idx = 2300
end_idx = 2700

# 提取对应范围的数据
actual_subset = actual[start_idx:end_idx]
predictions_subset = predictions[start_idx:end_idx]

# 绘制实际值与预测值的对比图
plt.figure(figsize=(10, 6))
plt.plot(actual_subset, label='Actual meter reading', color='blue')
plt.plot(predictions_subset, label='Predicted meter reading', color='red', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Meter Reading')
plt.title('Actual vs Predicted Meter Reading (2300-3700)')
plt.legend()
plt.savefig('3trueAndPrediction.png')
plt.show()

# 计算并显示预测误差
errors_subset = actual_subset - predictions_subset

# 绘制预测误差
plt.figure(figsize=(10, 6))
plt.plot(errors_subset, label='Prediction Error', color='green')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Prediction Error Over Time (2300-3700)')
plt.legend()
plt.savefig('3errors.png')
plt.show()

