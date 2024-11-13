import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.utils import dataProcess
# train.csv

# 数据预处理函数
def preprocess_data(df, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()

    df['meter_reading'] = scaler.fit_transform(df[['meter_reading']])
    return df, scaler


# 定义深度学习模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# 读取数据
data = pd.read_csv("../data/data/train.csv")
sub_data = data[(data["building_id"] == 150) & (data["meter"] == 0)]

# 尝试两种预处理方法
for method in ['zscore']:
    print(f"Using {method} scaling:")
    sub_data_processed, scaler = preprocess_data(sub_data, method)

    # 数据集构建
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = dataProcess(sub_data_processed)

    # 定义模型、损失函数和优化器
    input_size = 1
    hidden_size = 50
    num_layers = 1
    model = LSTMModel(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    loss_values = []
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())  # 记录当前损失值
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, color='blue', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs ({method.capitalize()} Scaling)')
    plt.legend()
    plt.savefig(f'lstm_loss_{method}.png')
    plt.show()
    predictions = scaler.inverse_transform(predictions)
    y_test_tensor = scaler.inverse_transform(y_test_tensor)
    # 绘制结果
    plt.figure(figsize=(10, 5))
    plt.plot(predictions[2000:2700], color='green', label=f"{method.capitalize()} LSTM Predictions")
    plt.plot(y_test_tensor[2000:2700], color="r", linestyle=":", label="Real")
    plt.legend()
    plt.title(f"LSTM Predictions with {method.capitalize()} Scaling")
    plt.savefig(f'111lstm_predictions_{method}.png')
    plt.show()
