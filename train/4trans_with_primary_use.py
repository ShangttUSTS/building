import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
# Set up logging
logging.basicConfig(level=logging.INFO,  # Default log level is INFO
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training_log.txt"),  # Log to a file
                              logging.StreamHandler()])  # Also log to console

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载源数据 (train_df.pkl) 和目标数据 (test_df.pkl)
data = pd.read_pickle('../src/with_primary_decode_train_rri.pkl')
train_data = data[(data["primary_use"] == 0)]
test_data = data[(data["primary_use"] == 7)]
scaler = StandardScaler()
# train_data = 0
# primary_use = 1——MSE (Finetuned): 0.0262  RMSE (Finetuned): 0.1614
#primary_use = 2——MSE (Finetuned): 0.3130 RMSE (Finetuned): 0.5594
# primary_use = 3——MSE (Finetuned): 0.0135 RMSE (Finetuned): 0.1162
# primary_use = 4——MSE (Finetuned): 0.0842  RMSE (Finetuned): 0.2902
#primary_use = 5——MSE (Finetuned): 0.0106 RMSE (Finetuned): 0.1031
# primary_use = 6——MSE (Finetuned): 0.0569 RMSE (Finetuned): 0.2385
# primary_use = 7——MSE (Finetuned): 0.0077 RMSE (Finetuned): 0.0876
# 特征处理函数
def preprocess_data(data):
    # 生成季节性特征
    data['month_sin'] = np.sin(2 * np.pi * data['month_datetime'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month_datetime'] / 12)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour_datetime'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour_datetime'] / 24)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_week'] / 7)

    # 特征缩放（标准化）

    scaled_features = scaler.fit_transform(data[['meter_reading', 'air_temperature', 'cloud_coverage', 'wind_speed',
                                                 'month_sin', 'month_cos', 'hour_sin', 'hour_cos','primary_use',
                                                 'day_of_week_sin', 'day_of_week_cos']])

    # 构建目标变量（标签），假设 meter_reading 是要预测的目标
    target = scaler.fit_transform(data[['meter_reading']])

    return scaled_features, target

# 处理源数据
source_features, source_target = preprocess_data(train_data)

# 处理目标数据
target_features, target_target = preprocess_data(test_data)

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
X_source, y_source = create_sequences(source_features, source_target, seq_length)
X_target, y_target = create_sequences(target_features, target_target, seq_length)

# 划分训练集和验证集
train_size = int(0.8 * len(X_source))
X_source_train, X_source_val = X_source[:train_size], X_source[train_size:]
y_source_train, y_source_val = y_source[:train_size], y_source[train_size:]

train_data = TensorDataset(X_source_train, y_source_train)
val_data = TensorDataset(X_source_val, y_source_val)

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
input_size = X_source.shape[2]  # 特征数
hidden_size = 64  # LSTM隐层大小
num_layers = 2  # LSTM层数
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
model.to(device)  # 将模型移动到GPU

# 训练设置
criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 更小的学习率

# Early stopping parameters
early_stop_patience = 10
best_val_loss = float('inf')
patience_counter = 0

# Model checkpoint path
checkpoint_path = 'best_model.pth'

# 训练设置
num_epochs = 500  # 训练轮数
train_losses = []
val_losses = []
def main(load_model=False):
    if load_model and os.path.exists(checkpoint_path):
        # Load the pretrained model
        model.load_state_dict(torch.load(checkpoint_path))
        logging.info("Model loaded from checkpoint.")
    else:
        # 训练过程
        best_val_loss = 1000
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                # 将数据移动到GPU
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs.float())

                loss = criterion(outputs, targets.float())
                loss.backward()
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
                    epoch_val_loss += criterion(outputs, targets.float()).item()

            val_losses.append(epoch_val_loss / len(val_loader))  # 记录验证损失

            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_train_loss / len(train_loader):.4f}, Val Loss: {epoch_val_loss / len(val_loader):.4f}')

            # Early stopping check
            if epoch_val_loss < best_val_loss:
                # print(best_val_loss)
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)  # Save best model
                logging.info(f"Model saved at epoch {epoch + 1}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logging.info("Early stopping triggered.")
                    break
        # 绘制训练损失和验证损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('Training_and_Validation_Loss.png')
        plt.show()

    # 微调模型
    finetune_model()

def finetune_model():
    # 使用源数据训练好的模型进行目标数据微调
    target_train_data = TensorDataset(X_target, y_target)
    target_loader = DataLoader(target_train_data, batch_size=64, shuffle=True)

    # 微调模型
    num_epochs_finetune = 20  # 微调轮数

    for epoch in range(num_epochs_finetune):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in target_loader:
            # 将数据移动到GPU
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.float())

            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        logging.info(f'Epoch [{epoch + 1}/{num_epochs_finetune}], Finetune Loss: {epoch_train_loss / len(target_loader):.4f}')

    # 微调后的模型评估
    model.eval()
    test_input = X_target.to(device)  # 将目标数据移动到GPU
    predictions = model(test_input.float())

    # 将预测结果与实际值进行比较
    predictions = predictions.detach().cpu().numpy()  # 转换为numpy数组，并移回CPU
    actual = y_target.numpy()

    def calculate_metrics(actual, predictions):
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actual, predictions)
        smape = np.mean(2 * np.abs(predictions - actual) / (np.abs(predictions) + np.abs(actual))) * 100
        r, _ = pearsonr(actual.flatten(), predictions.flatten())
        mae = mean_absolute_error(actual, predictions)
        return mse, rmse, mape, smape, r, mae

    mse, rmse, mape, smape, r, mae = calculate_metrics(actual, predictions)
    print(f'Pearson Correlation (Finetuned): {r:.4f}')
    print(f'SMAPE (Finetuned): {smape:.4f}')
    print(f'MAE (Finetuned): {mae:.4f}')
    print(f'RMSE (Finetuned): {rmse:.4f}')
    # print(f'MSE (Finetuned): {mse:.4f}')

    # print(f'MAPE (Finetuned): {mape:.4f}')



    # 计算MSE和RMSE
    # mse = mean_squared_error(actual, predictions)
    # rmse = np.sqrt(mse)
    #
    # logging.info(f'MSE (Finetuned): {mse:.4f}')
    # logging.info(f'RMSE (Finetuned): {rmse:.4f}')
    # predictions = scaler.inverse_transform(predictions)
    # actual = scaler.inverse_transform(actual)
    # 绘制预测和实际结果对比图
    plt.figure(figsize=(10, 6))
    plt.plot(actual[4000:5000], label='Actual')
    plt.plot(predictions[4000:5000], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Meter Reading')
    plt.title('Prediction vs Actual (Finetuned Model)')
    plt.legend()
    plt.savefig('6Prediction_vs_Actual.png')
    plt.show()

if __name__ == "__main__":
    main(load_model=True)  # 设置为True加载已有模型进行微调
