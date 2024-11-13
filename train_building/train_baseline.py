import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils

# 设置设备为GPU（如果可用），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
def load_data(filepath):
    return pd.read_pickle(filepath)

# 特征工程
def feature_engineering(data):
    data['month_sin'] = np.sin(2 * np.pi * data['month_datetime'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month_datetime'] / 12)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour_datetime'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour_datetime'] / 24)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_week'] / 7)
    return data

# 特征缩放（标准化）
def scale_features(data):
    scaler = StandardScaler()
    features = ['meter_reading', 'air_temperature', 'cloud_coverage', 'wind_speed',
                'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
    scaled_features = scaler.fit_transform(data[features])
    target = scaler.fit_transform(data[['meter_reading']])
    return scaled_features, target

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

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 输出单个预测值

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 使用LSTM的最后一层输出
        return out

# 训练LSTM模型
def train_lstm_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())

            if torch.any(torch.isnan(outputs)):
                print(f"NaN detected in outputs at epoch {epoch}")
                continue

            loss = criterion(outputs, targets.float())

            if torch.isnan(loss):
                print(f"NaN detected in loss at epoch {epoch}")
                continue

            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.float())

                if torch.any(torch.isnan(outputs)):
                    print(f"NaN detected in outputs during validation at epoch {epoch}")
                    continue

                epoch_val_loss += criterion(outputs, targets.float()).item()

        val_losses.append(epoch_val_loss / len(val_loader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_train_loss / len(train_loader):.4f}, Val Loss: {epoch_val_loss / len(val_loader):.4f}')

    return train_losses, val_losses

# 绘制损失曲线
def plot_losses(train_losses, val_losses, num_epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# 预测与评价
def evaluate_lstm_model(model, X_val, y_val):
    model.eval()

    # Move the model to CPU
    model_cpu = model.to('cpu')

    # Move the input to CPU
    test_input = X_val.cpu()

    # Perform inference
    with torch.no_grad():
        predictions = model_cpu(test_input.float()).detach().cpu().numpy()

    # Move the actual values to CPU
    actual = y_val.cpu().numpy()

    return predictions, actual

# 训练和预测其他模型（SVR，RandomForest，AdaBoost）
def train_sklearn_model(model, X_train, y_train):
    model.fit(X_train, y_train.ravel())
    return model

def evaluate_sklearn_model(model, X_val):
    predictions = model.predict(X_val)
    return predictions

# 绘制预测结果
def plot_predictions(actual, predictions, start_idx, end_idx, model_name):
    actual_subset = actual[start_idx:end_idx]
    predictions_subset = predictions[start_idx:end_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(actual_subset, label='Actual meter reading', color='blue')
    plt.plot(predictions_subset, label='Predicted meter reading', color='red', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Meter Reading')
    plt.title(f'Actual vs Predicted Meter Reading ({model_name})')
    plt.legend()
    plt.savefig(f'{model_name}_trueAndPrediction.png')
    plt.show()

    errors_subset = actual_subset - predictions_subset
    plt.figure(figsize=(10, 6))
    plt.plot(errors_subset, label='Prediction Error', color='green')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title(f'Prediction Error Over Time ({model_name})')
    plt.legend()
    plt.savefig(f'{model_name}_errors.png')
    plt.show()

def calculate_metrics(actual, predictions):
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predictions)
    smape = np.mean(2 * np.abs(predictions - actual) / (np.abs(predictions) + np.abs(actual))) * 100
    r, _ = pearsonr(actual.flatten(), predictions.flatten())
    mae = mean_absolute_error(actual, predictions)
    return mse, rmse, mape, smape, r, mae

# 主函数
def main():
    train_rri = pd.read_pickle('../src/train_rri.pkl')
    testing_months = [5, 8, 12]

    # 1.基于月份
    train_data = train_rri[train_rri['month_datetime'].isin(testing_months)]
    test_data = train_rri[~train_rri['month_datetime'].isin(testing_months)]
    # train_data = pd.read_pickle('../src/train_df.pkl')
    # test_data = pd.read_pickle('../src/test_df.pkl')
    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)

    scaled_train_features, train_target = scale_features(train_data)
    scaled_test_features, test_target = scale_features(test_data)

    seq_length = 24
    X_train, y_train = create_sequences(scaled_train_features, train_target, seq_length)
    X_test, y_test = create_sequences(scaled_test_features, test_target, seq_length)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=64)

    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    lstm_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    lstm_model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=1e-4)
    num_epochs = 20

    # train_losses, val_losses = train_lstm_model(lstm_model, train_loader, val_loader, criterion, optimizer, num_epochs)
    # plot_losses(train_losses, val_losses, num_epochs)
    # lstm_predictions, lstm_actual = evaluate_lstm_model(lstm_model, X_test, y_test)
    # plot_predictions(lstm_actual, lstm_predictions, 1000, 1700, 'LSTM')
    # mse, rmse, mape, smape, r, mae = calculate_metrics(lstm_actual.flatten(), lstm_predictions.flatten())
    # print(f'LSTM Metrics: MSE={mse}, RMSE={rmse}, MAPE={mape}, SMAPE={smape}, Pearson r={r}, MAE={mae}')

    # SVR
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_model = train_sklearn_model(svr_model, X_train.reshape(X_train.shape[0], -1), y_train)
    svr_predictions = evaluate_sklearn_model(svr_model, X_test.reshape(X_test.shape[0], -1))
    plot_predictions(y_test.numpy().flatten(), svr_predictions.flatten(), 1000, 1700, 'SVR')
    mse, rmse, mape, smape, r, mae = calculate_metrics(y_test.numpy().flatten(), svr_predictions.flatten())
    print(f'SVR Metrics: MSE={mse}, RMSE={rmse}, MAPE={mape}, SMAPE={smape}, Pearson r={r}, MAE={mae}')

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model = train_sklearn_model(rf_model, X_train.reshape(X_train.shape[0], -1), y_train)
    rf_predictions = evaluate_sklearn_model(rf_model, X_test.reshape(X_test.shape[0], -1))
    plot_predictions(y_test.numpy().flatten(), rf_predictions.flatten(), 1000, 1700, 'RandomForest')
    mse, rmse, mape, smape, r, mae = calculate_metrics(y_test.numpy().flatten(), rf_predictions.flatten())
    print(f'Random Forest Metrics: MSE={mse}, RMSE={rmse}, MAPE={mape}, SMAPE={smape}, Pearson r={r}, MAE={mae}')

    # AdaBoost
    ada_model = AdaBoostRegressor(n_estimators=100)
    ada_model = train_sklearn_model(ada_model, X_train.reshape(X_train.shape[0], -1), y_train)
    ada_predictions = evaluate_sklearn_model(ada_model, X_test.reshape(X_test.shape[0], -1))
    plot_predictions(y_test.numpy().flatten(), ada_predictions.flatten(), 1000, 1700, 'AdaBoost')
    mse, rmse, mape, smape, r, mae = calculate_metrics(y_test.numpy().flatten(), ada_predictions.flatten())
    print(f'AdaBoost Metrics: MSE={mse}, RMSE={rmse}, MAPE={mape}, SMAPE={smape}, Pearson r={r}, MAE={mae}')

if __name__ == '__main__':
    main()
