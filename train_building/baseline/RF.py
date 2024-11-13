import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


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
    return np.array(sequences), np.array(labels)


# 训练SVR模型并进行预测
def train_and_predict_svr(X_train, y_train, X_test):
    # 扁平化输入特征以适应SVR输入格式
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    svr = SVR(kernel='rbf')
    svr.fit(X_train_flat, y_train.flatten())

    predictions = svr.predict(X_test_flat)
    return predictions


# 计算评价指标
def calculate_metrics(actual, predictions):
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predictions)
    smape = np.mean(2 * np.abs(predictions - actual) / (np.abs(predictions) + np.abs(actual))) * 100
    r, _ = pearsonr(actual.flatten(), predictions.flatten())
    mae = mean_absolute_error(actual, predictions)
    return mse, rmse, mape, smape, r, mae

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

# 主函数
def main():
    train_data = pd.read_pickle('../../src/train_df.pkl')
    test_data = pd.read_pickle('../../src/test_df.pkl')

    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)

    scaled_train_features, train_target = scale_features(train_data)
    scaled_test_features, test_target = scale_features(test_data)

    seq_length = 24
    X_train, y_train = create_sequences(scaled_train_features, train_target, seq_length)
    X_test, y_test = create_sequences(scaled_test_features, test_target, seq_length)

    # 训练SVR模型并进行预测
    predictions = train_and_predict_svr(X_train, y_train, X_test)

    # 计算评价指标
    mse, rmse, mape, smape, r, mae = calculate_metrics(y_test, predictions)
    print(f'MSE: {mse}, RMSE: {rmse}, MAPE: {mape}, sMAPE: {smape}, Pearson R: {r}, MAE: {mae}')

    # 绘制预测结果
    plot_predictions(y_test, predictions, 2300, 2700, 'SVR')


if __name__ == "__main__":
    main()
