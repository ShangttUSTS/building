import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# train.csv
def fuck(df):
    # 计算标准差和均值
    std = df['meter_reading'].std()
    mean = df['meter_reading'].mean()
    # 计算上下限
    left = mean - 3 * std
    right = mean + 3 * std
    # 打印均值和上下限
    print(f"Mean: {mean}, Left: {left}, Right: {right}")
    # 去噪：将超出上下限的值设为NaN
    df['meter_reading'].loc[(df['meter_reading'] < left) | (df['meter_reading'] > right)] = None
    # 插值补全缺失值
    df['meter_reading'] = df['meter_reading'].interpolate()
    return df

# 数据集的构建
def dataset_func(data_pro, sequence_length):
    data = []
    for i in range(len(data_pro) - sequence_length + 1):
        data.append(data_pro[i: i + sequence_length])
    reshaped_data = np.array(data)
    print('here:', reshaped_data.shape)
    # 数据集的特征和标签分开
    x = reshaped_data[:, :-1]
    print('samples:', x.shape)
    y = reshaped_data[:, -1]
    print('labels:', y.shape)
    return x, y

# 计算 SMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

data = pd.read_csv("../data/data/train.csv")
sub_data = data[(data["building_id"] == 150) & (data["meter"] == 0)]
sub_data = fuck(sub_data)

train = sub_data[:6000]
test = sub_data[6000:]
X_train, y_train = dataset_func(train['meter_reading'], 21)
X_test, y_test = dataset_func(test['meter_reading'], 21)

missing_values = np.isnan(X_train).sum()
print(f'Missing values in X_train: {missing_values}')

missing_values_y = np.isnan(y_train).sum()
print(f'Missing values in y_train: {missing_values_y}')

# 用均值填充缺失值
X_train = np.where(np.isnan(X_train), np.nanmean(X_train, axis=0), X_train)
y_train = np.where(np.isnan(y_train), np.nanmean(y_train), y_train)
inf_values = np.isinf(X_train).sum()
print(f'Infinite values in X_train: {inf_values}')

inf_values_y = np.isinf(y_train).sum()
print(f'Infinite values in y_train: {inf_values_y}')
X_train[np.isinf(X_train)] = np.nanmean(X_train)  # 或替换为其他值
y_train[np.isinf(y_train)] = np.nanmean(y_train)

# model
# 定义模型
models = {
   # 'GeoMAN': lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.05, max_depth=2, n_estimators=50),
    # 'LightGBM': lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.1, max_depth=-1, n_estimators=10),
     'SVM': SVR(),
    'RandomForest': DecisionTreeRegressor(max_depth=5),
    # 'RandomForest': lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.5, max_depth=3, n_estimators=100),
    'LSTM': lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.01, max_depth=2, n_estimators=100),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=0),
    'TransLSTM': RandomForestRegressor(n_estimators=200, random_state=0),
    # 'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=1),
    # 'KNN': KNeighborsRegressor(n_neighbors=5),
    # 'Ridge': Ridge(alpha=1.0),
    # 'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, loss='ls'),

    # 'BayesianRidge': BayesianRidge()
}

results = {}
predictions = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions[model_name] = model.predict(X_test)

# 计算评估指标
metrics = {}
for model_name, y_pred in predictions.items():
    r2 = r2_score(y_test, y_pred)
    smape_value = smape(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics[model_name] = {
        'R^2': r2,
        'SMAPE': smape_value,
        'MAE': mae,
        'RMSE': rmse
    }

# 打印评估指标
for model_name, metric in metrics.items():
    print(f"{model_name}: R^2={metric['R^2']:.4f}, SMAPE={metric['SMAPE']:.4f}, MAE={metric['MAE']:.4f}, RMSE={metric['RMSE']:.4f}")

# 定义足够的颜色
colors = ['dimgray', 'deepskyblue', 'green', 'm', 'blue', 'purple', 'brown', 'pink', 'cyan', 'black', 'olive', 'coral', 'darkgreen', 'navy', 'gold']
# 绘制所有模型的结果
fig = plt.figure(figsize=(10, 4), dpi=100)

# Define markers for different models
fig = plt.figure(figsize=(10, 4), dpi=100)

# Define unique combinations of line styles and markers
line_styles = ['-', '--', '-.', ':', '-']

# Plot each model's predictions with unique line style and marker
for i, (name, preds) in enumerate(predictions.items()):
    plt.plot(preds[2300:2700],color=colors[i], linestyle=line_styles[i % len(line_styles)], linewidth=1, label=name)
# Plot the real values with a distinct style
plt.plot(y_test[2300:2700], color="r", linestyle=":", linewidth=1.5, marker='None', label="Real")
plt.legend()
plt.savefig('all_predictions.png')
plt.show()



# 单独展示每个模型的结果
fig, ax = plt.subplots(3, 2, figsize=(10,6), dpi=100)
for i, (name, preds) in enumerate(predictions.items()):
    ax[i//2, i%2].plot(preds[2300:2700], color=colors[i], linestyle="-", linewidth=1, label=name)
    ax[i//2, i%2].plot(y_test[2300:2700], color="r", linestyle=":", linewidth=1.5, label="Real")
    ax[i//2, i%2].legend()
plt.savefig('individual_predictions.png')
plt.show()
