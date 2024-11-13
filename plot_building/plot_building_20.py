import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_pickle('../src/with_primary_decode_train_rri.pkl')
# 过滤出 building_id = 20 的数据
building_data = data[data['building_id'] == 20]

# 确保 'timestamp' 列为 datetime 类型（如果有时间戳列的话）
building_data['timestamp'] = pd.to_datetime(building_data.index.tolist())

# 按时间排序
building_data = building_data.sort_values(by='timestamp')

# 提取季节性特征
building_data['month'] = building_data['timestamp'].dt.month
building_data['hour'] = building_data['timestamp'].dt.hour
building_data['day_of_week'] = building_data['timestamp'].dt.dayofweek

# 创建小时的周期性特征
building_data['hour_sin'] = np.sin(2 * np.pi * building_data['hour'] / 24)
building_data['hour_cos'] = np.cos(2 * np.pi * building_data['hour'] / 24)

# 创建月份的周期性特征
building_data['month_sin'] = np.sin(2 * np.pi * building_data['month'] / 12)
building_data['month_cos'] = np.cos(2 * np.pi * building_data['month'] / 12)

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2行2列子图

# 第一个子图：按时间绘制电力需求
axs[0, 0].plot(building_data['timestamp'], building_data['meter_reading'], label='Meter Reading', color='tab:blue')
axs[0, 0].set_xlabel('Timestamp')
axs[0, 0].set_ylabel('Meter Reading')
axs[0, 0].set_title('Meter Reading Over Time for Building 20')
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].grid(True)
axs[0, 0].legend()

# 第二个子图：按月份绘制平均电力需求
monthly_avg = building_data.groupby('month')['meter_reading'].mean()
axs[0, 1].bar(monthly_avg.index, monthly_avg, color='skyblue')
axs[0, 1].set_xlabel('Month')
axs[0, 1].set_ylabel('Average Meter Reading')
axs[0, 1].set_title('Average Monthly Electricity Consumption (Building 20)')
axs[0, 1].tick_params(axis='x', rotation=0)

# 第三个子图：按小时绘制电力需求的平均值
hourly_avg = building_data.groupby('hour')['meter_reading'].mean()
axs[1, 0].plot(hourly_avg.index, hourly_avg, color='green')
axs[1, 0].set_xlabel('Hour of Day')
axs[1, 0].set_ylabel('Average Meter Reading')
axs[1, 0].set_title('Average Hourly Electricity Consumption (Building 20)')
axs[1, 0].grid(True)

# 第四个子图：按天绘制电力需求的平均值（可选，如果需要绘制）
daily_avg = building_data.groupby('day_of_week')['meter_reading'].mean()
axs[1, 1].plot(daily_avg.index, daily_avg, marker='o', color='orange')
axs[1, 1].set_xlabel('Day of Week')
axs[1, 1].set_ylabel('Average Meter Reading')
axs[1, 1].set_title('Average Daily Electricity Consumption (Building 20)')
axs[1, 1].grid(True)

# 调整布局
plt.tight_layout()
plt.savefig('Building_20_data_process')
plt.show()
