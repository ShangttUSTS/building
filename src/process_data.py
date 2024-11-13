import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src.utils import reduce_mem_usage
root = '../data/data/'
train = pd.read_csv(root + 'train.csv')
weather_train = pd.read_csv(root + 'weather_train.csv')
weather_test_df = pd.read_csv(root + 'weather_test.csv')
building_meta_df = pd.read_csv(root + 'building_metadata.csv')
test_df = pd.read_csv(root + 'test.csv')
sample_submission = pd.read_csv(root + 'sample_submission.csv')
print('Size of train data', train.shape)
print('Size of weather_train data', weather_train.shape)
print('Size of weather_test_df data', weather_test_df.shape)
print('Size of building_meta_df data', building_meta_df.shape)
# Merge the dataframes on the key fields to get the full dataset
train_full = train.merge(building_meta_df, on='building_id', how='left')
test_full = test_df.merge(building_meta_df, on='building_id', how='left')
train_full = train_full.merge(weather_train, on=['site_id', 'timestamp'], how='left')
test_full = test_full.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
# Reduce memory usage
train = reduce_mem_usage(train)
test_df = reduce_mem_usage(test_df)
train_full = reduce_mem_usage(train_full)
test_full = reduce_mem_usage(test_full)
weather_train = reduce_mem_usage(weather_train)
weather_test_df = reduce_mem_usage(weather_test_df)
building_meta_df = reduce_mem_usage(building_meta_df)
# Converting from string to datetime columns
train["timestamp"] = pd.to_datetime(train["timestamp"], format='%Y-%m-%d %H:%M:%S')
train_full["timestamp"] = pd.to_datetime(train_full["timestamp"])
test_full["timestamp"] = pd.to_datetime(test_full["timestamp"])
# print(train.head)
print(train.timestamp.is_monotonic_increasing)
# print(weather_train.head)
print('Weather data time span: ', weather_train.timestamp.min(), '—', weather_train.timestamp.max())
print('Unique sites: ', weather_train.site_id.unique())


# # Count how many available values (in %) there are for each column, except the target column
# train_data = (train_full.count() / len(train_full)).drop('meter_reading').sort_values().values
# ind = np.arange(len(train_data))
# width = 0.35
#
# fig, axes = plt.subplots(1,1,figsize=(14, 6), dpi=100)
# tr = axes.bar(ind, train_data, width, color='royalblue')
#
# test_data = (test_full.count() / len(test_full)).drop('row_id').sort_values().values
# tt = axes.bar(ind+width, test_data, width, color='seagreen')
#
# axes.set_ylabel('Amount of data available')
# axes.set_xticks(ind + width / 2)
# axes.set_xticklabels((train_full.count() / len(train_full)).drop('meter_reading').sort_values().index, rotation=40)
# axes.legend([tr, tt], ['Train', 'Test'])
#
# train = train.set_index(['timestamp'])
# buildings_count = len(train.building_id.unique())
#
# meters_count = len(train.meter.unique())
# meters = ['electricity', 'chilledwater', 'steam', 'hotwater']
#
# PLOT_MISSING_VALUES = True
#
# # Plot missing values per building/meter
# if PLOT_MISSING_VALUES:
#     f, a = plt.subplots(1, 4, figsize=(20, 30))
#
#     for meter in np.arange(meters_count):
#         # group by the consumed energy type (meter)
#         df = train[train.meter == meter].copy().reset_index()
#
#         # Convert from seconds to hours elapsed since the starting point of the dataset
#         df['timestamp'] = df['timestamp'] - df['timestamp'].min()
#         df['timestamp'] = df['timestamp'].dt.total_seconds() // 3600
#
#         # Generate a matrix of missing and zero values
#         missmap = np.empty((buildings_count, int(df.timestamp.max()) + 1))
#         missmap.fill(np.nan)
#
#         for l in df.values:
#             if l[2] != meter:
#                 continue
#             # l[1] is a building id, l[0] is an hour, l[3] is a meter reading
#             missmap[int(l[1]), int(l[0])] = 0 if l[3] == 0 else 1
#
#         a[meter].set_title(f'{meters[meter]}')
#         sns.heatmap(missmap, cmap='Paired', ax=a[meter], cbar=False)

# meter_electricity_data = train_full[train_full['meter'] == 0]
# del meter_electricity_data['meter']
# meter_chilledwater_data = train_full[train_full['meter'] == 1]
# del meter_chilledwater_data['meter']
# meter_steam_data = train_full[train_full['meter'] == 2]
# del meter_steam_data['meter']
# meter_hotwater_data = train_full[train_full['meter'] == 3]
# del meter_hotwater_data['meter']
train = train_full[train_full.meter == 0]
del train['meter']
# train

unique_buildings = train.building_id.unique()
# unique_buildings = 1413
# Extract time series data for each building from the dataset as a separate dataframe
buildings_slices = [train[train['building_id'] == id].set_index('timestamp')
                    for id in unique_buildings]
print('All buildings with electricity meter: ', len(buildings_slices))
# Define a desired time range with uniform frequency of 1 hour
desired_index = pd.date_range(start=train.timestamp.min(), end=train.timestamp.max(), freq='1H')
buildings_slices_resampled = []
weather_train.set_index('timestamp', inplace=True)

for slice_df in buildings_slices:
    """For each building time series:
        1. Reindex based on the desired time range
        2. Perform resampling (this will introduce NaNs for missing time periods)
        3. Reindex back to the desired time range
    """
    slice_resampled = slice_df.reindex(slice_df.index.union(desired_index)).asfreq('1h').reindex(desired_index)

    # 4. Fill in the NaN values introduced by resampling
    new_rows_mask = slice_resampled.isna().all(axis=1)  # newly inserted rows have all values NaN

    # Pick a sample non-empty row that was originally in the dataset
    original_sample_row = slice_df.iloc[0]

    # Fill in all the building_metadata for the new rows
    slice_resampled.loc[new_rows_mask, 'building_id': 'floor_count'] = original_sample_row.loc['building_id': 'floor_count'].values

    # Fill in all the weather data for the new rows
    slice_resampled.update(weather_train[weather_train['site_id'] == original_sample_row['site_id']])

    # Finish with assigning a NaN value to the meter_reading, in order to interpolate it later
    slice_resampled.loc[new_rows_mask, 'meter_reading'] = np.nan

    buildings_slices_resampled.append(slice_resampled)

train_r = pd.concat(buildings_slices_resampled)

fig, axes = plt.subplots(1, 1, figsize=(14, 6))

train_r.index.floor('d').value_counts().sort_index()\
.plot(ax=axes).set_xlabel('Date', fontsize=14)

axes.set_title('Number of observations by day', fontsize=16);

# Plot a histogram of the missing values rate after resampling

n = len(desired_index)
building_nans = train_r.groupby('building_id')['meter_reading'].apply(lambda x: x.isna().sum() / n)
building_nans.plot(kind='hist', weights=np.ones_like(building_nans) / len(building_nans))

NANS_THRESHOLD = 0

buildings_to_discard = (building_nans[building_nans > NANS_THRESHOLD]).index

# First resampled, now reduced
train_rr = train_r[~train_r['building_id'].isin(buildings_to_discard)]
train_rr.isna().sum()
# Plot the number of NaNs in each column

fill_metanans_strategy = 'DROP'  # 'VIRTUAL', 'MEDIAN'
"""Fill in the stationary metabuilding NaN columns according to 3 strategies:
    1. MEDIAN: impute the NaN values with median values of the year_built and floor_count columns
    2. VIRTUAL: method based on the idea that the absence of the metabuilding info can actually reveal
    something about the building (e.g. that the building is old). In this case, we impute all the NaN
    values with 'virtual' stationary negative value, like -103 (the choice of a value doesn't matter here)
    3. DROP: drop the year_built and floor_count columns
"""

if fill_metanans_strategy == 'VIRTUAL':
    train_rr['floor_count'] = train_rr['floor_count'].fillna(-103)
    train_rr['year_built'] = train_rr['year_built'].fillna(-103)
elif fill_metanans_strategy == 'MEDIAN':
    nans_mask = train_rr.isna()

    floor_nans = nans_mask['floor_count']
    year_nans = nans_mask['year_built']

    train_rr.loc[floor_nans, 'floor_count'] = train_rr['floor_count'].median(skipna=True)
    train_rr.loc[year_nans, 'year_built'] = train_rr['year_built'].median(skipna=True)
elif fill_metanans_strategy == 'DROP':
    train_rr = train_rr.drop(columns=['year_built', 'floor_count'])
else:
    print('Unsupported strategy.')

train_rr.isna().sum()

interp_method = 'linear'

unique_buildings = train_rr.building_id.unique()
features_to_fill = [ 'meter_reading',
                     'air_temperature',
                     'cloud_coverage',
                     'dew_temperature',
                     'precip_depth_1_hr',
                     'sea_level_pressure',
                     'wind_direction',
                     'wind_speed',  ]

building_slices = [train_rr[train_rr['building_id'] == id]
                    for id in unique_buildings]

train_rri = pd.concat([slice.interpolate(method=interp_method, axis='index', extrapolate=True)for slice in building_slices])

# After interpolation, there are still some NaNs left outside the interpolation bounds.
# We'll fill them with the median values

nans_mask = train_rri.isna()
for f in features_to_fill:
    feature_nans = nans_mask[f]

    # Check if there are NaN values before filling
    if feature_nans.any():
        # Calculate and fill the median for the feature
        median_value = train_rri[f].median(skipna=True)
        train_rri.loc[feature_nans, f] = median_value

# Check if there are still NaN values
train_rri.isna().sum()
from numpy.random import choice, seed

seed(42)

""" Optionally drop a portion of the buildings left (to save computational time later)
    If BUILDINGS_DROP_VALUE is less than 1, it specifies a fraction of the buildings to drop (e.g. 0.5)
    If BUILDINGS_DROP_VALUE is bigger (or equal) than 1, it specifices the absolute value of the buildings to leave in the dataset (e.g. 1, 10, ...)"""

BUILDINGS_DROP_VALUE = 0.9

new_buildings = train_rri.building_id.unique()
new_buildings_count = len(new_buildings)

if BUILDINGS_DROP_VALUE < 1:  # drop a fraction of the buildings left
    buildings_drop_count = int(np.ceil(new_buildings_count * BUILDINGS_DROP_VALUE))

    buildings_to_drop = choice(new_buildings, buildings_drop_count, replace=False)
else:  # leave only BUILDINGS_DROP_VALUE buildings
    assert BUILDINGS_DROP_VALUE < new_buildings_count

    buildings_drop_count = new_buildings_count - BUILDINGS_DROP_VALUE
    buildings_to_drop = choice(new_buildings, buildings_drop_count, replace=False)

train_rri = train_rri[~train_rri['building_id'].isin(buildings_to_drop)]
print(f'{len(train_rri.building_id.unique())} buildings left in the dataset.')
if fill_metanans_strategy != 'DROP':
    train_rri['age'] = train_rri['year_built'].max() - train_rri['year_built'] + 1
    train_rri['year_built'] = train_rri['year_built']-1900

train_rri['month_datetime'] = train_rri.index.month.astype(np.int8)
train_rri['hour_datetime'] = train_rri.index.hour.astype(np.int8)
train_rri['day_week'] = train_rri.index.dayofweek.astype(np.int8)
train_rri['day_month_datetime'] = train_rri.index.day.astype(np.int8)

train_rri['square_feet'] = np.log(train_rri['square_feet'])
# Convert to more lightweight types to save some memory
d_types = {'building_id': np.int16,
          'site_id': np.int8,
          'primary_use': 'category',
          'square_feet': np.int32,
          'air_temperature': np.float32,
          'cloud_coverage': np.float16,
          'dew_temperature': np.float32,
          'precip_depth_1_hr': np.float16,
          'sea_level_pressure': np.float32,
          'wind_direction': np.float16,
          'wind_speed': np.float32}
if fill_metanans_strategy != 'DROP':
    d_types['year_built'] = np.float16,
    d_types['floor_count'] = np.float16,

for feature in d_types:
    train_rri[feature] = train_rri[feature].astype(d_types[feature])
# Convert our only categorical column to numeric
from sklearn.preprocessing import LabelEncoder
import numpy as np
primary_uses = list(train_rri['primary_use'].unique())

primary_use_building_counts = train_rri.groupby('primary_use')['building_id'].nunique()
# print("每个 primary_use 中不同 building_id 的数量：")
print(primary_use_building_counts)
le = LabelEncoder()
train_rri['primary_use'] = le.fit_transform(train_rri['primary_use']).astype(np.int8)
train_rri.to_pickle('train_rri_electricity.pkl')
# 查看对应关系
# primary_use=building_meta_df['primary_use'].unique()
# primary_use_transform = le.fit_transform(primary_use).astype(np.int8)
# label_mapping = dict(zip(primary_use, primary_use_transform))
# print("Label Mapping:")
# for k, v in label_mapping.items():
#     print(f"{k}: {v}")

testing_months = [5, 8, 12]

test_df = train_rri[train_rri['month_datetime'].isin(testing_months)]
train_df = train_rri[~train_rri['month_datetime'].isin(testing_months)]

print('Training dataset length:', len(train_df))
print('Test dataset length:', len(test_df))

# Save the datasets to be used later...
train_df.to_pickle('train_df.pkl')
test_df.to_pickle('test_df.pkl')