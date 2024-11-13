import pandas as pd
data = pd.read_pickle('with_primary_decode_train_rri.pkl')


primary_uses = list(data['primary_use'].unique())

primary_use_building_counts = data.groupby('primary_use')['building_id'].nunique()
# print("每个 primary_use 中不同 building_id 的数量：")
print(primary_use_building_counts)
print("各个 primary_use 及其对应的 building_id 数量：")

# 各个 primary_use 及其对应的 building_id 数量：
# primary_use
# 0    15
# 1     5
# 2     1
# 3     4
# 4     2
# 5    11
# 6     1
# 7     3

# primary_use
# Education                        15
# Entertainment/public assembly     5
# Healthcare                        1
# Lodging/residential               4
# Manufacturing/industrial          2
# Office                           11
# Other                             1
# Public services                   3
