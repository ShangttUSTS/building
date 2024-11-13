

from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
root = '../data/data/'
building_meta_df = pd.read_csv(root + 'building_metadata.csv')
le = LabelEncoder()
primary_use=building_meta_df['primary_use'].unique()
primary_use_transform = le.fit_transform(primary_use).astype(np.int8)
building_meta_df['primary_use'] = le.fit_transform(primary_use).astype(np.int8)

# 查看对应关系
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_mapping)