import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from hotmat import heatmap,annotate_heatmap
# 假设你的数据已经加载到DataFrame中
data = pd.read_pickle('../src/with_primary_decode_train_rri.pkl')
data = data[(data["building_id"] ==54)]
# 去掉 'building_id' 和 'site_id' 这两列
data_filtered = data.drop(columns=['building_id', 'site_id', 'square_feet', 'primary_use'])

# 计算数据特征之间的相关性矩阵
feature = data_filtered.columns.tolist()
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
correlation_matrix = data_filtered.corr()
im, cbar = heatmap(correlation_matrix, feature, feature, ax=ax, cbar_kw={"shrink": 0.7},
                   cmap="YlGn", cbarlabel="")
texts = annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
plt.savefig('building_id_54_Feature_Heatmap.png')
plt.show()

