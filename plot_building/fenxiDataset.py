import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("../data/work/building62postoffice.csv")
X = data[["OAT", "Building 6 kW"]].to_numpy()
x, y = X[:, 0], X[:, 1]
plt.scatter(x, y)
# plt.show()

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)  # 聚成几类
model.fit(X)
y_pred = model.predict(X)
plt.scatter(x, y, c = y_pred)
# plt.show()

from sklearn.cluster import DBSCAN
from sklearn import metrics
db = DBSCAN(eps=10, min_samples=100).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
colors = ["lightblue", "lightgreen"]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markersize=6)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.xlabel("xxxxxxxxxxxxxxxx")
plt.ylabel("yyyyyyyyyyyyyyyy")
plt.show()

# 归一化
x = data["Building 6 kW"]
plt.plot(data.index, data["Building 6 kW"])
plt.show()

data = pd.read_csv("../data/data/train.csv")
weather = pd.read_csv("../data/data/weather_train.csv")

energy1 = data[(data["building_id"] == 150) & (data["meter"] == 0)]
weather1 = weather[weather["site_id"] == 14]["air_temperature"]
dataset = energy1.merge(weather[weather["site_id"] == 14], on="timestamp")
dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
dataset["hour"] = dataset.timestamp.dt.hour

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
ax = plt.figure().add_subplot(projection='3d')
n = 200
x = dataset["meter_reading"][:n]
y = dataset["air_temperature"][:n]
z = dataset["hour"][:n]
ax.scatter(x, y, z, marker="o")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# 4
sub_data = data[(data["building_id"] ==150) & (data["meter"] == 0)]
plt.plot(sub_data["meter_reading"].to_numpy()[300:600])
plt.show()