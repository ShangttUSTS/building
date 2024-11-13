import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# 示例数据集，确保你已经定义并包含以下字段
dataset = {
    "meter_reading": np.random.rand(200),
    "air_temperature": np.random.rand(200),
    "hour": np.random.rand(200)
}

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 200
x = dataset["meter_reading"][:n]
y = dataset["air_temperature"][:n]
z = dataset["hour"][:n]

ax.scatter(x, y, z, marker="o")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
