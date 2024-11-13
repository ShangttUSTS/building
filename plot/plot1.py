import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 示例数据
A1 = np.array([0, 1, 2, 3, 4, 5])
Rounds = np.array([0, 2, 4, 6, 8, 10, 12, 14])
Cooperation_rate = np.array([
    [0.2, 0.24, 0.34, 0.38, 0.43, 0.44, 0.50, 0.61],
    [0.21, 0.40, 0.38, 0.40, 0.43, 0.60, 0.72, 0.75],
    [0.22, 0.43, 0.44, 0.44, 0.60, 0.77, 0.84, 0.92],
    [0.23, 0.42, 0.44, 0.43, 0.60, 0.77, 0.86, 0.93],
    [0.25, 0.40, 0.44, 0.43, 0.61, 0.77, 0.88, 0.93],
    [0.21, 0.38, 0.42, 0.43, 0.60, 0.72, 0.84, 0.93]
])

X, Y = np.meshgrid(Rounds, A1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制每条线并填充颜色平面
for i in range(Cooperation_rate.shape[0]):
    ax.plot(Y[i], X[i], Cooperation_rate[i], 'o-', color='blue')
    ax.plot(Y[i], X[i], np.zeros_like(Cooperation_rate[i]), color='gray', alpha=0.5)

# 在线条与基准平面之间创建有颜色的平面
for i in range(len(A1)):
    for j in range(len(Rounds) - 1):
        verts = [
            [Y[i, j], X[i, j], 0],
            [Y[i, j + 1], X[i, j + 1], 0],
            [Y[i, j + 1], X[i, j + 1], Cooperation_rate[i, j + 1]],
            [Y[i, j], X[i, j], Cooperation_rate[i, j]]
        ]
        ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.viridis(i / len(A1)), alpha=0.3))

# 注释点的数值，并稍微偏移以避免重叠
for i in range(Cooperation_rate.shape[0]):
    for j in range(Cooperation_rate.shape[1]):
        ax.text(Y[i, j], X[i, j], Cooperation_rate[i, j] + 0.03,
                f'{Cooperation_rate[i, j]:.2f}', color='black', ha='center')

# 用灰色虚线连接不同线条之间的对应点
for j in range(Cooperation_rate.shape[1]):
    ax.plot(Y[:, j], X[:, j], Cooperation_rate[:, j], '--', color='gray')

# 添加标签并自定义图形
ax.set_xlabel('A1')
ax.set_ylabel('Round')
ax.set_zlabel('Cooperation Rate')
plt.savefig('plot1.png')
plt.show()
