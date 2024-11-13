import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Sample data
A1 = ['LifeLongGo_Mf_CC_BP', 'LifeLongGo_Mf_BP_CC', 'LifeLongGo_BP_Mf_CC', 'LifeLongGo_BP_CC_Mf', 'LifeLongGo_CC_Mf_BP', 'LifeLongGo_CC_BP_Mf']
Rounds = ['Fmax', 'AUPR', 'AUC']
Cooperation_rate = np.array([
    [0.545, 0.536, 0.849],
    [0.545, 0.537, 0.851],
    [0.546, 0.541, 0.849],
    [0.538, 0.534, 0.857],
    [0.55, 0.55, 0.862],
    [0.547, 0.541, 0.855]
])

# Adjust meshgrid for correct axis labeling
X, Y = np.meshgrid(np.arange(len(Rounds)), np.arange(len(A1)))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot each line and fill with translucent planes
for i in range(Cooperation_rate.shape[0]):
    ax.plot(X[i], Y[i], Cooperation_rate[i], 'o-', color='blue')
    ax.plot(X[i], Y[i], np.zeros_like(Cooperation_rate[i]), color='gray', alpha=0.3)

# Add translucent color planes between each line and base
for i in range(len(A1)):
    for j in range(len(Rounds) - 1):
        verts = [
            [X[i, j], Y[i, j], 0],
            [X[i, j + 1], Y[i, j + 1], 0],
            [X[i, j + 1], Y[i, j + 1], Cooperation_rate[i, j + 1]],
            [X[i, j], Y[i, j], Cooperation_rate[i, j]]
        ]
        ax.add_collection3d(Poly3DCollection([verts], color=plt.cm.viridis(i / len(A1)), alpha=0.4))

# Annotate each point with its value
for i in range(Cooperation_rate.shape[0]):
    for j in range(Cooperation_rate.shape[1]):
        ax.text(X[i, j], Y[i, j], Cooperation_rate[i, j] + 0.02,
                f'{Cooperation_rate[i, j]:.3f}', color='black', ha='center', fontsize=8)

# Draw dotted lines connecting corresponding points across layers
for j in range(Cooperation_rate.shape[1]):
    ax.plot(X[:, j], Y[:, j], Cooperation_rate[:, j], '--', color='gray', alpha=0.5)

# Add color bar for height indication
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(Cooperation_rate)
cbar = plt.colorbar(mappable, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('Height (Cooperation Rate)')

# Set labels, title, and grid visibility
ax.set_xlabel('Round')
ax.set_ylabel('A1')
ax.set_zlabel('Cooperation Rate')
ax.set_title('3D Cooperation Rate Plot with Translucent Layers')
ax.set_xticks(np.arange(len(Rounds)))
ax.set_xticklabels(Rounds)
ax.set_yticks(np.arange(len(A1)))
ax.set_yticklabels(A1)
ax.grid(False)

plt.savefig('111.png')
plt.show()
