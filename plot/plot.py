import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define data points (you may adjust these based on your specific values)
rounds = np.array([0, 2, 4, 6, 8, 10, 12, 14])
a1_values = np.array([1, 2, 3, 4, 5, 6, 7])

# Cooperation rate values matrix
cooperation_rates = np.array([
    [0.0, 0.21, 0.42, 0.43, 0.44, 0.38, 0.40, 0.34],
    [0.0, 0.54, 0.75, 0.93, 0.84, 0.77, 0.72, 0.61],
    [0.0, 0.92, 0.93, 0.93, 0.93, 0.77, 0.84, 0.93],
    [0.0, 0.61, 0.77, 0.84, 0.93, 0.93, 0.93, 0.92],
    [0.0, 0.54, 0.75, 0.93, 0.84, 0.77, 0.72, 0.61],
    [0.0, 0.21, 0.42, 0.43, 0.44, 0.38, 0.40, 0.34],
    [0.0, 0.21, 0.42, 0.43, 0.44, 0.38, 0.40, 0.34]
])

# Create meshgrid for plotting
X, Y = np.meshgrid(rounds, a1_values)

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, cooperation_rates, cmap='viridis', alpha=0.7)

# Plot contour and lines
for i in range(len(a1_values)):
    ax.plot(rounds, [a1_values[i]]*len(rounds), cooperation_rates[i], color='blue', marker='o', markersize=4)

# Add color bar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('A1 Values')

# Labels and title
ax.set_xlabel('Round')
ax.set_ylabel('A1')
ax.set_zlabel('Cooperation Rate')
ax.set_title('3D Cooperation Rate Plot')
plt.savefig('plot.png')
plt.show()
