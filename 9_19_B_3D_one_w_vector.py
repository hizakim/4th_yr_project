import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example weights
w = np.array([0.5, 0.8, -1.0])  # (w1, w2, w3)

# Create grid for dz1, dz2
dz1_vals = np.linspace(-1, 1, 10)
dz2_vals = np.linspace(-1, 1, 10)
dz1, dz2 = np.meshgrid(dz1_vals, dz2_vals)

# Solve for δz3
dz3 = -(w[0]*dz1 + w[1]*dz2) / w[2]

# Plot plane
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(dz1, dz2, dz3, alpha=0.6, cmap="bwr")

ax.set_xlabel("dz1")
ax.set_ylabel("dz2")
ax.set_zlabel("dz3")
ax.set_title("dz plane: w·dz = 0")

plt.show()
