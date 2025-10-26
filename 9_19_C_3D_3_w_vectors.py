import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define 3 different weight vectors
w_list = [
    np.array([0.5, 0.6, 0.2]),
    np.array([0.5, 0.7, 0.3]),
    np.array([0.5, 0.8, 0.4])
]

# Create grid for dz1, dz2
dz1_vals = np.linspace(-1, 1, 20)
dz2_vals = np.linspace(-1, 1, 20)
dz1, dz2 = np.meshgrid(dz1_vals, dz2_vals)

# Plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

colors = ["Reds", "Blues", "Greens"]

for w, cmap in zip(w_list, colors):
    dz3 = -(w[0]*dz1 + w[1]*dz2) / w[2]
    ax.plot_surface(dz1, dz2, dz3, alpha=0.5, cmap=cmap)

ax.set_xlabel("dz1")
ax.set_ylabel("dz2")
ax.set_zlabel("dz3")
ax.set_title("Neutral planes")

plt.show()
