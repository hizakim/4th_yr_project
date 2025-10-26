import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define 3 different weight vectors
w_list = [
    np.array([0.2, 0.6, 0.2]),
    np.array([0.5, 0.7, 0.3]),
    np.array([0.9, 0.8, 0.4])
]

# Example z values for computing t = tanh(z)
z = np.array([0.3, 0.7, -1.0])
t = np.tanh(z)

# Create grid for dz1, dz2
dz1_vals = np.linspace(-1, 1, 10)
dz2_vals = np.linspace(-1, 1, 10)
dz1, dz2 = np.meshgrid(dz1_vals, dz2_vals)

# --- PLOT δz PLANES ---
fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121, projection='3d')
colors = ["Reds", "Blues", "Greens"]

for w, cmap in zip(w_list, colors):
    dz3 = -(w[0]*dz1 + w[1]*dz2) / w[2]
    ax1.plot_surface(dz1, dz2, dz3, alpha=0.5, cmap=cmap)

ax1.set_xlabel("δz1")
ax1.set_ylabel("δz2")
ax1.set_zlabel("δz3")
ax1.set_title("Neutral δz planes")

# --- PLOT δw SURFACES ---
ax2 = fig.add_subplot(122, projection='3d')

for w, cmap in zip(w_list, colors):
    dz3 = -(w[0]*dz1 + w[1]*dz2) / w[2]

    # compute δw = w * t * δz
    dw1 = w[0] * t[0] * dz1
    dw2 = w[1] * t[1] * dz2
    dw3 = w[2] * t[2] * dz3

    ax2.plot_surface(dw1, dw2, dw3, alpha=0.5, cmap=cmap)

ax2.set_xlabel("δw1")
ax2.set_ylabel("δw2")
ax2.set_zlabel("δw3")
ax2.set_title("Corresponding δw surfaces")

plt.show()
