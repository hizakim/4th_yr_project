import numpy as np
import matplotlib.pyplot as plt

def y(w, z):
    return w * np.tanh(z)

# grids
w = np.linspace(0, 5, 20)
z = np.linspace(0, 5, 20)
W, Z = np.meshgrid(w, z)

# function values
Y = y(W, Z)

# slices for line plots
w0_idx = len(w)//2  # middle w
z0_idx = len(z)//2  # middle z
w0 = w[w0_idx]
z0 = z[z0_idx]

# partial derivatives
dy_dw = np.tanh(Z)
dy_dz = W * (1 - np.tanh(Z)**2)
dy_min = np.minimum(dy_dw, dy_dz)

# Figure with 3x3 layout
fig = plt.figure(figsize=(18, 14))

# 1. Original 3D surface
ax1 = fig.add_subplot(331, projection='3d')
ax1.plot_surface(W, Z, Y, cmap='bwr', alpha=0.8)
ax1.set_title("Original 3D Surface")
ax1.set_xlabel("w")
ax1.set_ylabel("z")
ax1.set_zlabel("y")

# 2. Surface colored by dy/dw
ax2 = fig.add_subplot(332, projection='3d')
surf2 = ax2.plot_surface(
    W, Z, Y,
    facecolors=plt.cm.Blues((dy_dw - dy_dw.min()) / (dy_dw.max() - dy_dw.min())),
    alpha=0.9
)
mappable2 = plt.cm.ScalarMappable(cmap='Blues')
mappable2.set_array(dy_dw)
fig.colorbar(mappable2, ax=ax2, shrink=0.6, label='∂y/∂w')
ax2.set_title("Surface Colored by ∂y/∂w")
ax2.set_xlabel("w")
ax2.set_ylabel("z")
ax2.set_zlabel("y")

# 3. Surface colored by dy/dz
ax3 = fig.add_subplot(333, projection='3d')
surf3 = ax3.plot_surface(
    W, Z, Y,
    facecolors=plt.cm.Greens((dy_dz - dy_dz.min()) / (dy_dz.max() - dy_dz.min())),
    alpha=0.9
)
mappable3 = plt.cm.ScalarMappable(cmap='Greens')
mappable3.set_array(dy_dz)
fig.colorbar(mappable3, ax=ax3, shrink=0.6, label='∂y/∂z')
ax3.set_title("Surface Colored by ∂y/∂z")
ax3.set_xlabel("w")
ax3.set_ylabel("z")
ax3.set_zlabel("y")

# 4. Surface colored by min(∂y/∂w, ∂y/∂z)
ax4 = fig.add_subplot(334, projection='3d')
surf4 = ax4.plot_surface(
    W, Z, Y,
    facecolors=plt.cm.Purples((dy_min - dy_min.min()) / (dy_min.max() - dy_min.min())),
    alpha=0.9
)
mappable4 = plt.cm.ScalarMappable(cmap='Purples')
mappable4.set_array(dy_min)
fig.colorbar(mappable4, ax=ax4, shrink=0.6, label='min(∂y/∂w, ∂y/∂z)')
ax4.set_title("Surface Colored by min(∂y/∂w, ∂y/∂z)")
ax4.set_xlabel("w")
ax4.set_ylabel("z")
ax4.set_zlabel("y")

# 5. Slice at fixed w = w0 (y vs z)
ax5 = fig.add_subplot(335)
ax5.plot(z, Y[w0_idx, :], marker='o')
ax5.set_xlabel("z")
ax5.set_ylabel(f"y (w={w0:.2f})")
ax5.set_title(f"Slice at w = {w0:.2f}")

# 6. Slice at fixed z = z0 (y vs w)
ax6 = fig.add_subplot(336)
ax6.plot(w, Y[:, z0_idx], marker='o')
ax6.set_xlabel("w")
ax6.set_ylabel(f"y (z={z0:.2f})")
ax6.set_title(f"Slice at z = {z0:.2f}")

plt.tight_layout()
plt.show()
