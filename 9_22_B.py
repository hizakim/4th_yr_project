import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Define y(theta)
def y(theta):
    n = theta.size // 2
    w, z = theta[:n], theta[n:]
    return jnp.sum(w * jnp.tanh(z))

# Original theta
theta = jnp.array([0.4, 0.8])
y0 = y(theta)

# Compute gradient
grad_y = jax.grad(y)
g = grad_y(theta)

# Hessian approximation
H_loss = 2 * jnp.outer(g, g)
e_vals, e_vectors = jnp.linalg.eigh(H_loss)

# Perturb along zero eigenvalue direction
new_thetas = []
for i in range(len(e_vals)):
    if jnp.isclose(e_vals[i], 0):
        new_theta = theta + 0.1 * e_vectors[:, i]
        new_thetas.append(new_theta)

# Create grid for heatmap
w_vals = np.linspace(0, 1, 100)
z_vals = np.linspace(0, 1, 100)
W, Z = np.meshgrid(w_vals, z_vals)
Y = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        theta_grid = jnp.array([W[i,j], Z[i,j]])
        Y[i,j] = y(theta_grid)

# Plot heatmap
plt.figure(figsize=(8,6))
plt.contourf(W, Z, Y, levels=50, cmap='viridis')
plt.colorbar(label='y(w,z)')
plt.xlabel('w')
plt.ylabel('z')
plt.title('Heatmap of y(w,z)')

# Overlay original theta
plt.scatter(theta[0], theta[1], color='red', label=f'Original (y={y0:.3f})')

# Overlay new thetas
for new_theta in new_thetas:
    y_new = y(new_theta)
    plt.scatter(new_theta[0], new_theta[1], color='white', edgecolor='black',
                label=f'New (y={y_new:.3f})')

plt.legend()
plt.show()
