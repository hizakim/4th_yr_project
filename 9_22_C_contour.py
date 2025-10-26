import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Define y(theta)
def y(theta):
    n = theta.size // 2
    w, z = theta[:n], theta[n:]
    return jnp.sum(w * jnp.tanh(z))

grad_y = jax.grad(y)

# Original theta
theta = jnp.array([0.4, 0.8])
points = [theta]
y_values = [y(theta)]

# Repeat 4 steps: recalc gradient, Hessian, eigenvector, perturb
num_steps = 4
for step in range(num_steps):
    g = grad_y(theta)
    H = 2 * jnp.outer(g, g)
    e_vals, e_vectors = jnp.linalg.eigh(H)

    # Pick the zero-eigenvalue direction (or closest to zero)
    idx = jnp.argmin(jnp.abs(e_vals))
    theta = theta + 0.1 * e_vectors[:, idx]  # perturb along eigenvector
    points.append(theta)
    y_values.append(y(theta))

# Convert to array
points = jnp.array(points)

# Create heatmap grid
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
plt.title('Heatmap of y(w,z) with 4-step recalculated eigenvector path')

# Plot points and arrows
colors = ['red', 'white', 'orange', 'cyan', 'magenta']
for i, (p, yv) in enumerate(zip(points, y_values)):
    plt.scatter(p[0], p[1], color=colors[i], edgecolor='black', s=100,
                label=f'Step {i} (y={yv:.3f})')
    if i > 0:
        plt.arrow(points[i-1,0], points[i-1,1],
                  p[0]-points[i-1,0], p[1]-points[i-1,1],
                  color='black', head_width=0.02, length_includes_head=True)

plt.legend()
plt.show()
