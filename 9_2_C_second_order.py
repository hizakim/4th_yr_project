import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

'''
plot delta y = 0 contour
'''
# --------------------------
# Function y(theta)
n = 2
def y_func(theta):
    w, z = theta[:n], theta[n:]
    return jnp.sum(w * jnp.tanh(z))

# Random theta
key = jax.random.PRNGKey(0)
theta = jax.random.uniform(key, (2*n,), minval=0.0, maxval=1.0)

# Gradient and Hessian
grad_y = jax.grad(y_func)
hess_y = jax.hessian(y_func)
g = grad_y(theta)
H = hess_y(theta)

# Compute H-inverse and constant C
H_inv = jnp.linalg.inv(H)
C = float(g.T @ H_inv @ g)

# --------------------------
# Diagonalize H: H = Q Λ Q^T
eigvals, Q = jnp.linalg.eigh(H)
Lambda = jnp.diag(eigvals)

# --------------------------
# Create a 2D slice in original delta coordinates
# vary delta_w[0] and delta_z[0], fix delta_w[1] = delta_z[1] = 0
var_range = 1
grid_size = 300
dw0 = np.linspace(-var_range, var_range, grid_size)
dz0 = np.linspace(-var_range, var_range, grid_size)
DW0, DZ0 = np.meshgrid(dw0, dz0)

DeltaY_slice = np.zeros_like(DW0)

# Vectorized computation
for i in range(grid_size):
    for j in range(grid_size):
        delta = jnp.array([DW0[i,j], 0.0, DZ0[i,j], 0.0])  # delta = [dw0, dw1, dz0, dz1]
        # Complete-the-square: delta + H_inv g
        y_shifted = delta + H_inv @ g
        DeltaY_slice[i,j] = float(y_shifted @ H @ y_shifted)

# --------------------------
# Plot the zero-change contour
plt.figure(figsize=(6,5))
plt.contour(DW0, DZ0, DeltaY_slice, levels=[C], colors='black', linewidths=2)
plt.xlabel(r'$\Delta w_1$')
plt.ylabel(r'$\Delta z_1$')
plt.title(r'$\Delta y = 0$ contour (slice) for n=2')
plt.show()
