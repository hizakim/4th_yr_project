import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.scipy.optimize import minimize

'''
try and solve delta using optimisation
'''
# --------------------------
# Function y(theta)
n = 2
def y_func(theta):
    w, z = theta[:n], theta[n:]
    return jnp.sum(w * jnp.tanh(z))

# Random theta
key = jax.random.PRNGKey(20)
theta = jax.random.uniform(key, (2*n,), minval=0.0, maxval=1.0)

# Gradient and Hessian
grad_y = jax.grad(y_func)
hess_y = jax.hessian(y_func)
g = grad_y(theta)
H = hess_y(theta)

def loss(delta):
    Delta_y = g @ delta + 0.5 * delta @ H @ delta
    return Delta_y**2

# Optimisation
delta_init = jnp.full((2*n,), 0.1)
res = minimize(loss, delta_init, method='BFGS', options={'maxiter': 100})
delta_solution = res.x

# --------------------------
# Make grid over w1, z1 and compute Δy
grid_size = 100
var_range = 1.0
w1_vals = np.linspace(-var_range, var_range, grid_size)
z1_vals = np.linspace(-var_range, var_range, grid_size)
W1, Z1 = np.meshgrid(w1_vals, z1_vals)
DeltaY_grid = np.zeros_like(W1)

for i in range(grid_size):
    for j in range(grid_size):
        # keep w2, z2 at optimised solution
        delta = jnp.array([W1[i, j], delta_solution[1], Z1[i, j], delta_solution[3]])
        DeltaY_grid[i, j] = float(g @ delta + 0.5 * delta @ H @ delta)

# --------------------------
# Plot heatmap
plt.figure(figsize=(6,5))
plt.contourf(W1, Z1, DeltaY_grid, levels=50, cmap="coolwarm")
plt.colorbar(label=r'$\Delta y$')
plt.xlabel(r'$\Delta w_1$')
plt.ylabel(r'$\Delta z_1$')
plt.title(r'Heatmap of $\Delta y$ after optimisation (slice)')
plt.show()

# Print solution info
delta_y_val = g @ delta_solution + 0.5 * delta_solution @ H @ delta_solution
print("Delta solving Δy=0 via BFGS:", delta_solution)
print("Δy at solution:", delta_y_val)
