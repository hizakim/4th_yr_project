
## target y
# pick random theta samples
# grad descent to find theta that give target y
# make matrix Nx24 where N is no of samples

#for each parameter make histogram of freq vs values



from TOP import mitral_output, from_y_next_step
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Define model & initial parameters
m = 2
x = jnp.ones(m)
cortex = jnp.array([0.5, 0.5])

wr = jnp.array([0.2, 0.2])
wb = jnp.array([0.2, 0.2])
Wgr = jnp.array([[0.2, 0.2], [0.2, 0.2]])
Wrg = jnp.array([[0.2, 0.2], [0.2, 0.2]])
Wgm = jnp.array([[0.2, 0.2], [0.2, 0.2]])
alpha = jnp.array([0.8, 0.8])
wcor = jnp.array([0.2, 0.2])
wy = jnp.array([0.2, 0.2])
wgranule = jnp.array([0.2, 0.2])

theta_star = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(),
                              alpha, wcor, wy, wgranule])
y_star = mitral_output(theta_star, x, m, cortex)
print("y_star =", y_star)

# ============================================================
# Sample random theta
no_samples = 50
key = jax.random.PRNGKey(0)
theta_samples = jax.random.uniform(key, (no_samples, len(theta_star)), minval=0.0, maxval=1.0)

theta_y_samples = [(theta, mitral_output(theta, x, m, cortex)) for theta in theta_samples]
theta_y_samples_filtered = [(theta, y) for theta, y in theta_y_samples if jnp.all(y >= 0)]
print(f"Filtered samples: {len(theta_y_samples_filtered)} / {no_samples}")

# ============================================================
# Loss function
def loss_fn(theta):
    y = mitral_output(theta, x, m, cortex)
    return jnp.sum((y - y_star)**2)

# ============================================================
# Gradient descent
def gradient_descent(theta_init, loss_fn, lr=0.05, steps=100):
    theta = theta_init
    grad_fn = jax.jit(jax.grad(loss_fn))
    for _ in range(steps):
        theta = theta - lr * grad_fn(theta)
    return theta

# ============================================================
# Optimize theta samples
opt_results = []
for theta_sample, _ in theta_y_samples_filtered:
    theta_opt = gradient_descent(theta_sample, loss_fn, lr=0.05, steps=150)
    opt_results.append(theta_opt)

# Build N x D matrix
theta_matrix = jnp.stack(opt_results)
N, D = theta_matrix.shape
print("Optimized matrix shape:", theta_matrix.shape)

# ============================================================
# Histograms for each parameter

theta_matrix_np = np.array(theta_matrix)

n_rows = 6
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 20))
axes = axes.flatten()  # Flatten to 1D array for easier indexing

for param_idx in range(D):
    print('param_idx =',param_idx)
    # Extract values for this parameter across all samples
    param_values = theta_matrix_np[:, param_idx]

    # Create histogram
    axes[param_idx].hist(param_values, bins=15, edgecolor='black', alpha=0.7)
    axes[param_idx].set_title(f'Parameter {param_idx}', fontsize=10)
    axes[param_idx].set_xlabel('Value', fontsize=8)
    axes[param_idx].set_ylabel('Frequency', fontsize=8)
    axes[param_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Histograms of Parameter Values Across All Samples',
             fontsize=14, y=1.0, x=0.5)
plt.subplots_adjust(top=0.98)
plt.show()
