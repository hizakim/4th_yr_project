
## target y
# pick random theta samples
# grad descent to find theta that give target y
# make matrix Nx24 where N is no of samples
# participation ratio of all of them, remove if no evalue 0
# histogram of participation ratio



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
no_samples = 10
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
# Participation ratio
def participation_ratio(s):
    s2 = jnp.sum(s**2)
    s1 = jnp.sum(jnp.abs(s))
    return (s1*s1) / s2

PR_list = []
for theta in theta_matrix:
    g, u, s = from_y_next_step(theta, x, m, cortex)
    pr = float(participation_ratio(s))
    PR_list.append(pr)

PR_list = np.array(PR_list)
print("Participation ratios:", PR_list)

# ============================================================
# Plots
# Parameter trajectories
plt.figure(figsize=(10,6))
for theta in theta_matrix:
    plt.plot(np.array(theta[:24]), 'o-', alpha=0.7)
plt.xlabel("Parameter index")
plt.ylabel("Optimized parameter value")
plt.grid(True)
plt.show()

# Histogram of participation ratios
plt.figure(figsize=(8,5))
plt.hist(PR_list, bins=10, edgecolor='black')
plt.xlabel("Participation Ratio")
plt.ylabel("Count")
plt.grid(True)
plt.show()
