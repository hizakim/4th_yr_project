## using TOP.py

from TOP import mitral_output
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

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

theta_star = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(), alpha, wcor, wy, wgranule])
y_star = mitral_output(theta_star, x, m, cortex)
print("y_star =", y_star)

#=====================================
# sample in theta space

no_samples = 10

theta_y_samples = [] # list of tuples (theta, y_samples)
key = jax.random.PRNGKey(0)
theta_samples = jax.random.uniform(key, (no_samples, len(theta_star)), minval=0.0, maxval=1.0)# no_samples samples in theta space
for theta in theta_samples:
    y_samples = mitral_output(theta, x, m, cortex)

    theta_y_samples.append((theta, y_samples))

# Filter out samples with negative y values
theta_y_samples_filtered = [(theta, y) for theta, y in theta_y_samples if jnp.all(y >= 0)]


#########################

# loss function that depends on theta
def loss_fn(theta):
    y = mitral_output(theta, x, m, cortex)
    return jnp.sum((y - y_star) ** 2)

def gradient_descent(loss_fn, theta_init, lr=0.01, steps=100):
    theta = theta_init
    grad_fn = jax.jit(jax.grad(loss_fn))

    for _ in range(steps):
        g = grad_fn(theta)
        theta = theta - lr * g
    return theta

# Loop over filtered samples
theta_opts = []
for i, (theta_sample, _) in enumerate(theta_y_samples_filtered):

    # optimise theta to match y_star
    theta_opt = gradient_descent(loss_fn, theta_sample, lr=0.05, steps=100)
    y_opt = mitral_output(theta_opt, x, m, cortex)

    theta_opts.append(theta_opt)

    # print(f"Sample {i}:")
    # print("y_opt =", y_opt)


# plot results
# 24 parameters along x axis, their values along y axis
# for each set of parameters plot with same colour
for i in range(len(theta_opts)):
    # Convert JAX array to numpy and slice first 24 elements
    theta_array = np.array(theta_opts[i])[:24]
    plt.plot(theta_array, 'o-')
plt.xlabel('parameter index')
plt.ylabel('parameter value')
plt.grid(True)
plt.show()
