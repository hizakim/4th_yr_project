## using TOP.py


import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from TOP_2 import mitral_output

m = 2
x = jnp.ones(m)
cortex = jnp.array([0.5, 0.5])

w_osn_r = jnp.array([0.2, 0.2])
w_osn_b = jnp.array([0.2, 0.2])
w_osn_g = jnp.array([0.2, 0.2])
w_osn_y = jnp.array([0.2, 0.2])

w_b_r = jnp.array([0.2, 0.2])
w_b_g = jnp.array([0.2, 0.2])
w_b_y = jnp.array([0.2, 0.2])

w_r_y = jnp.array([0.2, 0.2])

W_g_r = jnp.array([[0, 0.2], [0.2, 0]])
W_r_g = jnp.array([[0, 0.2], [0.2, 0]])
W_g_m = jnp.array([[0, 0.2], [0.2, 0]])

alpha = jnp.array([0.8, 0.8])

w_cor = jnp.array([0.2, 0.2])
w_y = jnp.array([0.2, 0.2])
w_granule = jnp.array([0.2, 0.2])

theta_star = jnp.concatenate([w_osn_r, w_osn_b, w_osn_g, w_osn_y, w_b_r, w_b_g, w_b_y, w_r_y, W_g_r.flatten(), W_r_g.flatten(), W_g_m.flatten(), alpha, w_cor, w_y, w_granule])
_, _, _, y_star = mitral_output(theta_star, x, m, cortex)
print("y_star =", y_star)

#=====================================
# sample in theta space

no_samples = 10

theta_y_samples = [] # list of tuples (theta, y_samples)
key = jax.random.PRNGKey(0)
theta_samples = jax.random.uniform(key, (no_samples, len(theta_star)), minval=0.0, maxval=1.0)# no_samples samples in theta space
for theta in theta_samples:
    _, _, _, y_samples = mitral_output(theta, x, m, cortex)

    theta_y_samples.append((theta, y_samples))

# Filter out samples with negative y values
theta_y_samples_filtered = [(theta, y) for theta, y in theta_y_samples if jnp.all(y >= 0)]


#########################
## using TOP.py

from TOP import mitral_output
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

#select parameters that give target y and rgb output approx 0.5

#=====================================
y_target = jnp.array([0.5, 0.5])
g_target = jnp.array([0.5, 0.5])
b_target = jnp.array([0.5, 0.5])
r_target = jnp.array([0.5, 0.5])

no_params = 24

m = 2
x = jnp.ones(m)
cortex = jnp.array([0.5, 0.5])

# sample in theta space
no_samples = 10

theta_matrix = jnp.zeros((no_samples, no_params)) #make empty matrix no_samples x 24

key = jax.random.PRNGKey(0)
for i in range(no_samples):
    key, subkey = jax.random.split(key)
    theta_matrix = theta_matrix.at[i, :].set(jax.random.uniform(subkey, (no_params,), minval=0.0, maxval=1.0))


#########################

# loss function that depends on theta and g b r
def loss_fn(theta):
    g,b,r,y = mitral_output(theta, x, m, cortex)
    return jnp.sum((y - y_target) ** 2) + jnp.sum((g - g_target) ** 2) + jnp.sum((b - b_target) ** 2) + jnp.sum((r - r_target) ** 2)

def gradient_descent(loss_fn, theta_init, lr=0.01, steps=100):
    theta = theta_init
    grad_fn = jax.jit(jax.grad(loss_fn))

    for _ in range(steps):
        g = grad_fn(theta)
        theta = theta - lr * g #lr is learning rate
        theta = jnp.clip(theta, 0.0, None) #clip to 0
    return theta


theta_opt = jnp.zeros((no_samples, no_params))
y_opt = jnp.zeros((no_samples, m))
g_opt = jnp.zeros((no_samples, m))
b_opt = jnp.zeros((no_samples, m))
r_opt = jnp.zeros((no_samples, m))
for i in range(no_samples):
    theta_opt = theta_opt.at[i, :].set(gradient_descent(loss_fn, theta_matrix[i, :], lr=0.05, steps=100))
    g, b, r, y = mitral_output(theta_opt[i, :], x, m, cortex)
    y_opt = y_opt.at[i, :].set(y)
    g_opt = g_opt.at[i, :].set(g)
    b_opt = b_opt.at[i, :].set(b)
    r_opt = r_opt.at[i, :].set(r)

print(theta_opt)
print(y_opt)
print(g_opt)
print(b_opt)
print(r_opt)

#plot value vs each parameter index
for i in range(no_params):
    plt.plot(theta_opt[:, i], 'o-', label=f'Parameter {i}')
plt.xlabel('Parameter index')
plt.ylabel('Parameter value')
plt.show()