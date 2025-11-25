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