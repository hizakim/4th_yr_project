import jax
import jax.numpy as jnp
from jax import grad, hessian
import numpy as np

# parameters
n = 4
n_out = 1

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
w_in = jax.random.normal(subkey, (1, n))

# n_out x n
key, subkey = jax.random.split(key)
w_out = jax.random.normal(subkey, (n_out, n)) / jnp.sqrt(n)

# concatenate to get theta_star: shape (1, n + n_out*n)
theta_star = jnp.concatenate([w_in, w_out.reshape(1, -1)], axis=1)

phi = jnp.tanh

# MLP: theta -> output (scalar)
def mlp(theta): # split theta into w_in and w_out
    w_in = theta[:, :n]
    w_out = theta[:, n:].reshape(n_out, n) # reshape w_out
    return jnp.sum(w_out @ phi(w_in.T))

# loss: how differently the network behaves under theta vs theta_star
def loss(theta):
    return jnp.sum((mlp(theta) - mlp(theta_star)) ** 2)

# Hessian at theta_star
h = hessian(loss)(theta_star)
h = jnp.squeeze(h)  # if dimension is size 1, remove

# SVD
u, s, vt = jnp.linalg.svd(h)
np.savetxt("svals.txt", np.atleast_2d(np.array(s)))

# move along the bottom eigenvector
delta = u[:, -1]
theta = theta_star + delta

l_star = loss(theta_star)
l = loss(theta)

print(f"l_star = {l_star}, l = {l}")
