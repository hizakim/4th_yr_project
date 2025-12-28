import jax
import jax.numpy as jnp
from jaxopt import AndersonAcceleration
import numpy as np
import matplotlib.pyplot as plt

# Naka-Rushton function
def nr(x, n=4, k=1):
    return x**n / (x**n + k**n)

# fixed-point function
def mitral_fixed_point(g, x, alpha, Wgr, Wrg, wb):
    g_new = nr(alpha * x - wb * nr(alpha * x) + Wrg @ nr(alpha * x - Wgr @ g - wb * nr(alpha * x)))
    return g_new

def y_fixed_point(y, base_y, wcor, wy, wgranule, cortex):
    gr = nr(wcor * cortex + wy * y)
    y = base_y - wgranule * gr
    return y

# unpack parameter
def unpack_theta(theta, m):
    idx = 0
    wr = theta[idx:idx+m]; idx += m
    wb = theta[idx:idx+m]; idx += m
    Wgr = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    Wrg = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    Wgm = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    alpha = theta[idx:idx+m]; idx += m
    wcor = theta[idx:idx+m]; idx += m # all same value (if one granule)
    wy = theta[idx:idx+m]; idx += m
    wgranule = theta[idx:idx+m]; idx += m
    return wr, wb, Wgr, Wrg, Wgm, alpha, wcor, wy, wgranule


# solve fixed point via Anderson - return g
def solve_fixed_point(theta, x, m):
    wr, wb, Wgr, Wrg, Wgm, alpha, wcor, wy, wgranule = unpack_theta(theta, m)
    g0 = jnp.zeros(m)
    aa = AndersonAcceleration(mitral_fixed_point, history_size=5, tol=1e-10, maxiter=50)
    solution = aa.run(g0, x, alpha, Wgr, Wrg, wb)
    g_fixed = solution.params
    return g_fixed

# find mitral output using solved g
def mitral_output(theta, x, m):
    wr, wb, Wgr, Wrg, Wgm, alpha, wcor, wy, wgranule = unpack_theta(theta, m)

    g = solve_fixed_point(theta, x, m)
    b = nr(alpha * x, n=1, k=1)
    r = nr(alpha * x - Wgr @ g - wb * b, n=1, k=1)
    base_y = nr(alpha * x + wr * r - wb * b - Wgm @ g)

    #===================
    y0 = base_y
    aa_y = AndersonAcceleration(y_fixed_point, history_size=5, tol=1e-10, maxiter=200)
    solution = aa_y.run(y0, base_y, wcor, wy, wgranule, cortex)
    y_fixed = solution.params
    #===================
    return y_fixed

def participation_ratio(H):
    u, s, vt = jnp.linalg.svd(H)
    participation_ratio = (jnp.sum(s))**2 / jnp.sum(s**2)
    return participation_ratio

def loss(theta):
    y = mitral_output(theta, x, m)
    return jnp.sum(y)

################################

m = 2
x = jnp.ones(m)
cortex = jnp.array([0.5, 0.5])

key = jax.random.PRNGKey(0)
def sample_theta(key, m):
    keys = jax.random.split(key, 9)
    wr = jax.random.uniform(keys[0], (m,), minval=0.0, maxval=1.0)
    wb = jax.random.uniform(keys[1], (m,), minval=0.0, maxval=1.0)
    Wgr = jax.random.uniform(keys[2], (m, m), minval=0.0, maxval=1.0)
    Wrg = jax.random.uniform(keys[3], (m, m), minval=0.0, maxval=1.0)
    Wgm = jax.random.uniform(keys[4], (m, m), minval=0.0, maxval=1.0)
    alpha = jax.random.uniform(keys[5], (m,), minval=0.0, maxval=1.0)
    wcor = jax.random.uniform(keys[6], (m,), minval=0.0, maxval=1.0)
    wy = jax.random.uniform(keys[7], (m,), minval=0.01, maxval=1.0)
    wgranule = jax.random.uniform(keys[8], (m,), minval=0.0, maxval=1.0)
    theta = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(), alpha, wcor, wy, wgranule])
    return theta

grad_loss = jax.grad(loss)
hessian = jax.jit(jax.jacrev(grad_loss))

no_samples = 200
participation_ratios = []

for i in range(no_samples):
    key, subkey = jax.random.split(key)
    theta = sample_theta(subkey, m)
    H = hessian(theta)
    H = jnp.squeeze(H) # if dimension is size 1, remove
    pr = float(participation_ratio(H))
    participation_ratios.append(pr)

plt.figure(figsize=(8, 5))
plt.hist(participation_ratios, bins=30, color="skyblue", edgecolor="k", alpha=0.8)
plt.xlabel("Participation Ratio", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()