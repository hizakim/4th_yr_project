import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jaxopt import AndersonAcceleration
import numpy as np
import matplotlib.pyplot as plt

scale_g = 1
offset_g = 0
def sigmoid_g(x):
    return sigmoid(scale_g * x + offset_g)

scale_b = 1
offset_b = 0
def sigmoid_b(x):
    return sigmoid(scale_b * x + offset_b)

scale_r = 1
offset_r = 0
def sigmoid_r(x):
    return sigmoid(scale_r * x + offset_r)

scale_y = 1
offset_y = 0
def sigmoid_y(x):
    return sigmoid(scale_y * x + offset_y)

scale_gr = 1
offset_gr = 0
def sigmoid_gr(x):
    return sigmoid(scale_gr * x + offset_gr)


# fixed-point function
def mitral_fixed_point(g, x, alpha, Wgr, Wrg, wb):
    g_new = sigmoid_g(alpha * x - wb * sigmoid_b(alpha * x) + Wrg @ sigmoid_r(alpha * x - Wgr @ g - wb * sigmoid_b(alpha * x)))
    return g_new

def y_fixed_point(y, base_y, wcor, wy, wgranule, cortex):
    gr = sigmoid_gr(wcor * cortex + wy * y)
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
    b = sigmoid_b(alpha * x)
    r = sigmoid_r(alpha * x - Wgr @ g - wb * b)
    base_y = sigmoid_y(alpha *x + wr*r - wb*b - Wgm@g)

    #===================
    y0 = base_y
    aa_y = AndersonAcceleration(y_fixed_point, history_size=5, tol=1e-10, maxiter=20)
    solution = aa_y.run(y0, base_y, wcor, wy, wgranule, cortex)
    y_fixed = solution.params
    #===================
    return y_fixed


################################

key = jax.random.PRNGKey(0)
m = 2
x = jnp.ones(m)
cortex = jnp.array([0.5, 0.5])

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

key, base_key = jax.random.split(key)
#theta_star = sample_theta(base_key, m)
#y_star = mitral_output(theta_star, x, m)

def participation_ratio(H):
    u, s, vt = jnp.linalg.svd(H)
    participation_ratio = (jnp.sum(s))**2 / jnp.sum(s**2)
    return participation_ratio

def loss(theta):
    #y_ref = mitral_output(jax.lax.stop_gradient(theta), x, m)
    y = mitral_output(theta, x, m)
    return jnp.sum(y)

grad_loss = jax.grad(loss)
hessian = jax.jit(jax.jacrev(grad_loss))


no_samples = 100
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