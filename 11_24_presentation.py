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
    wcor = theta[idx:idx+m]; idx += m
    wy = theta[idx:idx+m]; idx += m
    wgranule = theta[idx:idx+m]; idx += m
    return wr, wb, Wgr, Wrg, Wgm, alpha, wcor, wy, wgranule


# solve fixed point via Anderson - return g
def solve_fixed_point(theta, x, m):
    _, wb, Wgr, Wrg, _, alpha, _, _, _ = unpack_theta(theta, m)
    g0 = jnp.zeros(m)
    aa = AndersonAcceleration(mitral_fixed_point, history_size=5, tol=1e-10, maxiter=50)
    solution = aa.run(g0, x, alpha, Wgr, Wrg, wb)
    g_fixed = solution.params
    return g_fixed

# find mitral output using solved g
def mitral_output(theta, x, m):
    wr, wb, Wgr, _, Wgm, alpha, wcor, wy, wgranule = unpack_theta(theta, m)

    g = solve_fixed_point(theta, x, m)
    b = nr(alpha * x)
    r = nr(alpha * x - Wgr @ g - wb * b)
    base_y = nr(alpha * x + wr * r - wb * b - Wgm @ g)

    #===================
    y0 = base_y
    aa_y = AndersonAcceleration(y_fixed_point, history_size=5, tol=1e-10, maxiter=200)
    solution = aa_y.run(y0, base_y, wcor, wy, wgranule, cortex)
    y_fixed = solution.params
    #===================
    return y_fixed


################################
# example
m = 2

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
wr = jax.random.uniform(subkey, (m,), minval=0.01, maxval=1.0)
key, subkey = jax.random.split(key)
wb = jax.random.uniform(subkey, (m,), minval=0.01, maxval=1.0)
key, subkey = jax.random.split(key)
Wgr = jax.random.uniform(subkey, (m, m), minval=0.01, maxval=1.0)
key, subkey = jax.random.split(key)
Wrg = jax.random.uniform(subkey, (m, m), minval=0.01, maxval=1.0)
key, subkey = jax.random.split(key)
Wgm = jax.random.uniform(subkey, (m, m), minval=0.01, maxval=1.0)
key, subkey = jax.random.split(key)
alpha = jax.random.uniform(subkey, (m,), minval=0.01, maxval=1.0)
key, subkey = jax.random.split(key)
wcor = jax.random.uniform(subkey, (m,), minval=0.01, maxval=1.0)
key, subkey = jax.random.split(key)
wy = jax.random.uniform(subkey, (m,), minval=0.01, maxval=1.0)
key, subkey = jax.random.split(key)
wgranule = jax.random.uniform(subkey, (m,), minval=0.01, maxval=1.0)

x = jnp.ones(m)
cortex = jnp.array([0.5, 0.5])

theta_star = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(), alpha, wcor, wy, wgranule])
y_star = mitral_output(theta_star, x, m)

def loss(theta):
    y = mitral_output(theta, x, m)
    return jnp.sum(y)

grad_loss = jax.grad(loss)
hessian = jax.jit(jax.jacrev(grad_loss))

H = hessian(theta_star)
H = jnp.squeeze(H) # if dimension is size 1, remove


# SVD
u, s, vt = jnp.linalg.svd(H)
np.savetxt("svals.txt", np.atleast_2d(np.array(s)))
#print("bottom eigenvector =", u[:, -1])


plt.figure(figsize=(8,4))
plt.semilogy(s, marker='o')
plt.xlabel("Parameter Index")
plt.ylabel("Singular Values")
plt.grid(True)
plt.tight_layout()
plt.show()

print("s =", s)

# participation ratio
participation_ratio = (jnp.sum(s))**2 / jnp.sum(s**2)
print(f"Participation ratio: {participation_ratio}")

# take the 6 bottom eigen vectors and plot value vs index (24)
u[:, -2:]

plt.figure(figsize=(8,4))
plt.plot(u[:, -2:], marker='o', linestyle='-')
plt.xlabel("Parameter Index")
plt.ylabel("$\Delta\Theta$")
plt.grid(True)
plt.show()

# check eigen vector -2 are perpendicular to gradient g
g = grad_loss(theta_star)
print('g =', g)
print('u[:, -6:].T @ g =', u[:, -6:].T @ g)