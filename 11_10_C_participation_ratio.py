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
def mitral_fixed_point(g, x, alpha, Wg_r, Wr_g, wb):
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

# def loss(theta):
#     y_ref = mitral_output(jax.lax.stop_gradient(theta), x, m)
#     y = mitral_output(theta, x, m)
#     return jnp.sum((y - y_ref)**2)

def loss(theta):
    #y_ref = mitral_output(jax.lax.stop_gradient(theta), x, m)
    y = mitral_output(theta, x, m)
    return jnp.sum(y)

grad_loss = jax.grad(loss)
hessian = jax.jit(jax.jacrev(grad_loss))

print("Let's start")
H = hessian(theta_star)
H = jnp.squeeze(H) # if dimension is size 1, remove

print("Hessian computed once")

H = hessian(2*theta_star)
print("Hessian computed twice")

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
participation_ratio = (jnp.sum(jnp.abs(s)))**2 / jnp.sum(s**2)
print(f"Participation ratio: {participation_ratio}")
