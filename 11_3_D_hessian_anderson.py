import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jaxopt import AndersonAcceleration
import numpy as np


# fixed-point function
def mitral_fixed_point(s, x, alpha, Wgr, Wrg, wb):
    m = x.shape[0]
    r, b, g = s[:m], s[m:2*m], s[2*m:]

    r_new = sigmoid(alpha * x - Wgr @ g - wb * b)
    b_new = sigmoid(alpha * x)
    g_new = sigmoid(alpha * x - wb * b_new + Wrg @ r)

    return jnp.concatenate([r_new, b_new, g_new])

# unpack parameter
def unpack_theta(theta, m):
    idx = 0
    wr = theta[idx:idx+m]; idx += m
    wb = theta[idx:idx+m]; idx += m
    Wgr = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    Wrg = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    Wgm = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    alpha = theta[idx:idx+m]; idx += m
    return wr, wb, Wgr, Wrg, Wgm, alpha


# solve fixed point via Anderson
def solve_fixed_point(theta, x, m):
    wr, wb, Wgr, Wrg, Wgm, alpha = unpack_theta(theta, m)
    s0 = jnp.zeros(3*m)

    aa = AndersonAcceleration(mitral_fixed_point, history_size=5, tol=1e-10, maxiter=50)
    solution = aa.run(s0, x, alpha, Wgr, Wrg, wb).params
    return solution # [r, b, g]


def mitral_output(theta, x, m):
    wr, wb, Wgr, Wrg, Wgm, alpha = unpack_theta(theta, m)

    s_true = solve_fixed_point(theta, x, m)

    r, b, g = s_true[:m], s_true[m:2*m], s_true[2*m:]
    y = sigmoid(alpha * x + wr * r - wb * b - Wgm @ g)
    return y

################################
# example, set theta_star
m = 3
x = jnp.array([1.0, 1.0, 1.0])

key = jax.random.PRNGKey(0)
wr = jax.random.normal(key, (m,))
wb = jax.random.normal(key, (m,))
key, subkey = jax.random.split(key)
Wgr = jax.random.normal(subkey, (m, m))
key, subkey = jax.random.split(key)
Wrg = jax.random.normal(subkey, (m, m))
key, subkey = jax.random.split(key)
Wgm = jax.random.normal(subkey, (m, m))
alpha = jax.random.normal(subkey, (m,))

theta_star = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(), alpha])
y_star = mitral_output(theta_star, x, m)

def loss(theta):
    y = mitral_output(theta, x, m)
    return jnp.sum((y - y_star)**2)

grad_loss = jax.grad(loss)
H = jax.jacrev(grad_loss)(theta_star) # Hessian of loss wrt theta
H = jnp.squeeze(H) # if dimension is size 1, remove

# SVD
u, s, vt = jnp.linalg.svd(H)
np.savetxt("svals.txt", np.atleast_2d(np.array(s)))
print("bottom eigenvector =", u[:, -1])

# TEST
# move along the bottom eigenvector
delta = u[:, -1]
theta = theta_star + delta

y_star_new = mitral_output(theta_star, x, m)
y_new = mitral_output(theta, x, m)

print(f"y_star_new = {y_star_new}, y_new = {y_new}")

l_star = loss(theta_star)
l = loss(theta)

print(f"l_star = {l_star}, l = {l}")
