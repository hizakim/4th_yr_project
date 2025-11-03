import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jaxopt import AndersonAcceleration
import numpy as np

# -----------------------------
# Fixed-point function
# -----------------------------
def mitral_fixed_point(s, x, alpha, Wgr, Wrg):
    """
    s: state vector [r, b, g]
    x: input
    """
    m = x.shape[0]
    r, b, g = s[:m], s[m:2*m], s[2*m:]
    
    r_new = sigmoid(alpha * x - Wgr @ g)
    b_new = sigmoid(alpha * x)
    g_new = sigmoid(alpha * x - b_new + Wrg @ r)
    
    return jnp.concatenate([r_new, b_new, g_new])

# -----------------------------
# Parameter unpacking
# -----------------------------
def unpack_theta(theta, m):
    idx = 0
    wr = theta[idx:idx+m]; idx += m
    wb = theta[idx:idx+m]; idx += m
    Wgr = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    Wrg = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    Wgm = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    alpha = theta[idx:idx+m]; idx += m
    return wr, wb, Wgr, Wrg, Wgm, alpha

# -----------------------------
# Solve fixed point via Anderson
# -----------------------------
def solve_fixed_point(theta, x, m):
    # Unpack parameters inside the solve function
    wr, wb, Wgr, Wrg, Wgm, alpha = unpack_theta(theta, m)
    
    # Create fixed point function that takes all parameters explicitly
    # AndersonAcceleration supports passing additional args to run()
    # This avoids closure issues with custom_vjp
    # vjp = vector jacobian product
    def fp_fun(s, x_arg, alpha_arg, Wgr_arg, Wrg_arg):
        return mitral_fixed_point(s, x_arg, alpha_arg, Wgr_arg, Wrg_arg)
    
    s0 = jnp.zeros(3*m)
    
    # Pass all parameters as arguments to run() instead of closing over them
    # This makes all dependencies explicit for JAX's autodiff
    aa = AndersonAcceleration(fp_fun, history_size=5, tol=1e-10, maxiter=50)
    solution = aa.run(s0, x, alpha, Wgr, Wrg).params
    return solution # [r, b, g]

# -----------------------------
# Mitral output as function of theta
# -----------------------------
def mitral_output(theta, x, m):
    wr, wb, Wgr, Wrg, Wgm, alpha = unpack_theta(theta, m)
    
    s_star = solve_fixed_point(theta, x, m)
    
    r, b, g = s_star[:m], s_star[m:2*m], s_star[2*m:]
    y = sigmoid(alpha * x + wr * r - wb * b - Wgm @ g)
    return y

# -----------------------------
# Example initialisation
# -----------------------------
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

theta = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(), alpha])

# -----------------------------
# Hessian of scalar output
# -----------------------------
# Use jacrev(jacrev(...)) instead of hessian to avoid forward-mode autodiff
# which is incompatible with custom_vjp in AndersonAcceleration
loss = lambda theta: jnp.sum(mitral_output(theta, x, m))
grad_loss = jax.grad(loss)
H = jax.jacrev(grad_loss)(theta)

# -----------------------------
# SVD to find insensitive directions
# -----------------------------
u, s, vt = jnp.linalg.svd(H)
delta = u[:, -1]  # bottom eigenvector
theta_new = theta + delta

# -----------------------------
# Check outputs
# -----------------------------
y_star = mitral_output(theta, x, m)
y_new = mitral_output(theta_new, x, m)

print("Original output y_star:", y_star)
print("Output after perturbation y_new:", y_new)
print("Hessian bottom eigenvector:", u[:, -1])
