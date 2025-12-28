import jax
import jax.numpy as jnp
from jaxopt import AndersonAcceleration
import numpy as np

# Naka-Rushton function
def nr(x, n=4, k=1, eps=1e-8):
    return x**n / (x**n + k**n + eps)

# fixed-point function
def mitral_fixed_point(g, x, alpha, Wgr, Wrg, wb, n=4, k=1):
    g_new = nr(alpha * x - wb * nr(alpha * x, n, k) + Wrg @ nr(alpha * x - Wgr @ g - wb * nr(alpha * x, n, k), n, k), n, k)
    return g_new

def y_fixed_point(y, base_y, wcor, wy, wgranule, cortex, n=4, k=1):
    gr = nr(wcor * cortex + wy * y, n, k)
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
def solve_fixed_point(theta, x, m, n=4, k=1):
    wr, wb, Wgr, Wrg, Wgm, alpha, wcor, wy, wgranule = unpack_theta(theta, m)
    g0 = jnp.zeros(m)
    aa = AndersonAcceleration(lambda g, x, alpha, Wgr, Wrg, wb: mitral_fixed_point(g, x, alpha, Wgr, Wrg, wb, n, k), history_size=5, tol=1e-10, maxiter=50)
    solution = aa.run(g0, x, alpha, Wgr, Wrg, wb)
    return solution.params

# find mitral output using solved g
def mitral_output(theta, x, cortex, m, n=4, k=1):
    wr, wb, Wgr, Wrg, Wgm, alpha, wcor, wy, wgranule = unpack_theta(theta, m)

    g = solve_fixed_point(theta, x, m, n, k)
    b = nr(alpha * x, n, k)
    r = nr(alpha * x - Wgr @ g - wb * b, n, k)
    base_y = nr(alpha * x + wr * r - wb * b - Wgm @ g, n, k)

    y0 = base_y
    aa_y = AndersonAcceleration(lambda y, base_y, wcor, wy, wgranule, cortex: y_fixed_point(y, base_y, wcor, wy, wgranule, cortex, n, k), history_size=5, tol=1e-10, maxiter=200)
    solution = aa_y.run(y0, base_y, wcor, wy, wgranule, cortex)
    y_fixed = solution.params
    return r, g, b, y_fixed

# ---------------------------
# Target parameter search: sampling + manual gradient descent
# ---------------------------
m = 2
no_samples = 10
key = jax.random.PRNGKey(0)

nr_n = 4
nr_k = 1
x = jnp.ones(m)
cortex = jnp.array([0.5, 0.5])

saved_params = []

print(f"Starting search over {no_samples} samples...")
for i in range(no_samples):
    key, *subkeys = jax.random.split(key, 10)
    wr = jax.random.uniform(subkeys[0], (m,), minval=0.01, maxval=1.0)
    wb = jax.random.uniform(subkeys[1], (m,), minval=0.01, maxval=1.0)
    Wgr = jax.random.uniform(subkeys[2], (m, m), minval=0.01, maxval=1.0)
    Wrg = jax.random.uniform(subkeys[3], (m, m), minval=0.01, maxval=1.0)
    Wgm = jax.random.uniform(subkeys[4], (m, m), minval=0.01, maxval=1.0)
    alpha = jax.random.uniform(subkeys[5], (m,), minval=0.01, maxval=1.0)
    wcor = jax.random.uniform(subkeys[6], (m,), minval=0.01, maxval=1.0)
    wy = jax.random.uniform(subkeys[7], (m,), minval=0.01, maxval=1.0)
    wgranule = jax.random.uniform(subkeys[8], (m,), minval=0.01, maxval=1.0)

    theta = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(), alpha, wcor, wy, wgranule])

    # initial filter: check if outputs roughly in 0.3-0.7 range
    r, g, b, _ = mitral_output(theta, x, cortex, m, n=nr_n, k=nr_k)
    vals = jnp.concatenate([r, g, b])
    if jnp.all(vals > 0.3) and jnp.all(vals < 0.7):
        # manual gradient descent to target 0.5
        theta_var = theta.copy()
        lr = 0.05  # learning rate

        def target_loss(theta):
            r, g, b, _ = mitral_output(theta, x, cortex, m, n=nr_n, k=nr_k)
            return jnp.sum((r-0.5)**2 + (g-0.5)**2 + (b-0.5)**2)

        grad_fn = jax.grad(target_loss)
        for _ in range(50):  # 50 gradient steps
            grads = grad_fn(theta_var)
            theta_var = theta_var - lr * grads

        # final check: outputs near 0.5
        r, g, b, _ = mitral_output(theta_var, x, cortex, m, n=nr_n, k=nr_k)
        vals = jnp.concatenate([r, g, b])
        if jnp.all(vals > 0.2) and jnp.all(vals < 0.8):
            saved_params.append(theta_var)

print(f"Found {len(saved_params)} parameter sets with outputs near 0.5")

# After saving all tuned parameters
if len(saved_params) == 0:
    print("No parameters found matching the criteria.")
else:
    saved_params_array = np.array(saved_params)
    for i in range(len(saved_params_array)):
        r, g, b, _ = mitral_output(saved_params_array[i], x, cortex, m, n=nr_n, k=nr_k)
        print(f"Sample {i+1}: r = {r}, g = {g}, b = {b}")