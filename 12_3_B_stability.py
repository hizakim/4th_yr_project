# TOP_stability.py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from TOP_2 import mitral_output


m = 2 #no mitral neurons                # per-channel dimension (you used m=2)
n_var = 4  # r,g,b,y
n = n_var * m  # total state length
x = jnp.ones(m)
cortex_base = jnp.array([0.5, 0.5])

no_params = 38
no_samples = 5

# Targets used for optimisation
y_target = jnp.array([0.5, 0.5])
g_target = jnp.array([0.5, 0.5])
b_target = jnp.array([0.5, 0.5])
r_target = jnp.array([0.5, 0.5])


key = jax.random.PRNGKey(0)
theta_matrix = jnp.zeros((no_samples, no_params))
for i in range(no_samples):
    key, subkey = jax.random.split(key)
    theta_matrix = theta_matrix.at[i, :].set(jax.random.uniform(subkey, (no_params,), minval=0.0, maxval=2.0))


# -------------------------
# gradient descent
def loss_fn(theta):
    g, b, r, y = mitral_output(theta, x, m, cortex_base)
    return jnp.sum((y - y_target) ** 2) + jnp.sum((g - g_target) ** 2) + jnp.sum((b - b_target) ** 2) + jnp.sum((r - r_target) ** 2)


def gradient_descent(loss_fn, theta_init, lr=0.01, steps=200):
    theta = theta_init
    grad_fn = jax.jit(jax.grad(loss_fn))
    for _ in range(steps):
        g = grad_fn(theta)
        theta = theta - lr * g
        theta = jnp.clip(theta, 0.0, None)
    return theta

################################
################################

# run gradient descent on each sampled theta
theta_opt = jnp.zeros((no_samples, no_params))
y_opt = jnp.zeros((no_samples, m))
g_opt = jnp.zeros((no_samples, m))
b_opt = jnp.zeros((no_samples, m))
r_opt = jnp.zeros((no_samples, m))

for i in range(no_samples):
    theta_opt = theta_opt.at[i, :].set(gradient_descent(loss_fn, theta_matrix[i, :], lr=0.05, steps=200))
    g, b, r, y = mitral_output(theta_opt[i, :], x, m, cortex_base)
    y_opt = y_opt.at[i, :].set(y)
    g_opt = g_opt.at[i, :].set(g)
    b_opt = b_opt.at[i, :].set(b)
    r_opt = r_opt.at[i, :].set(r)

print("Optimised thetas (theta_opt):\n", np.array(theta_opt))
print("Resulting outputs y_opt, g_opt, b_opt, r_opt:")
print("y_opt:\n", np.array(y_opt))
print("g_opt:\n", np.array(g_opt))
print("b_opt:\n", np.array(b_opt))
print("r_opt:\n", np.array(r_opt))


# -------------------------
# stability
# -------------------------

def F(s, theta, x, m, cortex):

    g_out, b_out, r_out, y_out = mitral_output(theta, x, m, cortex)

    # Construct s' in order [r_block, g_block, b_block, y_block]
    s_prime = jnp.concatenate([r_out, g_out, b_out, y_out])  # shape (n,)
    return s_prime


# -------------------------
# Jacobian
# -------------------------
def compute_jacobian(F, s_star, theta, x, m, cortex):
    # jax.jacobian returns an (n,n) array dF/ds
    J_fn = jax.jacobian(lambda ss: F(ss, theta, x, m, cortex))
    J = J_fn(s_star)
    return J

def stability_checks(F, s_star, theta, x, m, cortex, tau_vec, dt=0.01):

    n_local = s_star.shape[0]
    J = compute_jacobian(F, s_star, theta, x, m, cortex)    # (n,n)

    # Tau_inv diagonal matrix
    Tau_inv = jnp.diag(1.0 / tau_vec)                        # (n,n)
    I = jnp.eye(n_local)

    # continuous-time A matrix and eigenvalues
    A = Tau_inv @ (J - I)
    eigs_A = jnp.linalg.eigvals(A)
    eigs_A_np = np.array(eigs_A)
    stable_cont = np.all(np.real(eigs_A_np) < 0.0)
    return {"J": J, "eigs_A": eigs_A_np, "stable_continuous": stable_cont}


# -------------------------
# Run stability checks for each theta_opt sample

# -------------------------
tau_vec = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # 8 values for n=8 state variables
results_list = []
for i in range(no_samples):
    # model outputs at theta_opt[i] to use as candidate s_star
    g, b, r, y = mitral_output(theta_opt[i, :], x, m, cortex_base)
    s_star = jnp.concatenate([r, g, b, y])   # ordering [r_block, g_block, b_block, y_block]
    res = stability_checks(F, s_star, theta_opt[i, :], x, m, cortex_base, tau_vec, dt=0.05)
    results_list.append(res)

    print("\n=== Sample", i, "===\n")
    print("s_star (r,g,b,y blocks) =\n", np.array(s_star))
    print("\nJacobian J (dF/ds) =\n", np.array(res["J"]))
    print("\n eigenvalues:\n", res["eigs_A"])
