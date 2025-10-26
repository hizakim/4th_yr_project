import jax
import jax.numpy as jnp
from jax import grad, hessian
import numpy as np
import pandas as pd

phi = jnp.tanh

# inputs
u1, u2, u3 = 1.0, 1.0, 1.0

n=4
def unpack_theta(theta):
    (z_osn, z_b, z_g2, z_g3) = theta[:n]
    (w_osn, w_b, w_g2, w_g3) = theta[n:2*n]
    (p1, q2, q3) = theta[2*n:]
    return z_osn, z_b, z_g2, z_g3, w_osn, w_b, w_g2, w_g3, p1, q2, q3

# model
def model(theta, u1=1.0, u2=1.0, u3=1.0):
    z_osn, z_b, z_g2, z_g3, w_osn, w_b, w_g2, w_g3, p1, q2, q3 = unpack_theta(theta)
    term_osn = z_osn * phi(u1 * w_osn)
    term_b   = z_b   * phi(w_b * phi(p1 * u1))
    term_g2  = z_g2  * phi(w_g2 * phi(q2 * u2))
    term_g3  = z_g3  * phi(w_g3 * phi(q3 * u3))
    return term_osn + term_b + term_g2 + term_g3

# random initialization
key = jax.random.PRNGKey(0)
theta_star = jnp.abs(jax.random.normal(key, (3*n - 1,)))
theta_star = theta_star.at[1:n].set(-theta_star[1:n])

# loss: difference in outputs between theta and theta_star
def loss(theta):
    return (model(theta) - model(theta_star))**2

# Hessian at theta_star
h = hessian(loss)(theta_star)

# eigen-decomposition
u, s, vt = jnp.linalg.svd(h)

np.savetxt("svals.txt", np.atleast_2d(np.array(s))) #saves singular values in a file svals.txt

# move along smallest eigenvector
delta = u[:, -1]
theta = theta_star + delta

l_star = loss(theta_star)
l = loss(theta)

print(f"l_star = {l_star}, l = {l}")


param_names = ['z_osn', 'z_b', 'z_g2', 'z_g3', 'w_osn', 'w_b', 'w_g2', 'w_g3', 'p1', 'q2', 'q3']

# DataFrame - rows: parameters, columns: initial, final, delta
df = pd.DataFrame({
    'Initial': theta_star,
    'Final': theta,
    'Delta': delta
}, index=param_names)

print("\nParameter Comparison Table:")
print(df)
