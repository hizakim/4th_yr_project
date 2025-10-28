import jax
import jax.numpy as jnp
from jax import grad, hessian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def naka_rushton(x, n=2.0, sigma=1.0):
    return (x**n) / (x**n + sigma**n)

phi = naka_rushton

# inputs
u1, u2, u3, u4 = 1.0, 1.0, 1.0, 1.0
mitral_no = 4

def unpack_theta(theta):
    idx = 0
    z_osn = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    w_osn = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    z_b   = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    w_b   = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    p_vec = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    z_etc = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    w_etc = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    r_vec = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    z_mat = jnp.array(theta[idx:idx+mitral_no*mitral_no]).reshape((mitral_no, mitral_no)); idx += mitral_no*mitral_no
    w_g   = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    w_r   = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    q_vec = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    return z_osn, w_osn, z_b, w_b, p_vec, z_etc, w_etc, r_vec, z_mat, w_g, w_r, q_vec

def model(theta, u_vec):
    z_osn, w_osn, z_b, w_b, p_vec, z_etc, w_etc, r_vec, z_mat, w_g, w_r, q_vec = unpack_theta(theta)
    y = z_osn * w_osn * u_vec + z_b * w_b * phi(u_vec * p_vec) + z_etc * w_etc * phi(u_vec * r_vec) + z_mat@(w_g * phi(u_vec * q_vec) + w_r * phi(w_etc * phi(u_vec * r_vec)))
    return y


u_vec = jnp.array([1.0, 1.0, 1.0, 1.0])
z_osn = jnp.array([1,1,1,1])
w_osn = jnp.array([1,1,1,1])
z_b   = jnp.array([1,1,1,1])
w_b   = jnp.array([1,1,1,1])
p_vec = jnp.array([1,1,1,1])
z_etc = jnp.array([1,1,1,1])
w_etc = jnp.array([1,1,1,1])
r_vec = jnp.array([1,1,1,1])
z_mat = jnp.array([[0, 1,1,1], [1,0,1,1], [1,1,0,1], [1,1,1,0]])
w_g   = jnp.array([1,1,1,1])
w_r   = jnp.array([1,1,1,1])
q_vec = jnp.array([1,1,1,1])
theta = jnp.concatenate([z_osn, w_osn, z_b, w_b, p_vec, z_etc, w_etc, r_vec, z_mat.flatten(), w_g, w_r, q_vec])


# random initialization
key = jax.random.PRNGKey(0)
n_params = 8 * mitral_no + mitral_no * mitral_no + 3 * mitral_no  # z_osn, w_osn, z_b, w_b, p_vec, z_etc, w_etc, r_vec, z_mat, w_g, w_r, q_vec
theta_star = jax.random.normal(key, (n_params,))

# loss: difference in outputs between theta and theta_star
def loss(theta):
    out = model(theta, u_vec)
    out_star = model(theta_star, u_vec)
    return jnp.sum((out - out_star)**2)

# Hessian at theta_star
h = hessian(loss)(theta_star)

# eigen-decomposition
u, s, vt = jnp.linalg.svd(h)

np.savetxt("svals.txt", np.atleast_2d(np.array(s))) #saves singular values in a file svals.txt

######################
u_np = np.array(u)
s_np = np.array(s)

df = pd.DataFrame(
    u_np,  # u has shape (N, N)
    columns=[f"u{i+1}" for i in range(u_np.shape[1])]
)
df.insert(0, "singular_value", s_np)  # insert s as first column
print(df)
######################

# move along smallest eigenvector
delta = u[:, -1]
theta = theta_star + delta

l_star = loss(theta_star)
l = loss(theta)

print(f"l_star = {l_star}, l = {l}")

# Create parameter names based on actual model structure
param_names = []
for name in ['z_osn', 'w_osn', 'z_b', 'w_b', 'p_vec', 'z_etc', 'w_etc', 'r_vec']:
    for i in range(mitral_no):
        param_names.append(f'{name}[{i}]')
# z_mat (flattened)
for i in range(mitral_no):
    for j in range(mitral_no):
        param_names.append(f'z_mat[{i},{j}]')
# w_g, w_r, q_vec
for name in ['w_g', 'w_r', 'q_vec']:
    for i in range(mitral_no):
        param_names.append(f'{name}[{i}]')

# DataFrame - rows: parameters, columns: initial, final, delta
df = pd.DataFrame({
    'Initial': theta_star,
    'Final': theta,
    'Delta': delta
}, index=param_names)

print("\nParameter Comparison Table:")
print(df)