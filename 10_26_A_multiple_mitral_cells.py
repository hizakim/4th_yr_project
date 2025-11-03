import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Naka-Rushton function
def naka_rushton(x, n=2.0, sigma=1):
    return (x**n) / (x**n + sigma**n)

phi = naka_rushton

n = 4
mitral_no = 3

def unpack_theta(theta):
    idx = 0
    z_osn = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    w_osn = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    z_b   = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    w_b   = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    z_mat = jnp.array(theta[idx:idx+mitral_no*mitral_no]).reshape((mitral_no, mitral_no)); idx += mitral_no*mitral_no
    w_g   = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    p_vec = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    q_vec = jnp.array(theta[idx:idx+mitral_no]); idx += mitral_no
    return z_osn, w_osn, z_b, w_b, z_mat, w_g, p_vec, q_vec

def model(theta, u_vec):
    z_osn, w_osn, z_b, w_b, z_mat, w_g, p_vec, q_vec = unpack_theta(theta)
    y = z_osn * w_osn * u_vec + z_b * w_b * phi(u_vec * p_vec) + z_mat@(w_g * phi(u_vec * q_vec))
    return y

u_vec = jnp.array([1.0, 1.0, 1.0])
z_osn = jnp.array([2, 2, 2])
w_osn = jnp.array([0.8, 0.8, 0.8])
z_b   = jnp.array([-0.8, -0.8, -0.8])
w_b   = jnp.array([0.8, 0.8, 0.8])
z_mat = jnp.array([[0.0, 0.8, 0.8], [0.8, 0.0, 0.8], [0.8, 0.8, 0.0]])
w_g   = jnp.array([0.8, 0.8, 0.8])
p_vec = jnp.array([0.8, 0.8, 0.8])
q_vec = jnp.array([0.8, 0.8, 0.8])
theta = jnp.concatenate([z_osn, w_osn, z_b, w_b, z_mat.flatten(), w_g, p_vec, q_vec])

output = model(theta, u_vec)
print(f"Model output is {output}")
