# back to simple single mitral cell, add ETC
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Naka-Rushton function
def naka_rushton(x, n=2.0, sigma=1):
    return (x**n) / (x**n + sigma**n)

phi = naka_rushton

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

output = model(theta, u_vec)
print(f"Model output is {output}")

