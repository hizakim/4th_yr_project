
# dy/dt 27_10 but with our model function 
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt

def naka_rushton(x, n=2.0, sigma=1.0):
    return (x**n) / (x**n + sigma**n)

phi = naka_rushton

mitral_no = 3
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

tau = 1.0
def dydt(y, t, theta, u_vec):
    f = model(theta, u_vec)
    return (-y + f) / tau

# initial conditions
u_vec = jnp.array([0.2, 0.5, 0.8])
z_osn = jnp.array([1,1,1])
w_osn = jnp.array([1,1,1])
z_b   = jnp.array([1,1,1])
w_b   = jnp.array([1,1,1])
p_vec = jnp.array([1,1,1])
z_etc = jnp.array([1,1,1])
w_etc = jnp.array([1,1,1])
r_vec = jnp.array([1,1,1])
z_mat = jnp.array([[0, 1,1], [1,0,1], [1,1,0]])
w_g   = jnp.array([1,1,1])
w_r   = jnp.array([1,1,1])
q_vec = jnp.array([1,1,1])
theta = jnp.concatenate([z_osn, w_osn, z_b, w_b, p_vec, z_etc, w_etc, r_vec, z_mat.flatten(), w_g, w_r, q_vec])

y0 = jnp.zeros_like(u_vec)  

# time vector
#t = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
t = jnp.linspace(0.0, 5.0, 200)

# Integrate using JAX odeint
y_time = odeint(dydt, y0, t, theta, u_vec)  # y_time.shape = (len(t), len(u_vec))

plt.figure(figsize=(8,4))
for i in range(y_time.shape[1]):
    plt.plot(t, y_time[:,i], label=f"y[{i}]")
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Output y over time')
plt.legend()
plt.grid(True)
plt.show()
