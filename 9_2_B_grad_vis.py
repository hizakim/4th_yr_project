'''
lets consider only first order term
pick random (w1,w2, z1,z2)
find grad, then find vector orthogonal -> this is step direction such that Delta_Y = 0

now consider n=1
and plot gradient arrow to check direction aligns to plane
'''
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def y(theta):
    n = theta.size // 2
    w, z = theta[:n], theta[n:]
    y = jnp.sum(w * jnp.tanh(z))
    return y

n=4
key = jax.random.PRNGKey(3)
theta = jax.random.uniform(key, (2*n,), minval=0.0, maxval=1.0)
y0 = y(theta)
print('y0 =',y0)
print('theta =',theta)

grad_y = jax.grad(y) # grad of y wrt theta
g = grad_y(theta)
g_norm = jnp.dot(g, g)

if g_norm < 1e-10:
    print('zero gradient - any small step keeps y unchanged')
    exit()

# find vector orthogonal to g
v = jax.random.normal(key, (2*n,))
orth_dir = v - jnp.dot(v, g) / jnp.dot(g, g) * g
orth_dir = orth_dir / jnp.linalg.norm(orth_dir)  # normalise
print('orth_dir =',orth_dir)


##################
#CHECK - first order vs exact change

# take a small step
alpha = 0.1
theta_new = theta + alpha * orth_dir

# check deltay
dy_exact = y(theta_new) - y(theta)
dy_linear = jnp.dot(g, alpha * orth_dir)  # should be approx0
print(f"Δy (linear approx) = {dy_linear:.6f}, Δy (actual) = {dy_exact:.6f}")

##################
# 2D PLOT

# # create 2D grid
# w_vals = jnp.linspace(0, 1, 50)
# z_vals = jnp.linspace(0, 1, 50)
# W, Z = jnp.meshgrid(w_vals, z_vals)
# Y = W * jnp.tanh(Z)

# plt.figure(figsize=(8,6))
# # contour plot
# plt.contour(W, Z, Y, levels=20, cmap='seismic')

# # random point
# plt.plot(theta[0], theta[1], 'ko', label='Random point')

# # gradient arrow (red)
# plt.arrow(theta[0], theta[1], g[0]*0.2, g[1]*0.2, color='r', head_width=0.03, label='Gradient')

# # orthogonal arrow (blue)
# plt.arrow(theta[0], theta[1], orth_dir[0]*0.2, orth_dir[1]*0.2, color='b', head_width=0.03, label='Orthogonal')

# plt.xlabel("w")
# plt.ylabel("z")
# plt.title("Contour plot of y(w,z)")
# plt.legend()
# plt.show()
