import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

def y(theta):
    w, z = theta[:n], theta[n:]
    y = jnp.sum(w * jnp.tanh(z))
    return y

n=1 # scalar case!!
key = jax.random.PRNGKey(50)
theta = jax.random.uniform(key, (2*n,), minval=0.0, maxval=1.0)
y0 = y(theta)

grad_y = jax.grad(y) # grad of y wrt theta
g = grad_y(theta)

hess_y = jax.hessian(y) #Hessian function
H = hess_y(theta) #evaluate at theta

# -------------------------

var_range = 5.0
delta_range = jnp.linspace(-var_range, var_range, 100)
DW, DZ = jnp.meshgrid(delta_range, delta_range) #DW and DZ are both 100x100

DeltaY = jnp.zeros_like(DW)

# compute Taylor approx at each grid point
for i in range(DW.shape[0]):
    for j in range(DW.shape[1]):
        delta_w = jnp.array([DW[i,j]])
        delta_z = jnp.array([DZ[i,j]])
        delta = jnp.concatenate([delta_w, delta_z])
        DeltaY = DeltaY.at[i,j].set(g @ delta + 0.5 * delta @ H @ delta)

# -------------------------
# plot heatmap
plt.figure(figsize=(6,5))
plt.imshow(np.array(DeltaY),
           extent=[delta_range.min(), delta_range.max(), delta_range.min(), delta_range.max()],
           origin='lower', cmap='bwr', aspect='auto')
plt.colorbar(label=r'$\Delta y$')
plt.xlabel(r'$\Delta w$')
plt.ylabel(r'$\Delta z$')
plt.title(f'Heatmap of Δy (Taylor approx) for n=1, w={theta[0]:.3f}, z={theta[1]:.3f}')

# -------------------------

#plot line of equation where DeltaY=0
O = H[:n, :n]     # top-left block
A = H[:n, n:]     # top-right block
B = H[n:, n:]     # bottom-right block

delta_z_line = np.linspace(-var_range, var_range, 200)
delta_w_line = - (g[1]*delta_z_line + 0.5*B[0,0]*delta_z_line**2) / (g[0] + A[0,0]*delta_z_line)

# overlay zero-change line
plt.plot(delta_w_line, delta_z_line, color='black', linewidth=2, label=r'$\Delta y=0$')
plt.legend()
plt.xlim(-var_range, var_range)
plt.ylim(-var_range, var_range)


# -------------------------
#plot stationary point
delta_stat = -2 * jnp.linalg.inv(H) @ g
plt.scatter(delta_stat[0], delta_stat[1], color='yellow', s=80, edgecolor='k', label='Stationary point')
plt.show()

print(delta_stat)
