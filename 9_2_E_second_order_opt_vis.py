import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.scipy.optimize import minimize


'''
plot final delta y values for differetn starting points in w1,w2,z1,z2
'''

def y_func(theta):
    n = theta.size//2
    w, z = theta[:n], theta[n:]
    return jnp.sum(w*jnp.tanh(z))

def loss(delta, g, H):
    Delta_y = g @ delta + 0.5 * delta @ H @ delta
    return Delta_y**2

n = 2
grad_y = jax.grad(y_func)
hess_y = jax.hessian(y_func)

#inital guess of delta
step = 1 ### CAN SET THIS VALUE - inital step
delta_init = jnp.full((2*n,), step)

#========================
grid_size = 8
w1 = np.linspace(0,1,grid_size)
w2 = 1
z1 = np.linspace(0,1,grid_size)
z2 = 1

W1, Z1 = np.meshgrid(w1,z1)
DeltaY_grid = np.zeros_like(W1)

for i in range(grid_size):
    print('i=',i)
    for j in range(grid_size):
        theta = jnp.array([W1[i,j], w2, Z1[i,j], z2])
        g = grad_y(theta)
        H = hess_y(theta)

        # wrap loss with g, H fixed
        local_loss = lambda delta: loss(delta, g, H)

        res = minimize(local_loss, delta_init, method='BFGS', tol=1e-5, options={'maxiter': 50})
        DeltaY_grid[i, j] = res.fun #store final delta_y value

# plot heatmap
plt.figure()
plt.contourf(W1, Z1, DeltaY_grid, levels=50, cmap="plasma")
plt.colorbar(label=r'$\Delta y$')
plt.xlabel(r'$w_1$')
plt.ylabel(r'$z_1$')
plt.title(r'Heatmap of $\Delta y$ after optimisation')
plt.show()
