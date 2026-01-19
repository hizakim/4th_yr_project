from TOP import mitral_output, from_y_next_step
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

#===========================
# Participation ratio
def participation_ratio(s):
    s2 = jnp.sum(s**2)
    s1 = jnp.sum(jnp.abs(s))
    return (s1*s1) / (s2)

#===========================
# Base model parameters
m = 2
x = jnp.ones(m)
cortex = jnp.array([0.5, 0.5])

wr = jnp.array([0.2, 0.2])
wb = jnp.array([0.2, 0.2])
Wgr = jnp.array([[0.2, 0.2], [0.2, 0.2]])
Wrg = jnp.array([[0.2, 0.2], [0.2, 0.2]])
Wgm = jnp.array([[0.2, 0.2], [0.2, 0.2]])
alpha = jnp.array([0.8, 0.8])
wcor = jnp.array([0.2, 0.2])
wy = jnp.array([0.2, 0.2])
wgranule = jnp.array([0.2, 0.2])

theta_base = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(), alpha, wcor, wy, wgranule])

#===========================
# Sweep wr[0] vs wb[0]
grid = 10
wr_vals = np.linspace(0, 1, grid)
wb_vals = np.linspace(0, 1, grid)
WR, WB = np.meshgrid(wr_vals, wb_vals)

PR = np.zeros_like(WR)

#===========================
# Compute PR on grid
for i in range(grid):
    print('i=',i)
    for j in range(grid):

        theta = theta_base

        # modify wr[0] and wb[0]
        theta = theta.at[0].set(WR[i, j])  # wr[0]
        theta = theta.at[2].set(WB[i, j])  # wb[0]

        # compute g,u,s at this theta
        g, u, s = from_y_next_step(theta, x, m, cortex)

        # compute participation ratio
        PR[i, j] = float(participation_ratio(s))

#===========================
# Plot heatmap
plt.figure(figsize=(9, 6))
plt.contourf(WR, WB, PR, levels=50, cmap='viridis')
plt.colorbar(label='Participation Ratio')
plt.xlabel('wr[0]')
plt.ylabel('wb[0]')
plt.tight_layout()
plt.show()
