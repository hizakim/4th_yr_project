from TOP import mitral_output, from_y_next_step
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

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

# base theta (24 parameters)
theta_base = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(), alpha, wcor, wy, wgranule])

# compute y_star
y_star = mitral_output(theta_base, x, m, cortex)
print("y_star =", y_star)

#========================
# Heat map
w1 = np.linspace(0, 1, 10)
w2 = np.linspace(0, 1, 10)
W1, W2 = np.meshgrid(w1, w2)

Y = np.zeros_like(W1)

for i in range(len(w1)):
    print('i=',i)
    for j in range(len(w2)):
        theta = theta_base.at[:].get()

        # modify wr[0] and wb[0]
        theta = theta.at[0].set(w1[i])   # wr[0]
        theta = theta.at[2].set(w2[j])   # wb[0]

        y_val = mitral_output(theta, x, m, cortex)
        Y[i, j] = float(y_val[0])#take first value of y

plt.figure(figsize=(10, 6))
plt.contourf(W1, W2, Y, levels=50, cmap='viridis')
plt.colorbar(label='y[0]')
plt.xlabel('wr[0]')
plt.ylabel('wb[0]')
plt.show()
