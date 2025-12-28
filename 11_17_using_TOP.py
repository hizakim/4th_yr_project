## using TOP.py

from TOP import from_y_next_step, mitral_output, possible_deltas
import jax.numpy as jnp

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

theta_star = jnp.concatenate([wr, wb, Wgr.flatten(), Wrg.flatten(), Wgm.flatten(), alpha, wcor, wy, wgranule])
y_star = mitral_output(theta_star, x, m, cortex)

g, u, s = from_y_next_step(theta_star, x, m, cortex)

# delta1 = u[:, -1]
# delta2 = u[:, -2]
# delta3 = u[:, -3]
# theta1 = theta_star + delta1
# theta2 = theta_star + delta2
# theta3 = theta_star + delta3

# y1 = mitral_output(theta1, x, m, cortex)
# y2 = mitral_output(theta2, x, m, cortex)
# y3 = mitral_output(theta3, x, m, cortex)

# print("y_star =", y_star)
# print("y1 =", y1)
# print("y2 =", y2)
# print("y3 =", y3)

# print("s =", s)

deltas = possible_deltas(g,u,s)
print("Deltas =", deltas)

print(y_star)
for delta in deltas:
    theta = theta_star + delta
    y = mitral_output(theta, x, m, cortex)
    print("y =", y)
