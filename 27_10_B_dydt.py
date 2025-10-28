import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt

# dummy model - simple nonlinear
def model(theta, u_vec):
    return theta * u_vec**2

tau = 1.0

def dydt(y, t, theta, u_vec):
    f = model(theta, u_vec)
    return (-y + f) / tau

# initial conditions
theta = jnp.array([2.0, 3.0, 1.0, 4.0])  # simple theta
u_vec = jnp.array([1.0, 0.5, 2.0, 1.5])
y0 = jnp.zeros_like(u_vec)  # start at zero

# time vector
t = jnp.linspace(0.0, 5.0, 200)

# --- Integrate using JAX odeint ---
y_time = odeint(dydt, y0, t, theta, u_vec)  # y_time.shape = (len(t), len(u_vec))

# --- Pick output at a specific time ---
t_idx = 150
y_at_t = y_time[t_idx]
print(f"y at time t={t[t_idx]:.2f} is {y_at_t}")

# --- Plot ---
plt.figure(figsize=(8,4))
for i in range(y_time.shape[1]):
    plt.plot(t, y_time[:,i], label=f"y[{i}]")
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Dynamic evolution of y toward model output')
plt.legend()
plt.grid(True)
plt.show()
