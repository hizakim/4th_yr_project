import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt

# constant input from cortex

N_E = 5  # number of mitral cells
N_I = 3  # number of granule cells

# External input to mitral cells
h_E0 = jnp.array([0.0, 0.1, 0.2, 0.3, 0.1])
h_E_step = jnp.array([0.3, 0.8, 0.6, 0.9, 0.7])
t_step = 0.05  # time (in seconds) where input changes

# Granule input from cortex
h_I = jnp.array([0.0, 0.5, 0.7])

key = jax.random.PRNGKey(0)
M_IE = 2.0 * jax.random.uniform(key, (N_I, N_E)) # mitral -> granule (excitatory)
M_EI = -3.0 * jax.random.uniform(key, (N_E, N_I)) # granule -> mitral (inhibitory)


tau_E = 0.01  # faster excitatory
tau_I = 0.08  # slower inhibitory

# Activation function
#def f(x):
#    return jnp.tanh(x)
def f(x, n=2.0, sigma=1.0):
    return (x**n) / (x**n + sigma**n)

# ------------------------
# Differential equations
def network_dynamics(y, t, params):
    M_EI, M_IE, h_E0, h_E_step, h_I, tau_E, tau_I, t_step = params
    # Split y into v_E and v_I
    v_E = y[:N_E]
    v_I = y[N_E:]
    
    # Step input
    h_E = jnp.where(t < t_step, h_E0, h_E_step)
    
    # ODEs
    dv_E = (-v_E + f(h_E + M_EI @ v_I)) / tau_E
    dv_I = (-v_I + f(h_I + M_IE @ v_E)) / tau_I
    
    return jnp.concatenate([dv_E, dv_I])

# ------------------------
# Initial conditions 
y0 = jnp.zeros(N_E + N_I)
T = 1.0  # seconds
time = jnp.linspace(0, T, 2000)

params = (M_EI, M_IE, h_E0, h_E_step, h_I, tau_E, tau_I, t_step)

# ------------------------
# Solve ODEs
y = odeint(network_dynamics, y0, time, params)

# Extract results
v_E = y[:, :N_E]
v_I = y[:, N_E:]

# ------------------------
# Plot results
for i in range(N_E):
    plt.plot(time, v_E[:, i], label=f'Mitral {i+1}')
plt.axvline(t_step, color='k', linestyle='--', alpha=0.5, label='Step input')
plt.title('Mitral Cell Firing Rates (JAX odeint)')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (v_E)')
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
for i in range(N_I):
    plt.plot(time, v_I[:, i], label=f'Granule {i+1}')
plt.axvline(t_step, color='k', linestyle='--', alpha=0.5)
plt.title('Granule Cell Firing Rates (JAX odeint)')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (v_I)')
plt.legend()
plt.show()
