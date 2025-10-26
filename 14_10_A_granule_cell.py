'''
Python simulation that shows transient dynamics after a step change in mitral input
hE
'''
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Parameters
# ------------------------
N_E = 5  # number of mitral cells
N_I = 3  # number of granule cells

# Initial external input to mitral cells
h_E0 = np.array([0.0, 0.1, 0.2, 0.3, 0.1])

# Step change at t_step
h_E_step = np.array([0.3, 0.8, 0.6, 0.9, 0.7])
t_step = 50  # timestep at which input changes

# Granule input from cortex
h_I = np.array([0.0, 0.5, 0.7])

# Weight matrices
np.random.seed(0)
M_IE = 2.0 * np.random.rand(N_I, N_E)    # mitral -> granule (excitatory)
M_EI = -3.0 * np.random.rand(N_E, N_I)   # granule -> mitral (inhibitory)

def f(x):
    return np.tanh(x)
    # return np.maximum(0, np.tanh(x))  # Ensure non-negative firing rates

# Time constants and integration
tau_E = 0.01  # Faster excitatory dynamics
tau_I = 0.08  # Slower inhibitory dynamics
dt = 0.001
T = 2.0  # total time in seconds - extended to see oscillations better
steps = int(T / dt)

# ------------------------
# Initialize variables
# ------------------------
v_E = np.zeros((steps, N_E))
v_I = np.zeros((steps, N_I))

v_E_curr = np.zeros(N_E)
v_I_curr = np.zeros(N_I)

# ------------------------
# Simulation loop
# ------------------------
for t in range(steps):
    # input
    if t < t_step:
        h_E = h_E0
    else:
        h_E = h_E_step

    # Differential updates
    dv_E = (-v_E_curr + f(h_E + M_EI @ v_I_curr)) * dt / tau_E
    dv_I = (-v_I_curr + f(h_I + M_IE @ v_E_curr)) * dt / tau_I

    v_E_curr += dv_E
    v_I_curr += dv_I

    # Store
    v_E[t, :] = v_E_curr
    v_I[t, :] = v_I_curr

# ------------------------
# Plotting
# ------------------------
time = np.arange(steps) * dt

plt.figure(figsize=(12,6))
for i in range(N_E):
    plt.plot(time, v_E[:, i], label=f'Mitral {i+1}')
plt.title('Mitral Cell Firing Rates (Transient Response)')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (v_E)')
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
for i in range(N_I):
    plt.plot(time, v_I[:, i], label=f'Granule {i+1}')
plt.title('Granule Cell Firing Rates (Transient Response)')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (v_I)')
plt.legend()
plt.show()
