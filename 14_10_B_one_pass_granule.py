'''
One-pass sequential solution of mitral-granule network

use equations
dv_E/dt = (-v_E + f_E(h_E + M_EI * v_I))/tau_E
dv_I/dt = (-v_I + f_I(h_I + M_IE * v_E))/tau_I

where M_EI = - K * M_IE
K is a constant

f_I is a threshold function
f_E is a tanh function

there are N_E mitral cells and N_I inhibitory granule cells

The train of thought is that initially v_E and v_I are zero
Then input is h_E and h_I, and we solve for v_E : tau_E * dv_E/dt = f_E(h_E)
Using v_E, we solve for v_I : tau_I * dv_I/dt = -v_I + f_I(h_I + M_IE * v_E)
Using v_I, we solve for v_E : tau_E * dv_E/dt = -v_E + f_E(h_E + M_EI * v_I)
Output is the final value of v_E
'''

import numpy as np
import matplotlib.pyplot as plt

N_E = 5  # number of mitral cells
N_I = 3  # number of granule cells

# inputs (moderate values to see enhancement effect)
h_E = np.array([0.8, 0.6, 0.9, 0.5, 0.7])  # input to mitral cells (varied strengths)
h_I = np.array([0.2, 0.15, 0.25])  # cortical input to granule cells

# Time constants
tau_E = 0.02  # mitral cell time constant
tau_I = 0.05  # granule cell time constant

# Weight matrices
np.random.seed(42)
M_IE = 0.8 * np.random.rand(N_I, N_E)  # mitral -> granule (excitatory)
K = 0.5  # inhibition strength - balanced to show enhancement without suppressing too much
M_EI = -K * M_IE.T  # granule -> mitral (inhibitory, transpose for correct dimensions)

# Time integration parameters
dt = 0.001  # time step
T_relax = 0.2  # relaxation time for each stage

# Activation functions
def f_E(x):
    return np.maximum(0, np.tanh(x))

def f_I(x):
    for i in range(len(x)):
        if x[i] > 0.7:
            x[i] = x[i]
        else:
            x[i] = 0
    return x

# ------------------------

# Stage 0: Initialize to zero
v_E = np.zeros(N_E)
v_I = np.zeros(N_I)
print("\nStage 0 (Initial):")
print(f"  v_E = {v_E}")
print(f"  v_I = {v_I}")

print("\nStage 1: Solve v_E from h_E only")
steps = int(T_relax / dt)
for _ in range(steps):
    dv_E = f_E(h_E) * dt / tau_E
    v_E += dv_E
print(f"  v_E = {np.round(v_E, 4)}")

print("\nStage 2: Solve v_I using h_I and v_E from Stage 1")
for _ in range(steps):
    dv_I = (-v_I + f_I(h_I + M_IE @ v_E)) * dt / tau_I
    v_I += dv_I
print(f"  v_I = {np.round(v_I, 4)}")

print("\nStage 3: Solve v_E with inhibition from v_I")
for _ in range(steps):
    dv_E = (-v_E + f_E(h_E + M_EI @ v_I)) * dt / tau_E
    v_E += dv_E
print(f"  v_E (FINAL) = {np.round(v_E, 4)}")

print("FINAL OUTPUT")
print(f"Mitral cells (v_E): {np.round(v_E, 4)}")
