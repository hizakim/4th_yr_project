import numpy as np
import matplotlib.pyplot as plt

## simple fixed point 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Update function for (r, b, g)
def mitral_update(state, x, alpha, Wgr, Wrg):
    m = x.shape[0]
    r, b, g = state[:m], state[m:2*m], state[2*m:]

    r_new = sigmoid(alpha * x - Wgr @ g)
    b_new = sigmoid(alpha * x)  
    g_new = sigmoid(alpha * x - b_new + Wrg @ r_new)

    return np.concatenate([r_new, b_new, g_new])

# Parameters
m = 3
x = np.array([1.0, 2.0, 3.0])
alpha = np.array([0.8, 0.8, 0.8])

Wgr = np.array([[0, 0.8, 0.8],
                [0.8, 0, 0.8],
                [0.8, 0.8, 0]])

Wrg = np.array([[0, 0.8, 0.8],
                [0.8, 0, 0.8],
                [0.8, 0.8, 0]])

# Initial guess: all zeros
state = np.ones(3 * m)

# Store values over iterations
r_vals, b_vals, g_vals = [], [], []

for _ in range(5): 
    r_vals.append(state[:m])
    b_vals.append(state[m:2*m])
    g_vals.append(state[2*m:])
    state = mitral_update(state, x, alpha, Wgr, Wrg)

r_vals = np.array(r_vals)
b_vals = np.array(b_vals)
g_vals = np.array(g_vals)

# Plotting
plt.figure(figsize=(8, 5))
for i in range(m):
    plt.plot(r_vals[:, i], label=f"r[{i}]", linestyle='-')
    plt.plot(b_vals[:, i], label=f"b[{i}]", linestyle='--')
    plt.plot(g_vals[:, i], label=f"g[{i}]", linestyle=':')

plt.xlabel("Iteration")
plt.ylabel("Signal")
plt.legend()
plt.grid(True)
plt.show()
