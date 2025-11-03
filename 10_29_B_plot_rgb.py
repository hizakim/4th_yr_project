import numpy as np
import matplotlib.pyplot as plt

def sigma(x):
    return 1 / (1 + np.exp(-x))

def mitral_dynamics(x, alpha, wr, wb, Wg, Wr, Wm, n_iter=10):
    """
    Simulate iterative convergence of g, r, b, and y.
    Returns the time series of each variable.
    """
    m = len(x)
    g = np.ones(m)
    
    # Storage arrays for plotting
    g_hist = np.zeros((n_iter, m))
    r_hist = np.zeros((n_iter, m))
    b_hist = np.zeros((n_iter, m))
    y_hist = np.zeros((n_iter, m))

    for t in range(n_iter):
        r = sigma(alpha * x - Wg @ g)
        b = sigma(alpha * x)
        g = sigma(alpha * x - b + Wr @ r)
        y = sigma(alpha * x + wr * r - wb * b - Wm @ g)

        g_hist[t] = g
        r_hist[t] = r
        b_hist[t] = b
        y_hist[t] = y

    return g_hist, r_hist, b_hist, y_hist


# === Parameters ===
m = 3
x = np.array([1.0, 1.0, 1.0])
alpha = np.array([0.8, 0.8, 0.8])
wr = np.array([0.8, 0.8, 0.8])
wb = np.array([0.8, 0.8, 0.8])

Wg = np.array([[0, 0.8, 0.8],
               [0.8, 0, 0.8],
               [0.8, 0.8, 0]])

Wm = np.array([[0, 0.8, 0.8],
               [0.8, 0, 0.8],
               [0.8, 0.8, 0]])

Wr = np.array([[0, 0.8, 0.8],
               [0.8, 0, 0.8],
               [0.8, 0.8, 0]])

# === Run simulation ===
g_hist, r_hist, b_hist, y_hist = mitral_dynamics(x, alpha, wr, wb, Wg, Wr, Wm, n_iter=20)

# === Plot results ===
time = np.arange(g_hist.shape[0])

fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

for i in range(m):
    axs[0].plot(time, g_hist[:, i], label=f'Cell {i+1}')
    axs[1].plot(time, r_hist[:, i], label=f'Cell {i+1}')
    axs[2].plot(time, b_hist[:, i], label=f'Cell {i+1}')
    axs[3].plot(time, y_hist[:, i], label=f'Cell {i+1}')

axs[0].set_ylabel("g")
axs[1].set_ylabel("r")
axs[2].set_ylabel("b")
axs[3].set_ylabel("y")
axs[3].set_xlabel("Iteration step")
for ax in axs:
    ax.legend(loc='best')
    ax.grid(True)

plt.suptitle("Mitral Model Dynamics (Iterative Convergence)")
plt.tight_layout()
plt.show()
