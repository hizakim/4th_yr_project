import numpy as np

def sigma(x):
    return 1 / (1 + np.exp(-x))  

def mitral_model(x, m, alpha, wr, wb, Wg, Wr, Wm, n_iter=20):
    g = np.zeros(m)  # initial g
    
    for _ in range(n_iter):
        r = sigma(alpha * x - Wg @ g)
        b = sigma(alpha * x)
        g = sigma(alpha * x - b + Wr @ r)
    
    # once g converges, final mitral output
    y = sigma(alpha * x + wr * r - wb * b - Wm @ g)
    return y

# Example test
m = 3
x = np.array([1.0, 1.0, 1.0])
alpha = np.array([0.8, 0.8, 0.8])
wr = np.array([0.8, 0.8, 0.8])
wb = np.array([0.8, 0.8, 0.8])

# connectivity matrices
Wg = np.array([[0, 0.8, 0.8],
               [0.8, 0, 0.8],
               [0.8, 0.8, 0]])

Wm = np.array([[0, 0.8, 0.8],
               [0.8, 0, 0.8],
               [0.8, 0.8, 0]])

Wr = np.array([[0, 0.8, 0.8],
               [0.8, 0, 0.8],
               [0.8, 0.8, 0]])

# run model
y = mitral_model(x, m, alpha, wr, wb, Wg, Wr, Wm)
print("Mitral outputs:", y)

  