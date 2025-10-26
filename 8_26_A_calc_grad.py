import numpy as np

def y(w,z):
    y = np.sum(w * np.tanh(z))
    return y

n=3
w = np.random.rand(n)
z = np.random.rand(n)

y0 = y(w,z)
print('y0 =',y0)

# Gradient manually
dy_dw = np.tanh(z)
dy_dz = w * (1 - np.tanh(z)**2)

g = np.concatenate([dy_dw, dy_dz])
print('g =',g)
