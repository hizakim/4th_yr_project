import numpy as np
from scipy.optimize import curve_fit

V = np.array([0, 3.9, 9.8, 17.1, 22.3, 25.0, 26.7, 27.4, 27.6])
t = np.array([0.300, 0.298, 0.295, 0.276, 0.253, 0.225, 0.184, 0.149, 0.053])
t0 = 0.300  # initial thickness

def model(V, K):
    return t0 * (1 - K * V**2)**(1/6)

# ignore the first outlier
V_fit = V[2:]
t_fit = t[2:]

# nonlinear least squares fit
popt, pcov = curve_fit(model, V, t, p0=[0.001])
#popt, pcov = curve_fit(model, V_fit, t_fit, p0=[0.001]) # ignoring outlier
K_opt = popt[0]
print("Optimized K:", K_opt)
