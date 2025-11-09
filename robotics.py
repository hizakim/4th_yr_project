import numpy as np
import matplotlib.pyplot as plt

V  = np.array([0,3.9,9.8,17.1,22.3,25,26.7,27.4,27.6])
t = np.array([0.3,0.298,0.295,0.276,0.253,0.225,0.184,0.149,0.053])

#K = 0.0012715
K = 0.0013

t0 =0.3

t= t0 * (1-K*V**2)**(1/6)

v_model = np.linspace(-30, 30, 100)
t_model = t0 * (1 - K * v_model**2)**(1/6)

plt.figure(figsize=(5, 4))
plt.plot(V, t, 'o', label='data')
plt.plot(v_model, t_model, '-', label='model')
plt.xlabel('V (kV)')
plt.ylabel('t (mm)')
plt.legend()
plt.grid(True)
plt.show()

################