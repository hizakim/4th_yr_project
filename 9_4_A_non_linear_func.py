import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0, 1, 200)

# Define functions
f1 = np.tanh(z)
f2 = np.tanh(2*(z+1))
f3 = z**2
f4 = z**3
f5 = np.sqrt(z)

# Plot all on the same figure
plt.figure(figsize=(8,6))
plt.plot(z, f1, label="tanh(z)")
plt.plot(z, f2, label="tanh(2(z+1))")
plt.plot(z, f3, label="z^2")
plt.plot(z, f4, label="z^3")
plt.plot(z, f5, label="sqrt(z)")

plt.xlabel("z")
plt.ylabel("f(z)")
plt.title("Activation functions over z [0,1]")
plt.legend()
plt.grid(True)
plt.show()