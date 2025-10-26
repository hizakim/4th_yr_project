import numpy as np
import matplotlib.pyplot as plt

# fixed w1
w1 = 0.5
# sweep over these w2 values
w2_values = [0.2, 0.4, 0.6, 0.8, 1.0]

dz1_vals = np.arange(0, 1.1, 0.1)


for w2 in w2_values:
    dz2_vals = -(w1 / w2) * dz1_vals
    plt.plot(dz1_vals, dz2_vals, 'o-', label=f"w2={w2}")

plt.xlabel("dz1")
plt.ylabel("dz2")
plt.title("Neutral directions for different w2 (w1=0.5)")
plt.legend()
plt.grid(True)
plt.show()
