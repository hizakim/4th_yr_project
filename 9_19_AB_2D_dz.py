import numpy as np
import matplotlib.pyplot as plt

# fixed w1
w1 = 0.5
# sweep over these w2 values
w2_values = [0.2, 0.4, 0.6, 0.8, 1.0]

# Example z for computing t
z = np.array([0.3, 0.7])
t = np.tanh(z)

dz1_vals = np.arange(0, 1.1, 0.1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# --- Plot δz ---
for w2 in w2_values:
    dz2_vals = -(w1 / w2) * dz1_vals
    ax1.plot(dz1_vals, dz2_vals, 'o-', label=f"w2={w2}")

ax1.set_xlabel("δz1")
ax1.set_ylabel("δz2")
ax1.set_title("Neutral δz directions (w1=0.5)")
ax1.legend()
ax1.grid(True)

# --- Plot δw ---
for w2 in w2_values:
    dz2_vals = -(w1 / w2) * dz1_vals
    # compute δw
    dw1 = w1 * t[0] * dz1_vals
    dw2 = w2 * t[1] * dz2_vals
    ax2.plot(dw1, dw2, 'o-', label=f"w2={w2}")

ax2.set_xlabel("δw1")
ax2.set_ylabel("δw2")
ax2.set_title("Corresponding δw directions")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
