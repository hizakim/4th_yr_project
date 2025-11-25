import numpy as np
import matplotlib.pyplot as plt

# Common parameters
Vin = 5.0   # Input Voltage (V)
Rk = 10.0   # kOhm (Known resistance)
ADC_MAX_LEVEL = 31  # 0 to 31
Delta_L = np.linspace(0.0, 20.0, 200)  # Extension from 0 to 20 mm

# Sensor parameters (A, B, C)
sensors = {
    "Sensor A": {"R0": 2, "L0": 50.0, "G": 2.4, "Rcp": 0.1},
    "Sensor B": {"R0": 3.5, "L0": 40.0, "G": 1.6, "Rcp": 0.1},
    "Sensor C": {"R0": 3, "L0": 40.0, "G": 0.6, "Rcp": 0.2},
}


plt.figure(figsize=(5, 4))

for label, params in sensors.items():
    R0 = params["R0"]
    L0 = params["L0"]
    G = params["G"]
    Rcp = params["Rcp"]

    # 1. Measured resistance
    Rm = R0 * (1 + G * (Delta_L / L0)) + 2 * Rcp

    # 2. Output voltage
    V_out = Vin * Rk / (Rk + Rm)

    # 3. Quantized ADC output
    M = np.round(ADC_MAX_LEVEL * (V_out / Vin))

    # Plot
    plt.plot(Delta_L, M, drawstyle='steps-post', linewidth=2, label=label)

# Labels, title, and legend
plt.xlabel('ΔL (mm)')
plt.ylabel('M (0 to 31)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

###################

# Plot V_out vs Delta_L for each sensor
plt.figure(figsize=(5, 4))

for label, params in sensors.items():
    R0 = params["R0"]
    L0 = params["L0"]
    G = params["G"]
    Rcp = params["Rcp"]

    # Measured resistance
    Rm = R0 * (1 + G * (Delta_L / L0)) + 2 * Rcp

    # Output voltage
    V_out = Vin * Rm / (Rk + Rm)

    # Plot
    plt.plot(Delta_L, V_out, linewidth=2, label=label)

plt.xlabel("ΔL (mm)")
plt.ylabel("V_out (V)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Plot V_out vs Rm for Delta_L = 10 mm for each sensor
plt.figure(figsize=(5, 4))

Delta_L_10 = 10  # mm

for label, params in sensors.items():
    R0 = params["R0"]
    L0 = params["L0"]
    G = params["G"]
    Rcp = params["Rcp"]

    # Measured resistance at Delta_L = 10mm
    Rm = R0 * (1 + G * (Delta_L_10 / L0)) + 2 * Rcp

    # Output voltage at Delta_L = 10mm
    V_out = Vin * Rm / (Rk + Rm)

    # Plot single point for each sensor
    plt.plot(Rm, V_out, marker='o', markersize=8, label=label)

plt.xlabel("Rm at ΔL = 10 mm (Ω)")
plt.ylabel("V_out (V)")
plt.title("V_out vs Rm at ΔL = 10 mm")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
