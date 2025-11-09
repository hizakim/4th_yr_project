import numpy as np
import matplotlib.pyplot as plt

#--- Sensor B Parameters (from the table) ---
# R0 = 3.5  # kOhm (Resistance at L0)
# L0 = 40.0 # mm (Unstretched length)
# G = 1.6   # (Gauge Factor)
# Rcp = 0.1 # kOhm (Contact Resistance)

# --- Sensor A Parameters (from the table) ---
# R0 = 2  # kOhm (Resistance at L0)
# L0 = 50.0 # mm (Unstretched length)
# G = 2.4  # (Gauge Factor)
# Rcp = 0.1 # kOhm (Contact Resistance)

#sensor C
R0 = 3  # kOhm (Resistance at L0)
L0 = 40.0 # mm (Unstretched length)
G = 0.6  # (Gauge Factor)
Rcp = 0.2 # kOhm (Contact Resistance)




Vin = 5.0   # V (Input Voltage)
Rk = 10.0   # kOhm (Known resistance)
ADC_MAX_LEVEL = 31 # (Quantization range is 0 to 31, total 32 levels)

# --- Range of Extension (Delta L) ---
# We analyze the range from 0 mm to 20 mm
Delta_L = np.linspace(0.0, 20.0, 100) # 100 points for a smooth line/staircase

# 1. Calculate Measured Resistance (Rm)
# Rm = R0 * (1 + G * (Delta_L / L0)) + 2 * Rcp
Rm = R0 * (1.0 + G * (Delta_L / L0)) + 2.0 * Rcp

# 2. Calculate Output Voltage (V_out)
# V_out = V_in * Rm / (Rk + Rm)
V_out = Vin * Rm / (Rk + Rm)

# 3. Calculate Quantized Measurement (M)
# M = round(31 * (V_out / V_in))
# The term V_out / V_in gives the fraction of the input voltage
M = np.round(ADC_MAX_LEVEL * (V_out / Vin))

# --- Plotting M vs. Delta L ---
plt.figure(figsize=(10, 6))

# Plot the quantized data points
plt.plot(Delta_L, M, drawstyle='steps-post', color='darkorange', linewidth=2.5,
         label='Quantized ADC Output (M)')

# Labeling and aesthetics
plt.xlabel('$\Delta L$ (mm)')
plt.ylabel(f'M (0 to {ADC_MAX_LEVEL})')
plt.yticks(np.arange(M.min(), M.max() + 1, 1)) # Ensure integer ticks on y-axis
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()





# --- Print key information ---
print(f"--- Sensor B (ΔL=0 to 20mm) Analysis ---")
print(f"R_m starts at: {Rm[0]:.3f} kΩ (at ΔL=0mm)")
print(f"R_m ends at:   {Rm[-1]:.3f} kΩ (at ΔL=20mm)")
print(f"M starts at:   {M[0]:.0f}")
print(f"M ends at:     {M[-1]:.0f}")
print(f"Total steps (resolution): {M[-1] - M[0]:.0f} levels used (out of {ADC_MAX_LEVEL+1})")

# Calculate the length change corresponding to one M step (average resolution)
total_M_change = M[-1] - M[0]
avg_resolution_mm_per_step = (Delta_L[-1] - Delta_L[0]) / total_M_change
print(f"Average resolution: approx. {avg_resolution_mm_per_step:.3f} mm per ADC step")
