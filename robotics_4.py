import numpy as np
import matplotlib.pyplot as plt

# --- Given Parameters ---
n1 = 1.49  # Refractive index of the PMMA fibre (core)

# --- Data Generation ---
# Generate n2 values in the range [1.0, 1.5]
# We only calculate the angle where n2 <= n1 (up to 1.49)
n2_values = np.linspace(1.0, 1.49, 1000)

# 1. Calculate the argument for arcsin: Numerical Aperture (NA)
# NA = sqrt(n1^2 - n2^2)
na_squared = n1**2 - n2_values**2

# Ensure the values are not slightly negative due to floating point precision
# (though linspace limits n2 to n1, it's good practice)
na_squared[na_squared < 0] = 0

na = np.sqrt(na_squared)

# 2. Apply the sin(alpha) <= 1 constraint
# If NA > 1, set it to 1, as the maximum acceptance angle is 90 degrees (sin(alpha)=1).
na[na > 1.0] = 1.0

# 3. Calculate acceptance angle (alpha) in degrees
alpha_radians = np.arcsin(na)
alpha_degrees = np.degrees(alpha_radians)

# --- Specific Material Points to Mark (from calculation in part i) ---
materials = {
    "Air ($n_2=1.00$)": 1.00,
    "Silicon Rubber ($n_2=1.42$)": 1.42,
    "Fused Silica ($n_2=1.46$)": 1.46,
}

# Calculate alpha for each material for marking
material_n2 = list(materials.values())
material_na = np.sqrt(n1**2 - np.array(material_n2)**2)
# Apply the cap for sin(alpha) <= 1
material_na[material_na > 1.0] = 1.0
material_alpha_degrees = np.degrees(np.arcsin(material_na))
material_labels = list(materials.keys())

# --- Plotting ---
plt.figure(figsize=(5, 4))
# Plot the main curve
plt.plot(n2_values, alpha_degrees, label=r'$\alpha = \arcsin(\sqrt{n_1^2 - n_2^2})$', color='#1f77b4', linewidth=3)

# Add markers for specific materials
for n2, alpha, label in zip(material_n2, material_alpha_degrees, material_labels):
    # Use a consistent red marker color for all materials
    marker_color = '#d62728'
    plt.plot(n2, alpha, marker='o', markersize=8, color=marker_color, linestyle='',
             label=label)

    # Add text annotation
    if n2 == 1.0:
        plt.annotate(f'Air {alpha:.0f}°',
                     (n2, alpha), textcoords="offset points", xytext=(10, 10),
                     ha='left', fontsize=9, color=marker_color)
    else:
        plt.annotate(f'{alpha:.1f}°',
                     (n2, alpha), textcoords="offset points", xytext=(5, 15),
                     ha='left', fontsize=9, color=marker_color)


# Add details and labels
plt.xlabel('Refractive index of surrounding medium $n_2$', fontsize=12)
plt.ylabel('Acceptance angle $\u03B1$ (Degrees)', fontsize=12)

# Set axes limits for clarity
plt.xlim(1.0, 1.5)
plt.ylim(0, 100)
plt.yticks(np.arange(0, 100, 10))

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.6)

# Highlight the threshold region where alpha is 90 degrees
plt.axhline(90, color='gray', linestyle=':', linewidth=1)

# Add vertical line at n2 = n1 (legend entry only, no on-graph annotation)
plt.axvline(n1, color='black', linestyle='--', linewidth=1, alpha=0.7, label=r'$n_2 = n_1$')

plt.axhline(30, color='purple', linestyle=':', linewidth=2, alpha=0.7, label=r'$\alpha = 30^\circ$')


# Legend including curve, materials, and n2 = n1 line
plt.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.show()
