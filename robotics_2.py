import numpy as np
import matplotlib.pyplot as plt



#robot A_1
R0 = 2
L0 = 50
G =2.4
Rcp=0 #0.1
Delta_L = np.linspace(0, 20, 100)
Rm = R0*(1+G*(Delta_L/L0) +2*Rcp)

#robot A_2
R0 = 2
L0 = 50
G =2.4
Rcp=0.1
Delta_L = np.linspace(0, 20, 100)
Rm = R0*(1+G*(Delta_L/L0) +2*Rcp)

#robot B_1
R0 = 3.5
L0 = 40
G =1.6
Rcp=0 #0.1
Delta_L = np.linspace(0, 20, 100)
Rm = R0*(1+G*(Delta_L/L0) +2*Rcp)

#robot B_2
R0 = 3.5
L0 = 40
G =1.6
Rcp=0.1
Delta_L = np.linspace(0, 20, 100)
Rm = R0*(1+G*(Delta_L/L0) +2*Rcp)

#robot C_1
R0 = 3
L0 = 40
G =0.6
Rcp=0
Delta_L = np.linspace(0, 20, 100)
Rm = R0*(1+G*(Delta_L/L0) +2*Rcp)

#robot C_2
R0 = 3
L0 = 40
G =0.6
Rcp=0.2
Delta_L = np.linspace(0, 20, 100)
Rm = R0*(1+G*(Delta_L/L0) +2*Rcp)

# Store all robot settings in a list to facilitate plotting
robot_configs = [
    # (R0, L0, G, Rcp, label)
    (2,   50, 2.4, 0,   "A_1 (Rcp=0)"),
    (2,   50, 2.4, 0.1, "A_2 (Rcp=0.1)"),
    (3.5, 40, 1.6, 0,   "B_1 (Rcp=0)"),
    (3.5, 40, 1.6, 0.1, "B_2 (Rcp=0.1)"),
    (3,   40, 0.6, 0,   "C_1 (Rcp=0)"),
    (3,   40, 0.6, 0.2, "C_2 (Rcp=0.2)"),
]

Delta_L = np.linspace(0, 20, 100)
plt.figure(figsize=(8,5))
for R0, L0, G, Rcp, label in robot_configs:
    Rm = R0*(1+G*(Delta_L/L0) +2*Rcp)
    plt.plot(Delta_L, Rm, label=label)

plt.xlabel('ΔL')
plt.ylabel('Rm')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Adjust the plotting to use line style and color according to robot type and Rcp
plt.figure(figsize=(5, 4))
for R0, L0, G, Rcp, label in robot_configs:
    Rm = R0*(1+G*(Delta_L/L0) +2*Rcp)
    # Determine robot group from label (A, B, or C)
    if label.startswith("A"):
        color = 'red'
    elif label.startswith("B"):
        color = 'blue'
    elif label.startswith("C"):
        color = 'green'
    else:
        color = 'black'
    # Dashed line for Rcp=0, solid otherwise
    linestyle = '--' if Rcp == 0 else '-'
    plt.plot(Delta_L, Rm, label=label, color=color, linestyle=linestyle)

plt.xlabel('ΔL')
plt.ylabel('Rm')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
