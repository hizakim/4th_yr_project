import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
def func(x):
    return x**2 * (3-2*x)

# Generate x values
x = np.linspace(0, 2, 200)  # Adjust range as needed

# Compute y values
y = func(x)

# Create the plot
plt.plot(x, y, label=r'$f(x) = x^2(3x-2)$')
plt.title("Plot of f(x) = x^2 * (3x - 2)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()