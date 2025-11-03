import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Naka-Rushton function
def naka_rushton(x, n=2.0, sigma=1):
    return (x**n) / (x**n + sigma**n)

phi = naka_rushton

n = 4
def unpack_theta(theta):
    (z_osn, z_b, z_g2, z_g3) = theta[:n]
    (w_osn, w_b, w_g2, w_g3) = theta[n:2*n]
    (p1, q2, q3) = theta[2*n:]
    return z_osn, z_b, z_g2, z_g3, w_osn, w_b, w_g2, w_g3, p1, q2, q3

def model(theta, u1=1.0, u2=1.0, u3=1.0):
    z_osn, z_b, z_g2, z_g3, w_osn, w_b, w_g2, w_g3, p1, q2, q3 = unpack_theta(theta)
    term_osn = z_osn * phi(u1 * w_osn)
    term_b   = z_b   * phi(w_b * phi(p1 * u1))
    term_g2  = z_g2  * phi(w_g2 * phi(q2 * u2))
    term_g3  = z_g3  * phi(w_g3 * phi(q3 * u3))
    return term_osn + term_b + term_g2 + term_g3

theta_base = [2.0, -0.8, -0.8, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
output = model(theta_base, 1.0, 1.0, 1.0)
print(f"Model output when theta = {theta_base} is {float(output):.6f}")




# heatmap varying two weights
def heatmap_vary(index_1, index_2, name_1, name_2, num_points=50):
    vals = jnp.linspace(0.0, 4.0, num_points)
    Z = np.zeros((num_points, num_points))

    for i, v1 in enumerate(vals):
        for j, v2 in enumerate(vals):
            theta = jnp.array(theta_base).at[index_1].set(v1)
            theta = jnp.array(theta).at[index_2].set(v2)
            Z[i, j] = float(model(theta, 1.0, 1.0, 1.0))

    plt.figure(figsize=(6,5))
    im = plt.imshow(Z, origin='lower', extent=[0, 4, 0, 4], aspect='auto', cmap='inferno')
    plt.colorbar(im, label="Model output")
    plt.xlabel(name_2)
    plt.ylabel(name_1)
    plt.title(f"Model output heatmap: vary {name_1}, {name_2}")
    plt.show()

print("Testing different weight combinations...")

heatmap_vary(4, 5, "w_osn", "w_b")
