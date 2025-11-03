import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Naka-Rushton function
def naka_rushton(x, n=2.0, sigma=0.5):
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
print(f"Static model output: {float(model(theta_base, 1.0, 1.0, 1.0)):.6f}")


def time_simulation(theta, tau=5.0, T=50.0, dt=0.1, u1_func=None):
    """
    Simulate τ * dy/dt + y = f(x)
    Using Euler integration: dy/dt = (f(x) - y) / τ
    """
    steps = int(T / dt)
    y = 0.0
    y_trace = np.zeros(steps)
    t_vals = np.linspace(0, T, steps)

    for i, t in enumerate(t_vals):
        u1 = u1_func(t) if u1_func else 1.0
        f_x = model(theta, u1, 1.0, 1.0)
        ydot = (f_x - y) / tau  # from τ ydot + y = f(x)
        y += ydot * dt
        y_trace[i] = y

    return t_vals, y_trace

u1_func = lambda t: 1.0 if t < 10 else 2.0
t, y_trace = time_simulation(theta_base, tau=5.0, T=50.0, dt=0.1, u1_func=u1_func)

plt.figure(figsize=(8,4))
plt.plot(t, y_trace, label="y(t)", linewidth=2)
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("Dynamics of τ·ẏ + y = f(x)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
