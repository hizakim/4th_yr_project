## using TOP.py

from TOP import mitral_output, from_y_next_step
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

#select parameters that give target y and rgb output approx 0.5

#=====================================
y_target = jnp.array([0.5, 0.5])


no_params = 24

m = 2
x = jnp.ones(m)
cortex = jnp.array([0.5, 0.5])

# sample in theta space
no_samples = 10000

# Vectorize random generation
key = jax.random.PRNGKey(0)
theta_matrix = jax.random.uniform(key, (no_samples, no_params), minval=0.0, maxval=1.0)


#########################

# Participation ratio
def participation_ratio(s):
    s2 = jnp.sum(s**2)
    s1 = jnp.sum(s)
    return (s1*s1) / (s2)

# loss function that depends on theta and g b r
def loss_fn(theta):
    g,b,r,y = mitral_output(theta, x, m, cortex)
    return jnp.sum((y - y_target) ** 2)

@jax.jit
def gradient_descent_step(theta, lr=0.05):
    grad = jax.grad(loss_fn)(theta)
    theta = theta - lr * grad
    theta = jnp.clip(theta, 0.0, None)
    return theta

def gradient_descent(theta_init, lr=0.05, steps=100):
    theta = theta_init
    for _ in range(steps):
        theta = gradient_descent_step(theta, lr)
    return theta

# Vectorize gradient descent over all samples
gradient_descent_vmap = jax.vmap(gradient_descent, in_axes=(0, None, None))

print("Running gradient descent on all samples...")
theta_opt = gradient_descent_vmap(theta_matrix, 0.05, 100)

# Vectorize mitral_output computation
@jax.jit
def compute_rgb_y(theta):
    g, b, r, y = mitral_output(theta, x, m, cortex)
    return g, b, r, y

compute_rgb_y_vmap = jax.vmap(compute_rgb_y, in_axes=(0,))

print("Computing rgb and y values...")
g_all, b_all, r_all, y_all = compute_rgb_y_vmap(theta_opt)

# Store rgb values as arrays for vectorized filtering
rgb_values = (g_all, b_all, r_all, y_all)

print(f"Initial number of parameters: {no_samples}")

# Filter 1: Remove parameters with negative signals (r, g, b, y) - vectorized
g_all, b_all, r_all, y_all = rgb_values
has_negative = jnp.any(g_all < 0, axis=1) | jnp.any(b_all < 0, axis=1) | jnp.any(r_all < 0, axis=1) | jnp.any(y_all < 0, axis=1)
valid_mask_neg = ~has_negative
valid_indices_after_neg = jnp.where(valid_mask_neg)[0]

if len(valid_indices_after_neg) > 0:
    theta_after_neg = theta_opt[valid_indices_after_neg]
else:
    theta_after_neg = jnp.zeros((0, no_params))
print(f"Parameters remaining after removing negative signals: {len(valid_indices_after_neg)}")

# Filter 2: Keep parameters where rgb values are close to 0.5 (0.35-0.65) - vectorized
g_filtered = g_all[valid_indices_after_neg]
b_filtered = b_all[valid_indices_after_neg]
r_filtered = r_all[valid_indices_after_neg]

g_in_range = jnp.any((g_filtered >= 0.4) & (g_filtered <= 0.6), axis=1)
b_in_range = jnp.any((b_filtered >= 0.4) & (b_filtered <= 0.6), axis=1)
r_in_range = jnp.any((r_filtered >= 0.4) & (r_filtered <= 0.6), axis=1)

valid_mask_rgb = g_in_range | b_in_range | r_in_range
valid_indices_after_rgb = valid_indices_after_neg[valid_mask_rgb]

if len(valid_indices_after_rgb) > 0:
    theta_after_rgb = theta_opt[valid_indices_after_rgb]
else:
    theta_after_rgb = jnp.zeros((0, no_params))
print(f"Parameters remaining after keeping rgb values in [0.4, 0.6]: {len(valid_indices_after_rgb)}")

# Filter 3: Keep parameters that have u values smaller than 0.000000001 - vectorized
@jax.jit
def compute_u(theta):
    g, u, s = from_y_next_step(theta, x, m, cortex)
    return u

compute_u_vmap = jax.vmap(compute_u, in_axes=(0,))

print("Computing u values...")
theta_for_u = theta_opt[valid_indices_after_rgb]
u_all = compute_u_vmap(theta_for_u)  # Shape: (n_samples, n_params, n_params) or similar

# Check if any u values are smaller than 0.000000001 (in absolute value)
# Flatten u for each parameter and check if any value is smaller than 0.0001
u_abs = jnp.abs(u_all)
# Reshape to flatten each u matrix: (n_samples, -1) then check any
has_small_u = jnp.any(u_abs.reshape(len(u_all), -1) < 0.000000001, axis=1)

# Keep parameters that have at least one u value smaller than 0.0001
valid_mask_u = has_small_u
valid_indices_final = valid_indices_after_rgb[valid_mask_u]

if len(valid_indices_final) > 0:
    theta_final = theta_opt[valid_indices_final]
else:
    theta_final = jnp.zeros((0, no_params))
print(f"Parameters remaining after keeping those with u values smaller than 0.000000001: {len(valid_indices_final)}")
##############################

#make a bar chart graph of set of parameters vs inital amount ie 10000, then Parameters remaining after removing negative signals: 84Parameters remaining after keeping rgb values in [0.35, 0.65]: 51 and then after removing participation ratios
#use matplotlib
plt.bar(['Initial', 'Only positive signals', 'r/g/b values in nonlinear range', 'Degenerate solutions'], [no_samples, len(valid_indices_after_neg), len(valid_indices_after_rgb), len(valid_indices_final)])
plt.ylabel('Set of parameters')
plt.show()