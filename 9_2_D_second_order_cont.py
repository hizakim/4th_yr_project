import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.scipy.optimize import minimize

'''
try and solve delta using optimisation
'''
# --------------------------
# Function y(theta)
n = 2
def y_func(theta):
    w, z = theta[:n], theta[n:]
    return jnp.sum(w * jnp.tanh(z))

# Random theta
key = jax.random.PRNGKey(20)
theta = jax.random.uniform(key, (2*n,), minval=0.0, maxval=1.0)

# Gradient and Hessian
grad_y = jax.grad(y_func)
hess_y = jax.hessian(y_func)
g = grad_y(theta)
H = hess_y(theta)


def loss(delta):
    Delta_y = g @ delta + 0.5 * delta @ H @ delta
    return Delta_y**2

# Compute gradient of loss wrt delta
# grad_loss = jax.jit(jax.grad(loss))

delta_init = jnp.full((2*n,), 0.1)  # initial guess = 0.5
res = minimize(loss, delta_init, method='BFGS', options={'maxiter': 100})
delta_solution = res.x

# Check result
delta_y_val = g @ delta_solution + 0.5 * delta_solution @ H @ delta_solution
print("Delta solving Δy=0 via BFGS:", delta_solution)
print("Δy at solution:", delta_y_val)
