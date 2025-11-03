import numpy as np

'''
def T(x, theta):
    return 0.5 * x + theta   

x = 0.0 # initial guess
theta = 1.0

for i in range(10):
    x = T(x, theta)
    print(f"Iteration {i}: x = {x}")
'''

from jaxopt import AndersonAcceleration
import jax.numpy as jnp

def T(x, theta):
    return 0.5 * x + theta

aa = AndersonAcceleration(fixed_point_fun=T, history_size=5, tol=1e-6)

x_init = jnp.array(10.0)
theta = jnp.array(1.0)

result = aa.run(x_init, theta)
print(result.params)   # should be close to 2