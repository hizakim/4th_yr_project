import jax
import jax.numpy as jnp
import numpy as np


def y(theta):
    n = theta.size // 2
    w, z = theta[:n], theta[n:]
    y = jnp.sum(w * jnp.tanh(z))
    return y


#n = 2
#key = jax.random.PRNGKey(42)
#theta = jax.random.uniform(key, (2*n,), minval=0.0, maxval=1.0)
theta = jnp.array([0.4,0.6,0.8,1.0])

y0 = y(theta)
print('y0 =',y0)
print(theta)

grad_y = jax.grad(y)
g = grad_y(theta) # shape is (4,)
print('Gradient g =', g)

H_loss = 2 * jnp.outer(g,g) # (2n,2n) matrix
#print(H_loss)

e_vals, e_vectors = jnp.linalg.eigh(H_loss)
print('Eigenvalues =', round(e_vals,3)) # round to 3 decimal places

for i in range(len(e_vals)):
    if round(e_vals,3)[i] == 0:
        new_theta = theta + e_vectors[i]*0.1
        print(y(new_theta))