import jax
import jax.numpy as jnp

def y(theta):
    w, z = theta[:n], theta[n:]
    y = jnp.sum(w * jnp.tanh(z))
    return y

n=5
# Random initialisation using JAX
key = jax.random.PRNGKey(42)
theta = jax.random.uniform(key, (2*n,), minval=0.0, maxval=1.0)

y0 = y(theta)
print('y0 =',y0)

# Gradient JAX
# grad of y wrt theta
grad_y = jax.grad(y)
g = grad_y(theta)

print('Gradient g =', g)

#note:
#g[:n] corresponds to dy/dw (first chunk)
#g[n:] corresponds to dy/dz

#================


#Hessian of y wrt theta
H = jax.hessian(y)(theta)
print('Hessian H =', H)

#eigen values and vectors of H
e_vals, e_vectors = jnp.linalg.eigh(H)
print('Eigenvalues =', e_vals)
