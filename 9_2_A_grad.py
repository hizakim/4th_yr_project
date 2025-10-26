'''
lets consider only first order term
pick random (w1,w2, z1,z2)
find grad, then find vector orthogonal -> this is step direction such that Delta_Y = 0
'''
import jax
import jax.numpy as jnp

def y(theta):
    n = theta.size // 2
    w, z = theta[:n], theta[n:]
    y = jnp.sum(w * jnp.tanh(z))
    return y

n=2
key = jax.random.PRNGKey(20)
theta = jax.random.uniform(key, (2*n,), minval=0.0, maxval=1.0)
y0 = y(theta)
print('y0 =',y0)
print('theta =',theta)

grad_y = jax.grad(y) # grad of y wrt theta
g = grad_y(theta)
g_norm = jnp.dot(g, g)

if g_norm < 1e-10:
    print('zero gradient - any small step keeps y unchanged')
    exit()

# find vector orthogonal to g
v = jax.random.normal(key, (2*n,))
orth_dir = v - jnp.dot(v, g) / jnp.dot(g, g) * g
orth_dir = orth_dir / jnp.linalg.norm(orth_dir)  # normalise
print('orth_dir =',orth_dir)


##################
#CHECK - first order vs exact change

# take a small step
alpha = 0.1
theta_new = theta + alpha * orth_dir

# check deltay
dy_exact = y(theta_new) - y(theta)
dy_linear = jnp.dot(g, alpha * orth_dir)  # should be approx0
print(f"Δy (linear approx) = {dy_linear:.6f}, Δy (actual) = {dy_exact:.6f}")