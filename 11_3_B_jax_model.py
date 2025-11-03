from jaxopt import AndersonAcceleration
import jax.numpy as jnp
from jax.nn import sigmoid
import numpy as np

# fixed-point function (returns a single vector s = [r, b, g]) 
def mitral_fixed_point(s, x, alpha, Wgr, Wrg):
    m = x.shape[0]
    r, b, g = s[:m], s[m:2*m], s[2*m:]

    r_new = sigmoid(alpha * x - Wgr @ g)        
    b_new = sigmoid(alpha * x)                    
    g_new = sigmoid(alpha * x - b_new + Wrg @ r)  
    
    return jnp.concatenate([r_new, b_new, g_new])

# parameters and matrices 
m = 3 # number of mitral cells
x = jnp.array([1.0, 1.0, 1.0]) # input vector
alpha = jnp.array([0.8, 0.8, 0.8])
wr = jnp.array([0.8, 0.8, 0.8])
wb = jnp.array([0.8, 0.8, 0.8])

Wgr = jnp.array([[0, 0.8, 0.8],
                 [0.8, 0, 0.8],
                 [0.8, 0.8, 0]])

Wrg = jnp.array([[0, 0.8, 0.8],
                 [0.8, 0, 0.8],
                 [0.8, 0.8, 0]])

Wgm = jnp.array([[0, 0.8, 0.8],
                 [0.8, 0, 0.8],
                 [0.8, 0.8, 0]])

# create Anderson solver
aa = AndersonAcceleration(fixed_point_fun=mitral_fixed_point,
                          history_size=5, tol=1e-6)

#s0 = jnp.zeros(3 * m)
#s0 = jnp.ones(3 * m)
s0 = np.random.rand(3 * m)


# fixed point solver 
solution = aa.run(s0, x, alpha, Wgr, Wrg).params
r, b, g = solution[:m], solution[m:2*m], solution[2*m:]

print("r =", r)
print("b =", b)
print("g =", g)

# final mitral cell output 
y = sigmoid(alpha * x + wr * r - wb * b - Wgm @ g)
print("Mitral Output:", y)
