import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jaxopt import AndersonAcceleration
import jax.numpy as jnp


def mitral_fixed_point(s, x, alpha, Wgr, Wrg, wb):
    """
    s: state vector [r, b, g]
    x: input
    """
    m = x.shape[0]
    r, b, g = s[:m], s[m:2*m], s[2*m:]

    r_new = sigmoid(alpha * x - Wgr @ g - wb * b)
    b_new = sigmoid(alpha * x)
    g_new = sigmoid(alpha * x - wb * b_new + Wrg @ r)

    return jnp.concatenate([r_new, b_new, g_new])


m = 2
x = jnp.array([1.0, 1.0])
alpha = jnp.ones(m) * 0.8
Wgr = jnp.zeros((m,m))
Wrg = jnp.zeros((m,m))
wb = jnp.ones(m) * 0.5
s = jnp.zeros(3*m)
#print(mitral_fixed_point(s, x, alpha, Wgr, Wrg, wb))



#################
def T(s, a):
    return 0.5*s + a
aa = AndersonAcceleration(T, history_size=3)
s0 = jnp.array(0.0)
sol = aa.run(s0, 1.0).params
print(sol)

#############
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)
print(jax.random.normal(k1))  # one sample
print(jax.random.normal(k2))  # another independent sample