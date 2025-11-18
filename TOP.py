import jax
import jax.numpy as jnp
from jaxopt import AndersonAcceleration


# Naka-Rushton function
x_g =[]
def nr_g(x, n=4, k=1):
    x_g.append(x)
    return x**n / (x**n + k**n)

x_b =[]
def nr_b(x, n=4, k=1):
    x_b.append(x)
    return x**n / (x**n + k**n)

x_r =[]
def nr_r(x, n=4, k=1):
    x_r.append(x)
    return x**n / (x**n + k**n)

x_y =[]
def nr_y(x, n=4, k=0.5):
    x_y.append(x)
    return x**n / (x**n + k**n)

# fixed-point function
def mitral_fixed_point(g, x, alpha, Wgr, Wrg, wb):
    g_new = nr_g(alpha * x - wb * nr_b(alpha * x) + Wrg @ nr_r(alpha * x - Wgr @ g - wb * nr_b(alpha * x)))
    return g_new

def y_fixed_point(y, base_y, wcor, wy, wgranule, cortex):
    gr = nr_y(wcor * cortex + wy * y)
    y = base_y - wgranule * gr
    return y

# unpack parameter
def unpack_theta(theta, m):
    idx = 0
    wr = theta[idx:idx+m]; idx += m
    wb = theta[idx:idx+m]; idx += m
    Wgr = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    Wrg = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    Wgm = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    alpha = theta[idx:idx+m]; idx += m
    wcor = theta[idx:idx+m]; idx += m
    wy = theta[idx:idx+m]; idx += m
    wgranule = theta[idx:idx+m]; idx += m
    return wr, wb, Wgr, Wrg, Wgm, alpha, wcor, wy, wgranule


# solve fixed point via Anderson - return g
def solve_fixed_point(theta, x, m):
    _, wb, Wgr, Wrg, _, alpha, _, _, _ = unpack_theta(theta, m)
    g0 = jnp.zeros(m)
    aa = AndersonAcceleration(mitral_fixed_point, history_size=5, tol=1e-10, maxiter=50)
    solution = aa.run(g0, x, alpha, Wgr, Wrg, wb)
    g_fixed = solution.params
    return g_fixed

# find mitral output using solved g
def mitral_output(theta, x, m, cortex):
    wr, wb, Wgr, _, Wgm, alpha, wcor, wy, wgranule = unpack_theta(theta, m)

    g = solve_fixed_point(theta, x, m)
    b = nr_b(alpha * x)
    r = nr_r(alpha * x - Wgr @ g - wb * b)
    base_y = nr_y(alpha * x + wr * r - wb * b - Wgm @ g)

    #===================
    y0 = base_y
    aa_y = AndersonAcceleration(y_fixed_point, history_size=5, tol=1e-10, maxiter=200)
    solution = aa_y.run(y0, base_y, wcor, wy, wgranule, cortex)
    y_fixed = solution.params
    #===================
    return y_fixed


def from_y_next_step(theta, x, m, cortex):
    def loss(theta):
        y = mitral_output(theta, x, m, cortex)
        return jnp.sum(y)

    grad_loss = jax.grad(loss)
    hessian = jax.jit(jax.jacrev(grad_loss))

    g = grad_loss(theta)
    H = hessian(theta)
    H = jnp.squeeze(H)
    u, s, vt = jnp.linalg.svd(H)
    #delta = u[:, -1]
    return g, u, s

def possible_deltas(g,u,s):
    deltas = []
    indices_e7 = jnp.where(jnp.abs(s) < 1e-7)[0]
    for i in indices_e7:
        ub = u[:,i]
        A =jnp.stack([g,ub],axis=1)
        Q, R = jnp.linalg.qr(A)

        #vector orthogonal to g and ub
        q_last = Q[:,-1]
        deltas.append(q_last)

    return(deltas)
