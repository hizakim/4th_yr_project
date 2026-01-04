import jax
import jax.numpy as jnp
from jaxopt import AndersonAcceleration


# Naka-Rushton function
def nr(x, n=4):
    return x**n / (x**n + 1)


# fixed-point function
def mitral_fixed_point(g, x, alpha, Wgr, Wrg, wb):
    g_new = nr(alpha * x - wb * nr(alpha * x) + Wrg @ nr(alpha * x - Wgr @ g - wb * nr(alpha * x)))
    return g_new

def y_fixed_point(y, base_y, wcor, wy, wgranule, cortex):
    gr = nr(wcor * cortex + wy * y)
    y = base_y - wgranule * gr
    return y







# unpack parameter
def unpack_theta(theta, m):
    idx = 0
    w_osn_r = theta[idx:idx+m]; idx += m
    w_osn_b = theta[idx:idx+m]; idx += m
    w_osn_g = theta[idx:idx+m]; idx += m
    w_osn_y = theta[idx:idx+m]; idx += m
    w_b_r = theta[idx:idx+m]; idx += m
    w_b_g = theta[idx:idx+m]; idx += m
    w_b_y = theta[idx:idx+m]; idx += m
    w_r_y = theta[idx:idx+m]; idx += m
    W_g_r = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    W_r_g = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    W_g_m = theta[idx:idx+m*m].reshape((m, m)); idx += m*m
    alpha = theta[idx:idx+m]; idx += m
    w_cor = theta[idx:idx+m]; idx += m
    w_y = theta[idx:idx+m]; idx += m
    w_granule = theta[idx:idx+m]; idx += m
    return w_osn_r, w_osn_b, w_osn_g, w_osn_y, w_b_r, w_b_g, w_b_y, w_r_y, W_g_r, W_r_g, W_g_m, alpha, w_cor, w_y, w_granule


# solve fixed point via Anderson - return g
def solve_fixed_point(theta, x, m):
    _, wb, _, _, _, _, _, _, Wgr, Wrg, _, alpha, _, _, _ = unpack_theta(theta, m)
    g0 = jnp.zeros(m)
    aa = AndersonAcceleration(mitral_fixed_point, history_size=5, tol=1e-10, maxiter=50)
    solution = aa.run(g0, x, alpha, Wgr, Wrg, wb)
    g_fixed = solution.params
    return g_fixed

# find mitral output using solved g
def mitral_output(theta, x, m, cortex):
    wr, wb, _, _, _, _, _, _, Wgr, _, Wgm, alpha, wcor, wy, wgranule = unpack_theta(theta, m)

    g = solve_fixed_point(theta, x, m)
    b = nr(alpha * x)
    r = nr(alpha * x - Wgr @ g - wb * b)
    base_y = nr(alpha * x + wr * r - wb * b - Wgm @ g)

    #===================
    y0 = base_y
    aa_y = AndersonAcceleration(y_fixed_point, history_size=5, tol=1e-10, maxiter=200)
    solution = aa_y.run(y0, base_y, wcor, wy, wgranule, cortex)
    y_fixed = solution.params
    #===================
    return g,b,r,y_fixed


def from_y_next_step(theta, x, m, cortex):
    def loss(theta):
        _, _, _, y = mitral_output(theta, x, m, cortex)
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
