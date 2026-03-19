"""
Reusable batched generator for converged theta_opt with random initializations.
Extracted from TMR_after_26_2_22_symmetric_weights.ipynb; runs optimizations
in parallel via JAX vmap over batches of initial thetas.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxopt import AndersonAcceleration


# -----------------------------------------------------------------------------
# Model (symmetric weights, from notebook cells 1, 3, 4)
# -----------------------------------------------------------------------------


def nr(x, n):
    """Naka-Rushton activation function."""
    x = jnp.maximum(x, 0.0)
    return jnp.power(x, n) / (jnp.power(x, n) + 1)


def unpack_theta(theta, m):
    """Unpack parameter vector into symmetric weight arrays (notebook cell 3)."""
    w_osn_r = jnp.array([theta[0], theta[0]])
    w_osn_b = jnp.array([theta[1], theta[1]])
    w_osn_g = jnp.array([theta[2], theta[2]])
    w_osn_y = jnp.array([theta[3], theta[3]])
    w_b_r = jnp.array([theta[4], theta[4]])
    w_b_g = jnp.array([theta[5], theta[5]])
    w_b_y = jnp.array([theta[6], theta[6]])
    w_r_y = jnp.array([theta[7], theta[7]])
    w_r_g = jnp.array([theta[8], theta[8]])
    alpha = jnp.array([theta[9], theta[9]])
    W_g_r = jnp.array([[0.0, theta[10]], [theta[10], 0.0]])
    W_g_y = jnp.array([[0.0, theta[11]], [theta[11], 0.0]])
    return w_osn_r, w_osn_b, w_osn_g, w_osn_y, w_b_r, w_b_g, w_b_y, w_r_y, w_r_g, alpha, W_g_r, W_g_y


def mitral_fixed_point(g, x, w_osn_r, w_osn_b, w_osn_g, w_b_r, w_b_g, W_g_r, w_r_g, alpha, n_r, n_b, n_g):
    """Fixed point equation for green neurons with different n values."""
    b_input = w_osn_b * alpha * x
    b_out = nr(b_input, n=n_b)
    r_input = w_osn_r * alpha * x - w_b_r * b_out - W_g_r @ g
    r_out = nr(r_input, n=n_r)
    g_input = w_osn_g * alpha * x - w_b_g * b_out + w_r_g * r_out
    g_new = nr(g_input, n=n_g)
    return g_new


def solve_fixed_point(theta, x, m, n_r, n_b, n_g):
    """Solve fixed point via Anderson Acceleration."""
    w_osn_r, w_osn_b, w_osn_g, w_osn_y, w_b_r, w_b_g, w_b_y, w_r_y, w_r_g, alpha, W_g_r, W_g_y = unpack_theta(
        theta, m
    )
    g0 = jnp.zeros(m)

    def fixed_point_with_n(g, x, w_osn_r, w_osn_b, w_osn_g, w_b_r, w_b_g, W_g_r, w_r_g, alpha):
        return mitral_fixed_point(
            g, x, w_osn_r, w_osn_b, w_osn_g, w_b_r, w_b_g, W_g_r, w_r_g, alpha, n_r, n_b, n_g
        )

    aa = AndersonAcceleration(fixed_point_with_n, history_size=5, tol=1e-8, maxiter=100)
    solution = aa.run(g0, x, w_osn_r, w_osn_b, w_osn_g, w_b_r, w_b_g, W_g_r, w_r_g, alpha)
    return solution.params


def mitral_output(theta, x, m, n_r, n_b, n_g, n_y):
    """Compute mitral output (r, g, b, y) for given theta and config."""
    w_osn_r, w_osn_b, w_osn_g, w_osn_y, w_b_r, w_b_g, w_b_y, w_r_y, w_r_g, alpha, W_g_r, W_g_y = unpack_theta(
        theta, m
    )
    g = solve_fixed_point(theta, x, m, n_r, n_b, n_g)
    b = nr(w_osn_b * alpha * x, n=n_b)
    r = nr(w_osn_r * alpha * x - w_b_r * b - W_g_r @ g, n=n_r)
    y = nr(w_osn_y * alpha * x - w_b_y * b - W_g_y @ g + w_r_y * r, n=n_y)
    return r, g, b, y


# -----------------------------------------------------------------------------
# Config and loss
# -----------------------------------------------------------------------------


@dataclass
class ThetaOptConfig:
    """Problem configuration for theta optimization (matches notebook cell 7)."""

    x: jnp.ndarray
    m: int
    n_r: float
    n_b: float
    n_g: float
    n_y: float
    r_target: jnp.ndarray
    g_target: jnp.ndarray
    b_target: jnp.ndarray
    y_target: jnp.ndarray
    no_params: int


def get_default_config() -> ThetaOptConfig:
    """Default config matching notebook cell 7."""
    m = 2
    return ThetaOptConfig(
        x=jnp.ones(m),
        m=m,
        n_r=2.333,
        n_b=0.9,
        n_g=2.0626,
        n_y=1.562,
        r_target=jnp.array([0.55, 0.55]),
        g_target=jnp.array([0.55, 0.55]),
        b_target=jnp.array([0.55, 0.55]),
        y_target=jnp.array([0.55, 0.55]),
        no_params=12,
    )


def build_loss_fn(config: ThetaOptConfig):
    """Build loss_fn(theta) closing over config (notebook cell 8)."""

    def loss_fn(theta):
        r, g, b, y = mitral_output(
            theta,
            config.x,
            config.m,
            n_r=config.n_r,
            n_b=config.n_b,
            n_g=config.n_g,
            n_y=config.n_y,
        )
        return (
            jnp.sum((r - config.r_target) ** 2)
            + jnp.sum((g - config.g_target) ** 2)
            + jnp.sum((b - config.b_target) ** 2)
            + jnp.sum((y - config.y_target) ** 2)
        )

    return loss_fn


# -----------------------------------------------------------------------------
# Traced Adam (vmappable via fori_loop)
# -----------------------------------------------------------------------------

ALPHA_START_IDX = 9
ALPHA_END_IDX = 10


def _adam_step(
    carry,
    optimizer,
    grad_fn,
):
    """Single Adam step: (theta, opt_state) -> (theta_new, opt_state_new) with clipping."""
    theta, opt_state = carry
    grad = grad_fn(theta)
    updates, opt_state_new = optimizer.update(grad, opt_state, theta)
    theta_new = optax.apply_updates(theta, updates)
    theta_new = jnp.clip(theta_new, 0.0, None)
    theta_new = theta_new.at[ALPHA_START_IDX:ALPHA_END_IDX].set(
        jnp.clip(theta_new[ALPHA_START_IDX:ALPHA_END_IDX], 0.0, 1.0)
    )
    return (theta_new, opt_state_new)


def run_single_optimization(
    theta_init: jnp.ndarray,
    loss_fn,
    steps: int = 1500,
    lr: float = 0.02,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_grad_norm: float = 3.0,
) -> jnp.ndarray:
    """
    Run Adam optimization for a single initial theta; returns final theta.
    Pure and vmappable (no Python loop over steps).
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=lr, b1=beta1, b2=beta2, eps=eps),
    )
    opt_state = optimizer.init(theta_init)
    grad_fn = jax.grad(loss_fn)

    def body_fn(step, carry):
        return _adam_step(carry, optimizer, grad_fn)

    theta_final, _ = jax.lax.fori_loop(0, steps, body_fn, (theta_init, opt_state))
    return theta_final


# -----------------------------------------------------------------------------
# Batched generation API
# -----------------------------------------------------------------------------


def generate_theta_opt(
    n_samples: int,
    key: jnp.ndarray,
    config: Optional[ThetaOptConfig] = None,
    batch_size: int = 50,
    opt_steps: int = 1500,
    lr: float = 0.02,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_grad_norm: float = 3.0,
) -> jnp.ndarray:
    """
    Generate converged theta_opt for n_samples random initializations, in batches.

    Args:
        n_samples: Number of random initializations to optimize.
        key: JAX PRNG key for sampling initial thetas.
        config: Problem config; if None, uses get_default_config().
        batch_size: Number of optimizations to run in parallel per batch.
        opt_steps: Adam steps per run.
        lr, beta1, beta2, eps, max_grad_norm: Adam and clipping hyperparameters.

    Returns:
        theta_opt: shape (n_samples, no_params), float.
    """
    if config is None:
        config = get_default_config()
    loss_fn = build_loss_fn(config)
    no_params = config.no_params

    key, subkey = jax.random.split(key)
    theta_inits = jax.random.uniform(subkey, (n_samples, no_params), minval=0.0, maxval=1.0)

    # JIT-compile the vmapped optimizer for this batch size (and fixed loss_fn)
    def run_batch(theta_init_batch):
        return jax.vmap(
            lambda th: run_single_optimization(
                th,
                loss_fn,
                steps=opt_steps,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                max_grad_norm=max_grad_norm,
            )
        )(theta_init_batch)

    run_batch_jit = jax.jit(run_batch)
    results = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = theta_inits[start:end]
        theta_batch = run_batch_jit(batch)
        results.append(theta_batch)
    theta_opt = jnp.concatenate(results, axis=0)
    return theta_opt


# -----------------------------------------------------------------------------
# Optional helpers
# -----------------------------------------------------------------------------


def filter_valid_theta_opt(
    theta_opt: jnp.ndarray,
    config: ThetaOptConfig,
    low: float = 0.4,
    high: float = 0.6,
):
    """
    Keep only rows of theta_opt for which all mitral outputs (r,g,b,y) lie in [low, high].

    Returns:
        theta_valid: filtered array; valid_mask: boolean array of shape (n_samples,).
    """
    n_samples = theta_opt.shape[0]

    def outputs_in_range(theta):
        r, g, b, y = mitral_output(
            theta,
            config.x,
            config.m,
            n_r=config.n_r,
            n_b=config.n_b,
            n_g=config.n_g,
            n_y=config.n_y,
        )
        r_ok = jnp.all((r >= low) & (r <= high))
        g_ok = jnp.all((g >= low) & (g <= high))
        b_ok = jnp.all((b >= low) & (b <= high))
        y_ok = jnp.all((y >= low) & (y <= high))
        return r_ok & g_ok & b_ok & y_ok

    valid_mask = jax.vmap(outputs_in_range)(theta_opt)
    theta_valid = theta_opt[valid_mask]
    return theta_valid, valid_mask


def save_theta_opt(theta_opt: jnp.ndarray, path: str) -> None:
    """Save theta_opt to path (pickle)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(theta_opt, f)


def load_theta_opt(path: str) -> jnp.ndarray:
    """Load theta_opt from path (pickle)."""
    with open(path, "rb") as f:
        return pickle.load(f)
