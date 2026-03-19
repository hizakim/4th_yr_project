"""
Tests for generate_theta_opt module.
Run with: pytest tests/test_generate_theta_opt.py -v
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_theta_opt import (
    ThetaOptConfig,
    build_loss_fn,
    filter_valid_theta_opt,
    generate_theta_opt,
    get_default_config,
    mitral_output,
    run_single_optimization,
)


# -----------------------------------------------------------------------------
# Test 1: Reproducibility
# -----------------------------------------------------------------------------


def test_reproducibility():
    """Same key and n_samples -> same theta_opt on two calls."""
    key = jax.random.PRNGKey(123)
    config = get_default_config()
    theta_opt_1 = generate_theta_opt(5, key, config=config, batch_size=5, opt_steps=50)
    key = jax.random.PRNGKey(123)
    theta_opt_2 = generate_theta_opt(5, key, config=config, batch_size=5, opt_steps=50)
    np.testing.assert_allclose(theta_opt_1, theta_opt_2, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Test 2: Batch-size invariance
# -----------------------------------------------------------------------------


def test_batch_size_invariance():
    """Same key and n_samples; different batch_size -> same theta_opt."""
    key = jax.random.PRNGKey(456)
    config = get_default_config()
    theta_a = generate_theta_opt(10, key, config=config, batch_size=10, opt_steps=50)
    key = jax.random.PRNGKey(456)
    theta_b = generate_theta_opt(10, key, config=config, batch_size=5, opt_steps=50)
    np.testing.assert_allclose(theta_a, theta_b, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Test 3: Shape and dtype
# -----------------------------------------------------------------------------


def test_output_shape_and_dtype():
    """theta_opt has shape (n_samples, no_params) and float dtype."""
    key = jax.random.PRNGKey(0)
    config = get_default_config()
    theta_opt = generate_theta_opt(3, key, config=config, batch_size=2, opt_steps=30)
    assert theta_opt.shape == (3, 12)
    assert theta_opt.dtype == jnp.float32 or theta_opt.dtype == jnp.float64


# -----------------------------------------------------------------------------
# Test 4: Single-sample vs notebook-style optimizer
# -----------------------------------------------------------------------------


def test_single_optimization_matches_notebook_style():
    """
    One run of run_single_optimization with a given init matches a Python-loop
    Adam (notebook-style) for the same init and step count.
    """
    import optax

    config = get_default_config()
    loss_fn = build_loss_fn(config)
    key = jax.random.PRNGKey(789)
    theta_init = jax.random.uniform(
        key, (config.no_params,), minval=0.0, maxval=1.0
    )
    steps = 80
    lr = 0.02
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    max_grad_norm = 3.0

    # Notebook-style: Python for-loop Adam with same hyperparams and clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=lr, b1=beta1, b2=beta2, eps=eps),
    )
    opt_state = optimizer.init(theta_init)
    grad_fn = jax.jit(jax.grad(loss_fn))
    theta = theta_init
    for _ in range(steps):
        grad = grad_fn(theta)
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        theta = jnp.clip(theta, 0.0, None)
        theta = theta.at[9:10].set(jnp.clip(theta[9:10], 0.0, 1.0))
    theta_notebook = theta

    # Our traced optimizer
    theta_script = run_single_optimization(
        theta_init,
        loss_fn,
        steps=steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        max_grad_norm=max_grad_norm,
    )

    np.testing.assert_allclose(
        theta_notebook,
        theta_script,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Traced Adam should match notebook-style Adam for same init and steps",
    )


# -----------------------------------------------------------------------------
# Test 5: Valid filter
# -----------------------------------------------------------------------------


def test_filter_valid_theta_opt():
    """filter_valid_theta_opt returns correct count and shape."""
    config = get_default_config()
    # Build two thetas: one that yields outputs in [0.4, 0.6], one that doesn't.
    # Use a known good region: targets are 0.5, so theta that gives ~0.5 is valid.
    key = jax.random.PRNGKey(0)
    # First row: random, might be invalid
    theta1 = jax.random.uniform(key, (config.no_params,), minval=0.0, maxval=1.0)
    key, sub = jax.random.split(key)
    theta2 = jax.random.uniform(sub, (config.no_params,), minval=0.0, maxval=1.0)
    theta_opt = jnp.stack([theta1, theta2])

    theta_valid, valid_mask = filter_valid_theta_opt(theta_opt, config, low=0.4, high=0.6)
    assert valid_mask.shape == (2,)
    assert theta_valid.shape[0] == int(jnp.sum(valid_mask))
    assert theta_valid.shape[1] == config.no_params
    # Each kept row should actually be valid when recomputed
    for i in range(theta_valid.shape[0]):
        r, g, b, y = mitral_output(
            theta_valid[i],
            config.x,
            config.m,
            n_r=config.n_r,
            n_b=config.n_b,
            n_g=config.n_g,
            n_y=config.n_y,
        )
        assert jnp.all((r >= 0.4) & (r <= 0.6))
        assert jnp.all((g >= 0.4) & (g <= 0.6))
        assert jnp.all((b >= 0.4) & (b <= 0.6))
        assert jnp.all((y >= 0.4) & (y <= 0.6))
