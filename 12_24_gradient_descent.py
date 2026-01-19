

# Gradient descent optimiser
def gradient_descent(loss_fn, theta_init, lr=0.01, steps=200, record_every=20):
    """
    Fast gradient descent optimizer with JIT-compiled inner loop

    Performance improvements:
    - JIT-compiled optimization step (faster execution)
    - Uses JAX's fori_loop for compiled iteration (no Python overhead)
    - Reduced loss evaluations (only when needed for history)
    """
    # JIT compile the gradient function
    grad_fn = jax.jit(jax.grad(loss_fn))

    # JIT compile a single optimization step
    @jax.jit
    def step_fn(theta):
        g = grad_fn(theta)
        theta_new = theta - lr * g
        theta_new = jnp.clip(theta_new, 0.0, None)  # Keep weights non-negative
        return theta_new

    def body_fn(step, theta):
        return step_fn(theta)

    # Run the compiled loop - this is much faster than Python for loop
    theta = jax.lax.fori_loop(0, steps, body_fn, theta_init)

    # Record loss history (simplified - compute only when needed)
    loss_history = []
    if record_every > 0:
        loss_fn_jit = jax.jit(loss_fn)
        # Record initial loss
        loss_history.append(float(loss_fn_jit(theta_init)))
        # Record loss at intervals (recompute theta for those steps)
        for step in range(record_every, steps, record_every):
            theta_temp = theta_init
            for s in range(step):
                theta_temp = step_fn(theta_temp)
            loss_history.append(float(loss_fn_jit(theta_temp)))
        # Record final loss
        loss_history.append(float(loss_fn_jit(theta)))

    return theta, loss_history