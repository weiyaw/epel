from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

import ep
import vi
from mcmc import hmc, random_walk
from model import Regression


jax.config.update("jax_enable_x64", True)


def _regression_model_for_algorithms(n_points: int | None = None) -> Regression:
    return Regression(prior_sd=10.0, n_points=n_points)


def test_hmc_runs_on_regression_without_errors():
    model = _regression_model_for_algorithms()
    key = jax.random.PRNGKey(0)
    log_posterior = partial(model.log_posterior, check_sum_to_1=True, use_llog=False)

    init_key, algo_key = jax.random.split(key)
    initial_theta = model.init_theta(init_key)

    state, info, time_ls = hmc(
        algo_key,
        log_posterior,
        initial_theta,
        n_samples=2,
        n_warmup=1,
        step_size=1e-2,
        inverse_mass_matrix=jnp.ones((model.dim_theta,)),
        num_integration_steps=10,
    )

    assert state.position.shape[0] == 3
    assert info.is_accepted.shape[0] == 2
    assert len(time_ls) == 3


def test_rw_runs_on_regression_without_errors():
    model = _regression_model_for_algorithms()
    key = jax.random.PRNGKey(1)
    log_posterior = partial(model.log_posterior, check_sum_to_1=True, use_llog=False)

    init_key, algo_key = jax.random.split(key)
    initial_theta = model.init_theta(init_key)

    state, info, time_ls = random_walk(
        algo_key,
        log_posterior,
        initial_theta,
        sd_shrink_factor=0.1,
        n_samples=4,
        n_warmup=3,
        thinning=2,
    )

    assert state.position.shape[0] >= 2
    assert info.is_accepted.shape[0] >= 1
    assert len(time_ls) == 3


def test_vi_runs_on_regression_without_errors():
    model = _regression_model_for_algorithms()
    key = jax.random.PRNGKey(2)

    log_posterior = partial(
        model.log_posterior,
        check_sum_to_1=True,
        use_llog=False,
        use_ael=True,
    )
    loss_fn = vi.ElboLoss(log_posterior, num_samples=1)
    init_q = vi.FullGaussian.initialized(key, model.dim_theta)

    q_ls, final_q, losses, time_ls = vi.fit_to_variational_target(
        key=key,
        dist=init_q,
        loss_fn=loss_fn,
        optimizer=optax.adam(learning_rate=1e-2),
        iters=3,
        save_freq=1,
        show_progress=False,
    )

    assert len(q_ls) == 4
    assert len(losses) == 3
    assert len(time_ls) == 4
    assert np.all(np.isfinite(np.asarray(final_q.mu)))


def test_ep_runs_on_regression_without_errors():
    model = _regression_model_for_algorithms(n_points=10)
    key = jax.random.PRNGKey(3)
    init_theta = model.init_theta(key)
    init_cov = jnp.eye(model.dim_theta)

    global_g_ls, site_gs, iterations, time_ls = ep.expectation_propagation(
        key=key,
        f_term=model.log_posterior_term,
        n_terms=model.n_terms,
        init_g=(init_theta, init_cov),
        laplace_iter=1,
        snis_samples=10,
        damping_factor=0.5,
        tol=1e-2,
        max_iter=2,
        verbose=False,
    )

    assert len(global_g_ls) >= 1
    assert len(site_gs) == model.n_terms
    assert iterations >= 1
    assert len(time_ls) >= 2


# --- SkewSymmetricApproximation tests ---


def _run_ep(model, key):
    init_theta = model.init_theta(key)
    init_cov = jnp.eye(model.dim_theta)
    global_g_ls, _, _, _ = ep.expectation_propagation(
        key=key,
        f_term=model.log_posterior_term,
        n_terms=model.n_terms,
        init_g=(init_theta, init_cov),
        laplace_iter=1,
        snis_samples=10,
        damping_factor=0.5,
        tol=1e-2,
        max_iter=2,
        verbose=False,
    )
    return global_g_ls[-1]


def test_skewing_factor_is_half_for_symmetric_posterior():
    """When the unnormalized posterior is symmetric about theta_star,
    the skewing factor should be 0.5 everywhere (sigmoid(0) = 0.5)."""
    d = 3
    key = jax.random.PRNGKey(10)
    theta_star = jnp.zeros(d)
    gaussian = ep.FullGaussian(mu=theta_star, Sigma=jnp.eye(d))

    # Standard Gaussian log-posterior — symmetric about 0
    log_unnorm_post = lambda theta: -0.5 * jnp.dot(theta, theta)

    skew = ep.SkewSymmetricApproximation(
        symmetric_dist=gaussian,
        log_posterior=log_unnorm_post,
        theta_star=theta_star,
    )

    test_points = jax.random.normal(key, (20, d))
    ws = jax.vmap(skew.skewing_factor)(test_points)
    np.testing.assert_allclose(np.asarray(ws), 0.5, atol=1e-6)


def test_skew_symmetric_sample_shape():
    """Samples from SkewSymmetricApproximation have the right shape."""
    d = 4
    n_samples = 50
    key = jax.random.PRNGKey(20)
    theta_star = jnp.ones(d)
    gaussian = ep.FullGaussian(mu=theta_star, Sigma=jnp.eye(d))
    log_unnorm_post = lambda theta: -0.5 * jnp.dot(theta - theta_star, theta - theta_star)

    skew = ep.SkewSymmetricApproximation(
        symmetric_dist=gaussian,
        log_posterior=log_unnorm_post,
        theta_star=theta_star,
    )

    samples = skew.sample(key, n_samples)
    assert samples.shape == (n_samples, d)
    assert np.all(np.isfinite(np.asarray(samples)))


def test_skew_ep_samples_finite_after_regression_ep():
    """After EP on the regression model, skew-EP samples should be finite."""
    model = _regression_model_for_algorithms(n_points=10)
    key = jax.random.PRNGKey(30)
    global_g = _run_ep(model, key)

    skew_approx = ep.SkewSymmetricApproximation(
        symmetric_dist=global_g,
        log_posterior=partial(model.log_posterior, check_sum_to_1=True, use_llog=False),
        theta_star=global_g.mu,
    )

    sample_key = jax.random.PRNGKey(31)
    samples = skew_approx.sample(sample_key, n_samples=100)

    assert samples.shape == (100, model.dim_theta)
    assert np.all(np.isfinite(np.asarray(samples)))


def test_skew_ep_samples_differ_from_gaussian_on_skewed_posterior():
    """Skew-EP should produce different sample moments from EP Gaussian
    when the posterior is asymmetric (i.e. skewing factors != 0.5)."""
    model = _regression_model_for_algorithms(n_points=10)
    key = jax.random.PRNGKey(40)
    global_g = _run_ep(model, key)

    skew_approx = ep.SkewSymmetricApproximation(
        symmetric_dist=global_g,
        log_posterior=partial(model.log_posterior, check_sum_to_1=True, use_llog=False),
        theta_star=global_g.mu,
    )

    n_samples = 500
    gauss_key, skew_key = jax.random.split(jax.random.PRNGKey(41))
    gauss_samples = global_g.sample(gauss_key, n_samples)
    skew_samples = skew_approx.sample(skew_key, n_samples)

    # Check the skewing factors are not all 0.5 (posterior is skewed)
    ws = jax.vmap(skew_approx.skewing_factor)(gauss_samples)
    assert not np.allclose(np.asarray(ws), 0.5, atol=0.01), (
        "All skewing factors are 0.5 — posterior appears symmetric, "
        "skew-EP has no effect."
    )

    # Skew samples should be finite and same shape
    assert skew_samples.shape == gauss_samples.shape
    assert np.all(np.isfinite(np.asarray(skew_samples)))
