import jax
import jax.numpy as jnp
import numpy as np

import ep

jax.config.update("jax_enable_x64", True)


def _bounded_log_posterior(theta):
    """Log-posterior of Uniform([0,1]^d): 0 inside, -inf outside."""
    in_support = jnp.all((theta >= 0.0) & (theta <= 1.0))
    return jnp.where(in_support, 0.0, -jnp.inf)


def _make_skew_approx(dim=2):
    theta_star = jnp.full((dim,), 0.5)
    symmetric_dist = ep.FullGaussian(mu=theta_star, Sigma=jnp.eye(dim))
    return ep.SkewSymmetricApproximation(
        symmetric_dist=symmetric_dist,
        log_posterior=_bounded_log_posterior,
        theta_star=theta_star,
    )


def test_skewing_factor_returns_half_when_both_posteriors_zero():
    """Remark 3: w(θ) = 0.5 when both π(θ)L(θ) and π(2θ*-θ)L(2θ*-θ) are zero."""
    approx = _make_skew_approx(dim=2)
    # Point outside [0,1]^2; its reflection 2*0.5 - (-1, -1) = (2, 2) is also outside.
    outside = jnp.array([-1.0, -1.0])
    w = approx.skewing_factor(outside)
    assert float(w) == 0.5, f"Expected 0.5, got {w}"


def test_skewing_factor_no_nan_outside_support():
    approx = _make_skew_approx(dim=2)
    outside = jnp.array([-1.0, -1.0])
    w = approx.skewing_factor(outside)
    assert jnp.isfinite(w), f"skewing_factor produced non-finite value: {w}"


def test_skewing_factor_valid_inside_support():
    approx = _make_skew_approx(dim=2)
    inside = jnp.array([0.5, 0.5])
    w = approx.skewing_factor(inside)
    # symmetric point, so w should be 0.5
    assert jnp.isfinite(w)
    assert 0.0 <= float(w) <= 1.0


def test_sample_no_nan_with_bounded_support():
    """Sampler must not produce NaN even when Gaussian places mass outside bounded support."""
    approx = _make_skew_approx(dim=2)
    key = jax.random.PRNGKey(0)
    samples = approx.sample(key, n_samples=200)
    assert samples.shape == (200, 2)
    assert np.all(np.isfinite(samples)), "sample() returned NaN or Inf values"
