import jax
import numpy as np
import pytest

from model import (
    Breastfeed,
    Cushings,
    CushingsLogistic,
    GEE,
    Job,
    Kyphosis,
    Orings,
    QuantRegression,
    QuantRegression3,
    Regression,
    Regression10,
)


jax.config.update("jax_enable_x64", True)


MODEL_FACTORIES = [
    ("regression", lambda: Regression(prior_sd=10.0), True),
    ("regression10", lambda: Regression10(prior_sd=10.0), True),
    (
        "quantregression",
        lambda: QuantRegression(tau=0.7, exact_score=False, prior_sd=10.0),
        True,
    ),
    (
        "quantregression3",
        lambda: QuantRegression3(tau=0.7, exact_score=False, prior_sd=10.0),
        True,
    ),
    ("kyphosis", lambda: Kyphosis(prior_sd=10.0), True),
    ("orings", lambda: Orings(prior_sd=10.0), True),
    ("breastfeed", lambda: Breastfeed(prior_sd=10.0), True),
    ("job", lambda: Job(), False),
    ("gee", lambda: GEE(prior_sd=10.0), True),
    ("cushings", lambda: Cushings(prior_sd=10.0), True),
    ("cushings_logistic", lambda: CushingsLogistic(prior_sd=10.0), True),
]


def _normalized_theta(model, key):
    theta = model.init_theta(key)
    theta = np.asarray(theta)
    if theta.ndim == 0:
        theta = theta[None]
    return theta


@pytest.mark.parametrize("_,factory,_supports_term", MODEL_FACTORIES, ids=[m[0] for m in MODEL_FACTORIES])
def test_log_posterior_runs_for_all_models(_, factory, _supports_term):
    model = factory()
    theta = _normalized_theta(model, jax.random.PRNGKey(0))

    lp = model.log_posterior(theta)

    if isinstance(model, Job):
        assert np.asarray(lp).ndim == 0
    elif isinstance(model, Orings):
        assert np.isfinite(np.asarray(lp)).all()
    else:
        assert np.isfinite(np.asarray(lp)).all()


@pytest.mark.parametrize("name,factory,supports_term", MODEL_FACTORIES, ids=[m[0] for m in MODEL_FACTORIES])
def test_log_posterior_term_behavior(name, factory, supports_term):
    model = factory()
    theta = _normalized_theta(model, jax.random.PRNGKey(1))

    if not supports_term:
        with pytest.raises((NotImplementedError, TypeError)):
            model.log_posterior_term(theta, 0)
        return

    prior_term = model.log_posterior_term(theta, 0)
    assert np.isfinite(np.asarray(prior_term)).all(), f"prior term failed for {name}"

    data_term = model.log_posterior_term(theta, 1)
    assert np.isfinite(np.asarray(data_term)).all(), f"data term failed for {name}"


def test_log_posterior_terms_allow_uneven_final_batch():
    model = Regression(prior_sd=10.0, n_points=30)
    theta = _normalized_theta(model, jax.random.PRNGKey(2))

    # 100 data points split into batches of 30 gives 4 data terms plus the prior.
    assert model.n_terms == 5

    terms = [model.log_posterior_term(theta, idx) for idx in range(model.n_terms)]
    assert np.isfinite(np.asarray(terms)).all()

    out_of_bounds = model.log_posterior_term(theta, model.n_terms)
    assert np.isnan(np.asarray(out_of_bounds)).all()


def test_log_posterior_terms_sum_to_full_log_posterior_without_divisibility():
    model = Regression(prior_sd=10.0, n_points=30)
    theta = _normalized_theta(model, jax.random.PRNGKey(3))

    term_sum = sum(model.log_posterior_term(theta, idx) for idx in range(model.n_terms))
    full_log_posterior = model.log_posterior(theta, check_sum_to_1=False)

    np.testing.assert_allclose(term_sum, full_log_posterior, rtol=1e-6, atol=1e-6)
