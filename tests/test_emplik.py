from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap
import chex
import pytest

import numpy as np
from numpy.testing import assert_allclose

import emplik

jax.config.update("jax_enable_x64", True)
chex.set_n_cpu_devices(4)


@pytest.fixture
def data():
    return np.array([1, 2, 3, 4, 5])[:, np.newaxis]


def test_lambda(data):

    def weight_from_scratch(h, theta):
        H = vmap(h, in_axes=(0, None))(data, theta)
        lbd = emplik.calc_lambda(H)
        log_w = emplik.calc_log_w(lbd, H)
        return jnp.exp(log_w)

    # No constraint g(x, theta) = 0
    h = lambda x, theta: jnp.array([0])
    assert_allclose(
        np.asarray(weight_from_scratch(h, 2.0)),
        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    )
    assert_allclose(
        weight_from_scratch(h, 0.0),
        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    )

    # Unbiased estimator of theta = E[x], g(x, theta) = x - theta
    h = lambda x, theta: x - theta
    # Max at sample mean X_bar = 3.0
    assert_allclose(
        weight_from_scratch(h, 3.0),
        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    )
    # Symmetry
    assert_allclose(
        weight_from_scratch(h, 2.0),
        np.flip(weight_from_scratch(h, 4.0)),
    )

    # Sum of weights equal to 1
    assert_allclose(sum(weight_from_scratch(h, 2.0)), 1.0, atol=1e-5)


@pytest.mark.parametrize("use_jit", [False, True])
def test_log_el(data, use_jit):
    # No constraint g(x, theta) = 0
    h = lambda x, theta: jnp.array([0])
    calc_log_pel = emplik.calc_log_pel
    if use_jit:
        calc_log_pel = jax.jit(calc_log_pel, static_argnums=(0,))
    assert_allclose(calc_log_pel(h, data, 2.0), -5 * np.log(5))

    # Unbiased estimator of theta = E[x], g(x, theta) = x - theta
    h = lambda x, theta: x - theta

    assert_allclose(calc_log_pel(h, data, 3.0), -5 * np.log(5))

    # 3.0 should maximize the empirical likelihood.
    assert_allclose(jax.grad(calc_log_pel, argnums=2)(h, data, 3.0), 0.0)

    calc_log_pel = partial(emplik.calc_log_pel, max_iter=1000, tol=1e-4)

    # At or outside the boundary, the objective should blow up.
    assert calc_log_pel(h, data, 1.0) < -50
    assert calc_log_pel(h, data, 5.0) < -50
    assert calc_log_pel(h, data, 6.0) < -50

    # Check if 1-dimensional g works properly.
    g = lambda x, theta: x - theta
    assert_allclose(calc_log_pel(g, data, 3.0), -5 * np.log(5))
