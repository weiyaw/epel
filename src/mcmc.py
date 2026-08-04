import jax
import jax.numpy as jnp
from jax import grad, jit
from jax import Array
import blackjax

from collections.abc import Callable
from typing import Any

from tqdm import tqdm
import numpy as np
from timeit import default_timer as timer
import logging

KeyArray = Any  # jax PRNG key
PyTree = Any  # jax PyTree


def nuts(
    key: KeyArray,
    log_posterior: Callable[[Array], float],
    initial_theta: Array,
    n_samples: int = 2000,
    n_warmup: int = 1000,
    initial_step_size: float = 1e-2,
) -> tuple[PyTree, PyTree, list[float]]:
    """Run NUTS sampler with a window adaptation step.

    Run NUTS sampler with a window adaptation step.

    :param key: PRNG key
    :param log_posterior: Callable, the log posterior function
    :param initial_theta: jnp.ndarray, the initial value of the parameter
    :param n_samples: int, the number of samples to draw
    :param n_warmup: int, the number of warmup steps
    :param initial_step_size: float, the initial step size for the sampler

    :return: A tuple of output.  The first element is an array of state/samples
             of HMC and its corresnpoding log density.  The second element is
             the diagnostic info (e.g. energy, acceptance probability).  The
             third element is the wall-clock time of each iteration.
    """

    # Window-adapted NUTS
    adapt = blackjax.window_adaptation(
        blackjax.nuts,
        log_posterior,
        initial_step_size=initial_step_size,
        progress_bar=True,
    )
    key, subkey = jax.random.split(key)
    adapt_res, _ = adapt.run(subkey, initial_theta, num_steps=n_warmup)
    sampler = blackjax.nuts(log_posterior, **adapt_res.parameters)
    state, info = adapt_res  # use the warmed state from adaptation

    # Iterate (actual sampling)
    state_ls = [state]
    info_ls = []
    time_ls = [0.0]
    step = jit(sampler.step)
    for _ in tqdm(range(n_samples)):
        start = timer()
        key, subkey = jax.random.split(key)
        state, info = step(subkey, state)
        time_ls.append(timer() - start)
        state_ls.append(state)
        info_ls.append(info)

    # Combine results from lists of 1D-arrays into stacked arrays
    stacked_state = jax.tree.map(lambda *x: np.stack(x), *state_ls)
    stacked_info = jax.tree.map(lambda *x: np.stack(x), *info_ls)

    return stacked_state, stacked_info, time_ls


def hmc(
    key: KeyArray,
    log_posterior: Callable[[Array], float],
    initial_theta: Array,
    n_samples: int = 2000,
    n_warmup: int = 1000,
    step_size: float = 1e-2,
    inverse_mass_matrix: Array = None,
    num_integration_steps: int = 50,
) -> tuple[PyTree, PyTree, list[float]]:
    """Run HMC sampler.

    Run HMC sampler without any adaptation step.

    :param key: PRNG key
    :param log_posterior: Callable, the log posterior function
    :param initial_theta: jnp.ndarray, the initial value of the parameter
    :param n_samples: int, the number of samples to draw
    :param n_warmup: int, the number of warmup steps
    :param step_size: float, the initial step size for the sampler
    :param inverse_mass_matrix: jnp.ndarray, a vector specifying the diagonal of
        the inverse mass matrix
    :param num_integration_steps: int, the number of integration steps

    :return: A tuple of output.  The first element is an array of state/samples
             of HMC and its corresnpoding log density.  The second element is
             the diagnostic info (e.g. energy, acceptance probability).  The
             third element is the wall-clock time of each iteration.
    """

    sampler = blackjax.hmc(
        log_posterior,
        step_size=step_size,
        num_integration_steps=num_integration_steps,
        inverse_mass_matrix=inverse_mass_matrix,
    )
    state = sampler.init(initial_theta)

    jit_step = jit(sampler.step).lower(key, state).compile()

    # Iterate (warmup)
    for _ in tqdm(range(n_warmup)):
        key, subkey = jax.random.split(key)
        state, info = jit_step(subkey, state)

    # Iterate (actual sampling)
    state_ls = [state]
    info_ls = []
    time_ls = [0.0]
    for _ in tqdm(range(n_samples)):
        start = timer()
        key, subkey = jax.random.split(key)
        state, info = jit_step(subkey, state)
        jax.block_until_ready(state)
        time_ls.append(timer() - start)
        state_ls.append(state)
        info_ls.append(info)

    assert len(state_ls) == len(info_ls) + 1 == len(time_ls)
    # Combine results from lists of 1D-arrays into stacked arrays
    stacked_state = jax.tree.map(lambda *x: np.stack(x), *state_ls)
    stacked_info = jax.tree.map(lambda *x: np.stack(x), *info_ls)

    return stacked_state, stacked_info, time_ls


def random_walk(
    key: KeyArray,
    log_posterior: Callable[[Array], float],
    initial_theta: Array,
    sd_shrink_factor: float,
    n_samples: int = 5000,
    n_warmup: int = 1000,
    thinning: int = 1,
) -> tuple[PyTree, PyTree, list[float]]:
    """Run random walk Rosenbluth-Metropolis-Hastings sampler.

    First, run a warmup to estimate the covariance of the posterior, then set
    the proposal covariance to this estimate and do the actual sampling.

    :param key: PRNG key
    :param log_posterior: the log posterior function
    :param initial_theta: the initial value of the parameter
    :param sd_shrink_factor: the proposal sd (i.e. cholesky factor of the
        covariance) is estimated from the warmup run, but we also multiply its
        by this factor to adjust its magnitude
    :param n_samples: the number of samples to draw
    :param n_warmup: the number of warmup steps
    :param thinning: int, keep every thinning-th sample (default 1, no thinning)

    :return: A tuple of output.  The first element is an array of state/samples
             of random walk and its corresnpoding log density.  The second
             element is the diagnostic info (e.g. acceptance probability).  The
             third element is the wall-clock time to generate each saved sample
             (aggregated over `thinning` step calls).
    """

    initial_proposal_cov = jnp.eye(initial_theta.size) * 0.1**2
    warmup_cov_chol = jnp.linalg.cholesky(initial_proposal_cov, upper=False)
    warmup_sampler = blackjax.additive_step_random_walk.normal_random_walk(
        log_posterior, sigma=warmup_cov_chol
    )
    state = warmup_sampler.init(initial_theta)
    jit_warmup_step = jit(warmup_sampler.step).lower(key, state).compile()

    # Iterate (warmup)
    state_ls = [state]
    info_ls = []
    for _ in tqdm(range(n_warmup)):
        key, subkey = jax.random.split(key)
        state, info = jit_warmup_step(subkey, state)
        state_ls.append(state)
        info_ls.append(info)

    stacked_state = jax.tree.map(lambda *x: np.stack(x), *state_ls)
    stacked_info = jax.tree.map(lambda *x: np.stack(x), *info_ls)
    logging.info(f"Acceptance prob (warmup): {stacked_info.is_accepted.mean()}")

    # Compute the covariance from the warmup samples
    _, upper_tri = jax.scipy.linalg.qr(stacked_state.position, mode="economic")

    actual_sampler = blackjax.additive_step_random_walk.normal_random_walk(
        log_posterior,
        sigma=jnp.transpose(upper_tri) / jnp.sqrt(n_warmup - 1) * sd_shrink_factor,
    )
    jit_actual_step = jit(actual_sampler.step).lower(key, state).compile()

    state_ls = [state]
    info_ls = []
    time_ls = [0.0]
    start = timer()
    # Iterate (actual sampling)
    for i in tqdm(range(n_samples)):
        key, subkey = jax.random.split(key)
        state, info = jit_actual_step(subkey, state)
        jax.block_until_ready(state)
        if (i + 1) % thinning == 0:
            state_ls.append(state)
            info_ls.append(info)
            time_ls.append(timer() - start)
            start = timer()

    assert len(state_ls) == len(info_ls) + 1 == len(time_ls)
    # Combine results from lists of 1D-arrays into stacked arrays
    stacked_state = jax.tree.map(lambda *x: np.stack(x), *state_ls)
    stacked_info = jax.tree.map(lambda *x: np.stack(x), *info_ls)
    logging.info(f"Acceptance prob (actual): {stacked_info.is_accepted.mean()}, Thinning: {thinning}")

    return stacked_state, stacked_info, time_ls
