import jax
import jax.numpy as jnp
from jax import vmap, jit, grad, value_and_grad

# from flax import struct
import equinox as eqx
import chex
import optax

from functools import partial
import dataclasses

from jax import Array
from typing import Any
from abc import abstractmethod
from collections.abc import Callable
from timeit import default_timer as timer
import logging

PyTree = Any
KeyArray = Array
Covariance = Array  # full covariance or diagonal

Distribution = Any  # Gaussian or Uniform


class Uniform(eqx.Module):
    low: int
    high: int

    def sample(self, key):
        return jax.random.uniform(key, shape=(1,), minval=self.low, maxval=self.high)

    def log_prob(self, x):
        # Potentially unnormalized log density
        x = jnp.squeeze(x)
        return jax.scipy.stats.uniform.logpdf(
            x, loc=self.low, scale=self.high - self.low
        )


class Gaussian(eqx.Module):
    mu: Array

    @property
    @abstractmethod
    def Sigma(self) -> Covariance:
        pass

    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        # return unnormalized log density if raw is true.
        pass

    @abstractmethod
    def transform(self, eps: Array) -> Array:
        pass

    def base_sampler(self, key: KeyArray) -> Array:
        return jax.random.normal(key, shape=self.mu.shape)

    def sample(self, key: KeyArray, n_samples=None) -> Array:
        if n_samples is None:
            return self.transform(self.base_sampler(key))
        else:
            return vmap(self.sample)(jax.random.split(key, n_samples))


class DiagonalGaussian(Gaussian):
    log_std: Array  # a vector of log standard deviations

    def __init__(self, mu: Array, log_std: Array):
        self.mu = mu
        self.log_std = log_std

    @classmethod
    def initialized(cls, key: KeyArray, dim: int):
        subkey1, subkey2 = jax.random.split(key)
        return cls(
            mu=jnp.zeros((dim,)),
            log_std=jax.random.normal(subkey1, (dim,)) * 0.1,
        )

    @property
    def Sigma(self) -> Covariance:
        return jnp.diag(jnp.exp(self.log_std * 2))

    def transform(self, eps: Array) -> Array:
        return self.mu + jnp.exp(self.log_std) * eps

    @staticmethod
    def compute_log_prob(
        x: Array,
        mu: Array,
        log_std: Array,
    ) -> Array:
        return jnp.sum(vmap(jax.scipy.stats.norm.logpdf)(x, mu, jnp.exp(log_std)))

    def log_prob(self, x):
        # Potentially unnormalized log density
        return self.compute_log_prob(x, self.mu, self.log_std)

    def __repr__(self) -> str:
        return f"N(\u03BC={self.mu}, \u03C3={jnp.diag(self.Sigma)})"


class LowRankGaussian(Gaussian):
    log_std: Array  # a vector of log standard deviations
    F: Array

    def __init__(self, mu: Array, log_std: Array, F: Array):
        self.mu = mu
        self.log_std = log_std
        self.F = F

    def transform(self, eps: Array) -> Array:
        cov = jnp.diag(jnp.exp(self.log_std * 2)) + self.F @ self.F.T
        L = jnp.linalg.cholesky(cov)  # lower triangular cholesky factor
        return self.mu + L @ eps

    @staticmethod
    def compute_log_prob(
        x: Array,
        mu: Array,
        log_std: Array,
        F: Array,
    ) -> Array:
        cov = jnp.diag(jnp.exp(log_std * 2)) + F @ F.T
        return jax.scipy.stats.multivariate_normal.logpdf(x, mu, cov)

    def log_prob(self, x):
        return self.compute_log_prob(x, self.mu, self.log_std, self.F)


class FullGaussian(Gaussian):
    log_diag_L: Array  # a vector of log diagonal entries of the lower Cholesky
    off_L: Array  # a vector of off diagonal entries of the lower Cholesky

    def __init__(self, mu: Array, log_diag_L: Array, off_L: Array):
        self.mu = mu
        self.log_diag_L = log_diag_L
        self.off_L = off_L

    @classmethod
    def initialized(cls, key: KeyArray, dim: int):
        subkey1, subkey2 = jax.random.split(key)
        return cls(
            mu=jnp.zeros((dim,)),
            log_diag_L=jax.random.normal(subkey1, (dim,)) * 0.1,
            off_L=jax.random.normal(subkey2, ((dim - 1) * dim // 2,)) * 0.1,
        )

    @property
    def Sigma(self) -> Covariance:
        L = self.__class__.construct_cholesky(self.log_diag_L, self.off_L)
        return L @ L.T

    @staticmethod
    def construct_cholesky(log_diag, off_diag):
        # Construct a cholesky factor from its log-diagonal and off-diagonal elements
        chex.assert_shape(off_diag, (None,))
        chex.assert_shape(log_diag, (None,))
        idx = jnp.tril_indices(log_diag.shape[0], k=-1)
        return jnp.diag(jnp.exp(log_diag)).at[idx].set(off_diag)

    def transform(self, eps: Array) -> Array:
        L = self.__class__.construct_cholesky(self.log_diag_L, self.off_L)
        return self.mu + L @ eps

    @staticmethod
    def compute_log_prob(
        x: Array,
        mu: Array,
        log_diag_L: Array,
        off_L: Array,
    ) -> Array:
        L = FullGaussian.construct_cholesky(log_diag_L, off_L)
        return jax.scipy.stats.multivariate_normal.logpdf(x, mu, L @ L.T)

    def log_prob(self, x) -> Array:
        return self.compute_log_prob(x, self.mu, self.log_diag_L, self.off_L)

    def __repr__(self) -> str:
        return f"N(\u03BC={self.mu}, \u03A3={self.Sigma})"


def rejection_sampler(
    key: KeyArray, sampler: Callable[[KeyArray], Array], oracle: Callable[[Array], bool]
) -> Any:
    """
    A rejection sampler that draws a sampler from the sampler until it's
    accepted by the oracle.

    :param key: A PRNG key
    :param sampler: A function that takes in a key and generate a sample
    :param oracle: A function that in a sample and returns True if it's good

    :return: A tuple of (key, x) where x == sampler(key) and oracle(x) == True
    """

    def resample_x(carry):
        key, _, count = carry
        key, xkey = jax.random.split(key)
        return (xkey, sampler(xkey), count + 1)

    def if_reject(carry):
        _, x, _ = carry
        return ~oracle(x)

    key, xkey = jax.random.split(key)
    init_carry = (xkey, sampler(xkey), 1)
    xkey, x, count = jax.lax.while_loop(if_reject, resample_x, init_carry)
    # jax.debug.print("run rejection sampler: {} times", count)
    return xkey, x


def square_bound(gaussian, x):
    sd = jnp.sqrt(jnp.diag(gaussian.Sigma))
    chex.assert_shape(sd, gaussian.mu.shape)
    # return jnp.all(jnp.abs((x - gaussian.mu) / sd) < 0.5)
    return jnp.all(((jnp.abs(x) - gaussian.mu) / sd) < 0.5)


############## VERSION 2 ################


class ElboLoss:
    """The negative evidence lower bound (ELBO), approximated using samples. Adapted from flowjax.

    Args:
        params: Parameters for the model
        static: Static components of the model.
        *args: Arguments passed to the loss function.
        optimizer: Optax optimizer.
        opt_state: Optimizer state.
        loss_fn: The loss function. This should take params and static as the first two
            arguments.
        **kwargs: Key word arguments passed to the loss function.

    Returns:
        tuple: (params, opt_state, loss_value, debug_dict)
        num_samples: Number of samples to use in the ELBO approximation.
        target: The target, i.e. log posterior density up to an additive constant / the
            negative of the potential function, evaluated for a single point.
    """

    target: Callable
    num_samples: int

    def __init__(
        self,
        target: Callable,
        num_samples: int,
    ):
        self.target = target
        self.num_samples = num_samples

    @eqx.filter_jit
    def __call__(
        self,
        params: Distribution,
        static: Distribution,
        key: KeyArray,
    ) -> float:
        """Compute the ELBO loss.

        Args:
            params: The trainable parameters of the model.
            static: The static components of the model.
            key: Jax random key.
        """

        dist = eqx.combine(params, static)
        samples = dist.sample(key, (self.num_samples,))
        log_probs = dist.log_prob(samples)

        target_density = vmap(self.target)(samples)
        return jnp.mean(log_probs - target_density, axis=0)


def renyi_bound_loss(
    target: Callable[[Array], float],
    num_samples: int,
    alpha: float,
) -> Callable[[Distribution, Distribution, KeyArray], float]:
    """
    Variational Renyi bound.  See Li and Turner 2016, Eq 4.  Note that this is a
    loss function, i.e. the negative of the evidence bound.
    """

    @partial(jax.custom_jvp, nondiff_argnums=(1,))
    def loss_fn(
        params: Distribution,
        static: Distribution,
        key: KeyArray,
    ) -> float:
        """
        Compute the loss.  This is a biased estimator of the the negative of the
        Renyi variational bound.  The bias is due to taking the log of a Monte
        Carlo estimator.
        """
        q_dist = eqx.combine(params, static)

        def loss_per_sample(k):
            x, q_lp = q_dist.sample_and_log_prob(k)
            return (1 - alpha) * (target(x) - q_lp)

        loss_keys = jax.random.split(key, num_samples)
        batch_loss = vmap(loss_per_sample)(loss_keys)
        loss = jax.scipy.special.logsumexp(batch_loss) - jnp.log(num_samples)
        loss = loss / (alpha - 1)
        return loss

    @loss_fn.defjvp
    def loss_fn_jvp(static, primals, tangents):
        """
        Compute the pathwise gradient of the loss.  This is an unbiased Monte
        Carlo estimator of the gradient of the negative of the Renyi variational
        bound.  See Eq 8 of Li and Turner
        """
        params, key = primals
        params_dot, _ = tangents
        loss_keys = jax.random.split(key, num_samples)

        @value_and_grad
        def grad_diff_lp(params, k):
            # Grad of log prob difference, per sample
            q_dist = eqx.combine(params, static)
            x, q_lp = q_dist.sample_and_log_prob(k)
            diff_lp = target(x) - q_lp
            return diff_lp

        batch_diff_lp, batch_grad = vmap(grad_diff_lp, (None, 0))(params, loss_keys)
        batch_loss = (1 - alpha) * batch_diff_lp
        loss = jax.scipy.special.logsumexp(batch_loss) - jnp.log(num_samples)
        loss = loss / (alpha - 1)

        batch_weight = jax.nn.softmax((1 - alpha) * batch_diff_lp)
        chex.assert_tree_shape_prefix([batch_weight, batch_grad], (num_samples,))

        def grad_loss(w, tree_g):
            # Grad of loss, per sample
            return jax.tree.map(lambda g: -1 * w * g, tree_g)

        batch_grad_loss = vmap(grad_loss)(batch_weight, batch_grad)
        grad_loss = jax.tree.map(lambda x: jnp.mean(x, axis=0), batch_grad_loss)
        chex.assert_trees_all_equal_shapes(grad_loss, params_dot)
        tangent = jax.tree.map(lambda g, p: jnp.sum(g * p), grad_loss, params_dot)
        tangent = jax.tree.reduce(lambda t, a: t + a, tangent, 0.0)
        return loss, tangent

    return loss_fn


class BoundedScoreLoss(eqx.Module):
    """The L2 of the score difference in a boundary specified by an oracle, approximated using samples.

    Args:
        num_samples: Number of samples to use in the ELBO approximation.
        target: The target, i.e. log posterior density up to an additive constant / the
            negative of the potential function, evaluated for a single point.
    """

    target: Callable[[Array], float]
    num_samples: int
    oracle: Callable[[Array], bool]

    def __init__(
        self,
        target: Callable[[Array], Array],
        num_samples: int,
        oracle: Callable = False,
    ):
        self.target = target
        self.num_samples = num_samples
        self.oracle = oracle

    def __call__(
        self,
        params: Distribution,
        static: Distribution,
        key: KeyArray,
    ) -> float:
        """Compute the L2 of the score difference.

        Args:
            params: The trainable parameters of the model.
            static: The static components of the model.
            key: Jax random seed.
        """

        def truncated_sampler(key):
            # Return the key that, when passed to the sampler it will be accepted by
            # the oracle. NOTE: THIS FUNCTION CANNOT GO INSIDE L2 BECAUSE OF THE
            # WHILE LOOP INSIDE THE REJECTION SAMPLER
            if self.oracle is None:
                xkey = key
            else:
                q_dist = eqx.combine(params, static)
                xkey, _ = rejection_sampler(key, q_dist.sample, self.oracle)
            return xkey

        def l2_score(xkey):
            # L2 of the difference between the score of f and q
            q_dist = eqx.combine(params, static)
            f_score = grad(self.target)
            q_score = grad(q_dist.log_prob)
            x = q_dist.sample(xkey)  # this is guaranteed to have finite f and q score
            return jnp.sum((f_score(x) - q_score(x)) ** 2)

        subkeys = jax.random.split(key, self.num_samples)
        xkeys = vmap(truncated_sampler)(subkeys)

        return jnp.mean(vmap(l2_score)(xkeys), axis=0)


def vi_step(
    params: Distribution,
    static: Distribution,
    key: KeyArray,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: Callable[[Distribution, Distribution, KeyArray], float],
) -> tuple[Distribution, PyTree, float, dict]:
    """Perform a single training step of variational inference.

    :param params: Parameters for the model
    :param static: Static components of the model.
    :param key: Jax PRNGKey.
    :param optimizer: Optax optimizer.
    :param opt_state: Optimizer state.
    :param loss_fn: The loss function. This should take params and static as the first two
        arguments.

    :return: A tuple containing the updated parameters, optimizer state, loss value, and
        a dictionary of debugging information.
    """
    key, subkey = jax.random.split(key)
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, static, subkey)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)
    return (
        params,
        opt_state,
        loss_val,
        {"grads": optax.global_norm(grads)},
    )  # dictionary for debugging


def fit_to_variational_target(
    key: KeyArray,
    dist: Distribution,
    loss_fn: Callable[[Distribution, Distribution], float],
    optimizer: optax.GradientTransformation,
    iters: int = 100,
    save_freq: int = 1000,
    show_progress: bool = True,
) -> tuple[PyTree, list]:
    """Train a Distribution by variational inference.

    :param key: Jax PRNGKey.
    :param dist: Initial value in Distribution object.
    :param loss_fn: The loss function to optimize (e.g. the ElboLoss), taking in
        params, static, and a key.
    :param optimizer: Optax optimizer.
    :param iters: The number of training steps to run.  Defaults to 100.
    :param show_progress: Whether to print progress.
    :param save_freq: How often to save the variational approxiamtion.

    :return: A tuple containing a list of trained distributions along the
             optimization, the final/best distribution, a list of losses, and a
             list of times taken for each step.
    """
    params, static = eqx.partition(dist, eqx.is_array)
    opt_state = optimizer.init(params)

    @jit
    def jit_step(params, key, opt_state):
        return vi_step(params, static, key, optimizer, opt_state, loss_fn)

    jit_step = jit_step.lower(params, key, opt_state).compile()

    losses = []
    time_ls = [0.0]
    q_ls = [eqx.combine(params, static)]

    for i in range(iters):
        key, subkey = jax.random.split(key)
        if i % max((iters // 50), 1) == 0:
            # Print at every 100th iteration
            _, _, loss, debug_dict = jit_step(params, subkey, opt_state)
            debug_str = ", ".join([f"{k}: {v:.4f}" for k, v in debug_dict.items()])
            logging.info(f"iter: {i}, loss: {loss:.4f} " + debug_str)

        if i % save_freq == 0:
            # start at 0, save_freq, 2*save_freq, ...
            start = timer()

        params, opt_state, loss, _ = jit_step(params, subkey, opt_state)

        if (i - save_freq + 1) % save_freq == 0:
            # end at save_freq - 1, 2*save_freq - 1, ...
            losses.append(loss)
            q_ls.append(eqx.combine(params, static))
            jax.block_until_ready(params)
            time_ls.append(timer() - start)

    best_dist = q_ls[jnp.argmin(jnp.asarray(losses))]
    return q_ls, best_dist, losses, time_ls


def main():
    key = jax.random.PRNGKey(0)
    N = 10

    # treat q as a dict of {mu : ..., Lambda: ...}
    # log_target = DiagonalGaussian(low=-(0.2, high=0.2).log_prob
    ones = jnp.ones((2,))
    # p = DiagonalGaussian(mu=0.4 * ones, log_std=jnp.log(jnp.sqrt(1.9)) * ones)
    p = FullGaussian.initialized(jax.random.key(30), 2)
    p = dataclasses.replace(p, mu=p.mu + 3, log_diag_L=p.log_diag_L)
    print(p)
    oracle = partial(square_bound, p)
    log_target = p.log_prob
    # init_q = DiagonalGaussian(mu=-0.4 * ones, log_std=1.1 * ones)
    init_q = FullGaussian.initialized(jax.random.key(35), 2)

    loss_fn = BoundedScoreLoss(log_target, N, oracle)
    # final_q, score_ls = run_score_matching(key, log_target, init_q, N, oracle)
    final_q, score_ls = fit_to_variational_target(
        key=key,
        dist=init_q,
        loss_fn=loss_fn,
        steps=10000,
        learning_rate=0.01,
        optimizer=None,
        return_best=True,
        show_progress=True,
    )

    print(final_q)
    from matplotlib import pyplot as plt

    plt.plot(score_ls)


if __name__ == "__main__":
    main()

