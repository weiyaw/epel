import jax
import jax.flatten_util
import jax.numpy as jnp
from jax.scipy.special import expit, logit
from jax import Array

import chex
import numpy as np
import pandas as pd

import equinox as eqx
from functools import partial
from typing import Any
import math

from abc import abstractmethod
from collections.abc import Callable
import utils
import emplik

from sklearn.linear_model import QuantileRegressor, LogisticRegression

PyTree = Any
Distribution = Any
KeyArray = Any  # jax PRNG key


def standardize(array):
    """Standardize the array along its 0-th axis to have zero mean and unit variance"""
    mean = np.mean(array, axis=0)
    std = np.std(array, ddof=1, axis=0)  # unbiased estimator of the variance
    return (array - mean) / std


def log_prob_tree(tree: PyTree, dist: Distribution):
    """Compute the log density of a tree given a distribution.

    This method is equivalent to unravel a tree, then treat each element of the
    unravelled vector as an independent random variable from the given
    distribution, and compute the log density of the vector.
    """

    def accumulate_lp(x, y):
        lp_leave = dist.log_prob(y)
        chex.assert_equal_shape([lp_leave, y])
        return x + jnp.sum(lp_leave)

    lp = jax.tree_util.tree_reduce(accumulate_lp, tree, 0)
    return lp


class BayesELModel(eqx.Module):
    """A Bayesian empirical likelihood model with prior on the parameters of
    interest.

    :param data: The whole dataset.  The leading dimension of each array
        corresponds to samples.
    :param dim_theta: Number of parameters of interest.
    :param n_terms: Number of product terms in the posterior.
    """

    data: PyTree
    dim_theta: int
    n_terms: int

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def h(self, dp: PyTree, theta: Array) -> Array:
        """The constraint function of EL.

        :param dp: A PyTree of the data point.
        :param theta: A 1D array parameters of interest.

        :return: The constraint evaluated at dp and theta.
        """
        pass

    @abstractmethod
    def init_theta(self, key: KeyArray) -> Array:
        """
        Return a sensible initial guess of the parameter of interest.
        """
        pass

    @abstractmethod
    def log_prior(self, theta: Array) -> float:
        """Log prior of the parameters of interest.

        :param theta: A 1D array parameters of interest.  Must be compatible
            with self.h.

        :return: log prior
        """
        pass

    def log_posterior(
        self,
        theta: Array,
        check_sum_to_1: bool = True,
        use_llog: bool = False,
        use_ael: bool = False,
    ) -> float:
        """Log posterior on the whole dataset.

        :param theta: A 1D array parameters of interest.  Must be compatible
            with self.h.
        :param check_sum_to_1: Whether to check if the sum of the empirical
            likelihood weights sum to 1.
        :param use_llog: Whether to use llog when computing the individual
            weight of empirical likelihood, log(wi).
        :param use_ael: Whether to use the adjusted empirical likelihood.

        :return: log posterior
        """

        # log profile EL
        chex.assert_shape(theta, (self.dim_theta,))
        if use_ael:
            n_data = utils.get_tree_lead_dim(self.data)
            log_el = emplik.calc_log_ael(
                self.h,
                self.data,
                theta,
                a_n=math.log(n_data) / 2,  # from Chen et al 2008
                use_llog=use_llog,
                check_sum_to_1=check_sum_to_1,
            )
        else:
            log_el = emplik.calc_log_pel(
                self.h,
                self.data,
                theta,
                use_llog=use_llog,
                check_sum_to_1=check_sum_to_1,
            )
        log_prior = self.log_prior(theta)
        return log_el + log_prior

    def log_posterior_term(
        self, theta: Array, idx: int, use_llog: bool = False
    ) -> float:
        """Return the i^th term of the log posterior.

        The log posterior can often be written as a sum of many subcomponent.
        This routine returns the individual component of the log posterior.  The
        0th term is the prior, while 1st to Nth terms corresponds to the EL
        weight of a data point.

        :param theta: A 1D array parameters of interest.  Must be compatible
            with self.h.
        :param idx: The index of posterior component.  0th term is the prior.
            1-Nth is the EL weight.
        :param use_llog: Whether to use llog when computing the individual EL
            weight, log(wi).

        :return: Either the log prior or the log(wi) corresponding to the i^th
                 data point.
        """

        n_data = utils.get_tree_lead_dim(self.data)
        assert (
            n_data % (self.n_terms - 1) == 0
        ), "n_data must be a multiple of (n_terms - 1)"
        indices = jnp.reshape(jnp.arange(n_data), (self.n_terms - 1, -1))
        chex.assert_shape(theta, (self.dim_theta,))

        def log_wi():
            # The log weight on a batch of data point.
            H_full = jax.vmap(lambda x: self.h(x, theta))(self.data)
            lbd = emplik.calc_lambda(H_full, theta)
            H_batch = H_full[indices[idx - 1]]
            log_w = emplik.calc_log_w(lbd, H_batch, use_llog=use_llog)
            # Return nan if the index is out of bound
            return jnp.where(idx < self.n_terms, jnp.sum(log_w, axis=0), jnp.nan)

        return jax.lax.cond(
            idx > 0,
            log_wi,
            lambda: self.log_prior(theta),
        )


class InvGamma(eqx.Module):
    shape: float
    rate: float

    def log_prob(self, x: Array) -> Array:
        scale = 1 / self.rate
        lp = jax.scipy.stats.gamma.logpdf(1 / x, self.shape, scale=scale)
        lp = lp - 2 * jnp.log(x)  # density correction
        chex.assert_equal_shape([lp, x])
        return lp


def norm_raw_log_prob(x: Array, mean: float, sd: float) -> float:
    # unnormalized log density of a normal distribution
    chex.assert_shape(mean, ())
    chex.assert_shape(sd, ())
    lp = -((x - mean) ** 2) / (2 * sd**2)
    chex.assert_equal_shape([lp, x])
    return jnp.sum(lp)


def invgamma_raw_log_prob(x: Array, shape: float, scale: float) -> float:
    # unnormalized log density of an inverse gamma distribution
    chex.assert_shape(shape, ())
    chex.assert_shape(scale, ())
    lp = -(shape + 1) * jnp.log(x) - scale / x
    lp = jnp.where(x > 0, lp, -jnp.inf)
    chex.assert_equal_shape([lp, x])
    return jnp.sum(lp)


class Simple(BayesELModel):
    """Unbiased estimator of theta = E[x], g(x, theta) = x - theta, on a simple
    dataset
    """

    def __init__(self):
        self.data = np.array([1, 2, 3, 4, 5])[:, np.newaxis]
        self.dim_theta = 1
        self.n_terms = 5 + 1

    def h(self, dp: Array, theta: Array) -> Array:
        return jnp.array([dp]) - theta

    def init_theta(self, key: KeyArray) -> Array:
        return 3.0

    def log_prior(self, theta: Array) -> float:
        return jax.scipy.stats.norm.logpdf(theta, 0, 1)


class Regression(BayesELModel):
    """Regression coefficients on a scalar-y dataset.  Orthogonality constraint:
    g(x, theta) = [1 x] * (y - [1 x] @ theta) = 0

    :param true_theta: The true regression coefficients for generating the data.
    """

    prior_sd: float

    def __init__(self, prior_sd: float, n_terms: int | None = None):
        n_data = 100
        true_theta = jnp.array([0.5, 1.0])
        self.dim_theta = true_theta.size
        self.data = self.gen_data(n_data, true_theta)
        self.prior_sd = prior_sd
        self.n_terms = n_terms if n_terms is not None else 101

    @staticmethod
    def gen_data(n_data: int, true_theta: Array) -> PyTree:
        dim_theta = true_theta.size
        key1, key2 = jax.random.split(jax.random.key(20), 2)
        x = jax.random.normal(key1, shape=(n_data, dim_theta - 1))
        eps = jax.random.normal(key2, shape=(n_data,))
        y = true_theta[0] + x @ true_theta[1:] + eps
        return {"x": x, "y": y}

    def h(self, dp: Array, theta: Array) -> Array:
        y = dp["y"]
        x_with_1 = jnp.insert(dp["x"], 0, 1.0)  # add intercept
        chex.assert_equal_shape([x_with_1, theta])
        return x_with_1 * (y - jnp.dot(x_with_1, theta))

    def init_theta(self, key: KeyArray) -> Array:
        # Use least squares solution as the initial value of theta
        n_data = utils.get_tree_lead_dim(self.data)
        X = np.concatenate([np.ones((n_data, 1)), self.data["x"]], axis=1)
        ls_theta, *_ = np.linalg.lstsq(X, self.data["y"], rcond=None)
        assert ls_theta.shape == (self.dim_theta,)
        return ls_theta

    def log_prior(self, theta: Array) -> float:
        return jnp.sum(jax.scipy.stats.norm.logpdf(theta, 0, self.prior_sd))


class Regression10(Regression):
    """Regression coefficients on a scalar-y dataset, with a 10-dimensional
    x.  Orthogonality constraint: g(x, theta) = [1 x] * (y - [1 x] @ theta) = 0
    """

    def __init__(self, prior_sd: float, n_terms: int | None = None):
        n_data = 100
        true_theta = jnp.array([0.5, 1.0, 0.5, -1.0, 0.5, 0, 0, 0, 0, 0])
        self.dim_theta = true_theta.size
        self.data = self.gen_data(n_data, true_theta)
        self.prior_sd = prior_sd
        self.n_terms = n_terms if n_terms is not None else 101


class QuantRegression(BayesELModel):
    """A very simple quantile regression"""

    tau: float  # tau = P(X < quantile)
    exact_score: bool  # whether to use the exact score or a smoothed version
    prior_sd: float

    def __init__(
        self, tau: float, exact_score: bool, prior_sd: float, n_terms: int | None = None
    ):
        n_data = 100
        self.dim_theta = 2
        self.data = self.gen_data(n_data)
        self.prior_sd = prior_sd
        self.n_terms = n_terms if n_terms is not None else 101
        self.tau = tau
        self.exact_score = exact_score

    @staticmethod
    def gen_data(n_data: int) -> PyTree:
        key1, key2 = jax.random.split(jax.random.key(20), 2)
        x = jax.random.normal(key1, shape=(n_data, 1))
        eps = jax.random.normal(key2, shape=(n_data, 1))
        y = 0.5 + x + eps
        assert y.shape == (n_data, 1)
        return {"x": x, "y": np.squeeze(y)}

    @staticmethod
    def quantile_score(u, tau, exact):
        if exact:
            return jnp.where(u == 0, jnp.where(u < 0, 1 - tau, -tau), 0)
        else:
            # A smoothed version of the quantile score
            eps = 1e-1
            return expit(-u / eps) - tau

    def h(self, dp: PyTree, theta: Array) -> Array:
        y = dp["y"]
        x_with_1 = jnp.insert(dp["x"], 0, 1.0)  # add intercept
        resid = y - jnp.dot(x_with_1, theta)
        return self.quantile_score(resid, self.tau, self.exact_score) * x_with_1

    def init_theta(self, key: KeyArray) -> Array:
        # Frequentist estimator
        fitted = QuantileRegressor(quantile=self.tau, alpha=0).fit(
            self.data["x"], self.data["y"]
        )
        return np.insert(fitted.coef_, 0, fitted.intercept_)

    def log_prior(self, theta: Array) -> float:
        return jnp.sum(jax.scipy.stats.norm.logpdf(theta, 0, self.prior_sd))


class QuantRegression3(QuantRegression):
    """Quantile regression from Yang and He 2012, Model 3 of Section 4.2"""

    def __init__(
        self, tau: float, exact_score: bool, prior_sd: float, n_terms: int | None = None
    ):
        n_data = 100
        self.dim_theta = 3
        self.data = self.gen_data(n_data)
        self.prior_sd = prior_sd
        self.n_terms = n_terms if n_terms is not None else 101
        self.tau = tau
        self.exact_score = exact_score

    @staticmethod
    def gen_data(n_data: int) -> PyTree:
        # y = x1 + x2 + (x1/2 + 1)eps
        key1, key2, key3 = jax.random.split(jax.random.key(30), 3)
        x1 = jax.random.chisquare(key1, 2, shape=(100,))
        x2 = 2 * jax.random.bernoulli(key2, p=0.5, shape=(100,))
        eps = 2 * jax.random.normal(key3, shape=(100,))
        # y = x1 + x2 + (x1 / 2 + 1) * eps
        y = x1 + x2 - 3 + (x1 / 2) * eps
        return {"x": jnp.stack([x1, x2], axis=1), "y": y}


class Kyphosis(BayesELModel):
    """Logistic regression with Kyphosis data."""

    prior_sd: float

    def __init__(self, prior_sd: float, n_terms: int | None = None):
        raw_data = pd.read_csv("data/kyphosis.csv")
        self.data = {
            "x": standardize(raw_data[["Age", "Number", "Start"]].to_numpy()),
            "y": raw_data["Kyphosis"].map({"absent": 0, "present": 1}).to_numpy(),
        }

        self.dim_theta = 4
        self.prior_sd = prior_sd
        n_data = utils.get_tree_lead_dim(self.data)
        self.n_terms = n_terms if n_terms is not None else n_data + 1

    def h(self, dp: PyTree, theta: Array) -> Array:
        y = dp["y"]
        x_with_1 = jnp.insert(dp["x"], 0, 1.0)
        chex.assert_equal_shape([x_with_1, theta])
        return x_with_1 * (y - expit(jnp.dot(x_with_1, theta)))

    def init_theta(self, key: KeyArray) -> Array:
        # Use MLE as the initial value of theta
        fitted = LogisticRegression(random_state=0, penalty=None).fit(
            self.data["x"], self.data["y"]
        )
        return np.insert(fitted.coef_[0], 0, fitted.intercept_)

    def log_prior(self, theta: Array) -> float:
        return jnp.sum(jax.scipy.stats.norm.logpdf(theta, 0, self.prior_sd))


class Job(BayesELModel):
    """Job satisfaction (torus) shape.

    :param region_indices: A dictionary of data indices corresponding to each of
        the seven region.
    :param unflatten_theta: Convert a flat 1D parameter vector to a dictionary
        of vectors with meaningful parameter names.
    """

    region_indices: dict[str, list[int]]
    unflatten_theta: Callable[[Array], dict[str, Array]]

    def __init__(self):
        raw_data = pd.read_csv("data/job.csv")
        raw_data["GenderRace"] = raw_data["Gender"] + raw_data["Race"]
        self.region_indices = raw_data.groupby("Region").indices

        # Fixed and random effects (the beta and z term respectively)
        fixed_eff_X = pd.get_dummies(raw_data[["Age", "Gender", "Race", "GenderRace"]])
        fixed_eff_X.insert(0, "Intercept", True)
        random_eff_X = pd.get_dummies(raw_data["Region"])

        self.data = {
            "Satisfied": raw_data["Satisfied"].to_numpy(),
            "Total": raw_data["Total"].to_numpy(),
            "fixed": fixed_eff_X.to_numpy(),
            "random": random_eff_X.to_numpy(),
        }

        # Get a function
        theta = {
            "fixed": jnp.zeros(self.data["fixed"].shape[1]),
            "random": jnp.zeros(self.data["random"].shape[1]),
            "log_tau2": jnp.zeros(()),
        }
        flat_theta, self.unflatten_theta = jax.flatten_util.ravel_pytree(theta)
        self.dim_theta = flat_theta.size
        self.n_terms = None  # This has non-standard EL terms. Need to
        # redefined log_posterior_term

    def h(self, dp: PyTree, theta: Array) -> Array:
        theta = self.unflatten_theta(theta)  # convert to a dict
        log_odds = dp["fixed"] @ theta["fixed"] + dp["random"] @ theta["random"]
        p = expit(log_odds)
        y = dp["Satisfied"]
        n = dp["Total"]

        # Bartlett's constraint
        c1 = y - n * p
        c2 = ((y - n * p) ** 2 / (n * p * (1 - p))) - 1
        return jnp.array([c1, c2])

    def init_theta(self, key: KeyArray) -> Array:
        key, theta_key = jax.random.split(key)
        return jax.random.normal(theta_key, (self.dim_theta,))

    def log_prior(self, theta: Array) -> float:
        theta = self.unflatten_theta(theta)  # convert to a dict
        lp = 0.0

        # fixed effect: N(0, 100^2), random effect: N(0, tau^2)
        lp += norm_raw_log_prob(theta["fixed"], 0, 100)
        lp += norm_raw_log_prob(theta["random"], 0, jnp.exp(0.5 * theta["log_tau2"]))

        # tau^2: IG(0.1, 0.1)
        lp += invgamma_raw_log_prob(jnp.exp(theta["log_tau2"]), 0.1, 0.1)
        lp += theta["log_tau2"]  # change of variable density correction
        return lp

    def log_posterior(
        self, theta: Array, check_sum_to_1: bool = True, use_llog: bool = False
    ) -> float:
        """Small plot estimation.  See the setup in Chaudhuri et al (2017),
        Eq.28.  The weights are estimated separately for each region.
        """
        log_pel = 0.0
        for _, i in self.region_indices.items():
            region_data = jax.tree_map(lambda x: x[i], self.data)
            log_pel += emplik.calc_log_pel(
                self.h,
                region_data,
                theta,
                check_sum_to_1=check_sum_to_1,
                use_llog=use_llog,
            )
        log_prior = self.log_prior(theta)
        return log_pel + log_prior

    def log_posterior_term(self):
        raise NotImplementedError("The log posterior term is not defined.")


class GEE(BayesELModel):
    """
    General estimating equation example from Chang et al 2018, AoS, Section 5.3.
    This is a reduced version.
    """

    true_theta: Array
    prior_sd: float
    M1: Array
    M2: Array

    def __init__(self, prior_sd: float, n_terms: int | None = None):
        n_subjects = 50  # Number of subjects
        n_data = 2  # Number of data points per subject
        rho = 0.7  # Correlation in compound symmetry covariance

        # Define θ0
        # theta0 = np.zeros(p)
        theta0 = np.array([3, 1.5, 0, 0, 2])  # First five elements, rest are zero
        self.true_theta = theta0
        self.dim_theta = theta0.size

        self.M1 = np.eye(n_data)
        # Compound symmetry covariance matrix for epsilon_ij
        self.M2 = rho * np.ones((n_data, n_data)) + (1 - rho) * np.eye(n_data)

        self.data = self.gen_data(n_subjects, n_data, theta0, error_cov=self.M2)
        self.prior_sd = prior_sd
        self.n_terms = n_terms if n_terms is not None else (n_subjects + 1)

    @staticmethod
    def gen_data(n_subjects: int, n_data: int, theta0: Array, error_cov: Array):
        # Construct Sigma for xij
        p = theta0.size  # Dimension of θ0
        indices = np.arange(p)
        row_indices, col_indices = np.meshgrid(indices, indices)
        Sigma = 0.5 ** np.abs(row_indices - col_indices)  # (σkl)p×p

        # Generate xij from N(0, Sigma), Shape: (n, 2, p)
        key1, key2 = jax.random.split(jax.random.key(20), n_data)
        xij = jax.random.multivariate_normal(
            key1, mean=np.zeros(p), cov=Sigma, shape=(n_subjects, n_data)
        )

        # Generate random errors (εi1, εi2)^T from N(0, compound_symmetry), Shape: (n, 2)
        epsilon = jax.random.multivariate_normal(
            key2, mean=np.zeros(n_data), cov=error_cov, shape=(n_subjects,)
        )

        # Compute yij
        yij = np.einsum("ijk,k->ij", xij, theta0) + epsilon
        return {"x": xij, "y": yij}

    def h(self, dp: PyTree, theta: Array) -> Array:
        # Assuming the marginal variance of subject i is 1
        xij = dp["x"]
        yij = dp["y"]
        resid = yij - (xij @ theta)
        h1 = xij.T @ self.M1 @ resid
        h2 = xij.T @ self.M2 @ resid
        h = jnp.concatenate([h1, h2])
        return h

    def init_theta(self, key: KeyArray) -> Array:
        return jnp.asarray(self.true_theta)

    def log_prior(self, theta: Array) -> float:
        return jnp.sum(jax.scipy.stats.norm.logpdf(theta, 0, self.prior_sd))
