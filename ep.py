import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.linalg import qr, cho_solve
from jax import Array

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Self

import chex
import blackjax
import numpy as np
import equinox as eqx

from functools import partial
from tqdm import tqdm
import logging

import utils
from timeit import default_timer as timer


Covariance = Array  # full covariance or diagonal
Precision = Array  # full covariance or diagonal
KeyArray = Any  # jax PRNG key


"""
Expectation propagation
"""


class Gaussian(eqx.Module):
    eta: Array
    Lambda: Array

    @abstractmethod
    def __init__(
        self,
        mu: Array | None = None,
        eta: Array | None = None,
        Sigma: Array | None = None,
        Lambda: Array | None = None,
    ) -> None:
        pass

    @property
    @abstractmethod
    def mu(self) -> Array:
        pass

    @property
    @abstractmethod
    def Sigma(self) -> Covariance:
        pass

    @classmethod
    @abstractmethod
    def from_products(cls, qs: list[Self]) -> Self:
        pass

    @classmethod
    @abstractmethod
    def from_quotient(cls, q1: Self, q2: Self) -> Self:
        pass

    @classmethod
    @abstractmethod
    def from_power(cls, q: Self, power: float) -> Self:
        pass

    @classmethod
    @abstractmethod
    def symmetrify_cov(cls, q: Self) -> Self:
        pass

    @abstractmethod
    def sample(self, key: KeyArray, n_samples=None) -> Array:
        pass

    @abstractmethod
    def log_prob(self, x: Array, raw: bool = True) -> Array:
        # return unnormalized log density if raw is true.
        pass

    @classmethod
    def check_types_compatible(cls, qs: list[Self]) -> None:
        type_checks = [type(q) is cls for q in qs]
        if not all(type_checks):
            raise TypeError(
                f"All elements of `qs` must be of type " f"{cls}, but got {type_checks}"
            )

    @abstractmethod
    def _get_string_form(self) -> str:
        pass

    def __repr__(self) -> str:
        return self._get_string_form()

    def __str__(self) -> str:
        return self._get_string_form()


class FullGaussian(Gaussian):
    """Gaussian distribution

    One parameterization:
        mu, Sigma
    Another parameterization (preferred):
        eta, Lambda
    """

    def __init__(
        self,
        mu: Array | None = None,
        eta: Array | None = None,
        Sigma: Array | None = None,
        Lambda: Array | None = None,
    ) -> None:

        # only one pair is allowed
        if (mu is None) == (eta is None):
            raise ValueError
        if (Sigma is None) == (Lambda is None):
            raise ValueError

        self.Lambda = Lambda
        if self.Lambda is None:
            self.Lambda = jnp.linalg.inv(Sigma)

        self.eta = eta
        if self.eta is None:
            self.eta = self.Lambda @ mu

        # Check the dimension of Lambda and eta
        d = self.eta.shape[0]
        checks = [
            (self.eta.ndim == 1),
            (self.Lambda.ndim == 2),
            (self.Lambda.shape == (d, d)),
        ]
        if not all(checks):
            raise ValueError(checks)

    @property
    def mu(self) -> Array:
        return jnp.linalg.inv(self.Lambda) @ self.eta

    @property
    def Sigma(self) -> Array:
        return jnp.linalg.inv(self.Lambda)

    @classmethod
    def from_products(cls, qs: list[Self]) -> Self:
        cls.check_types_compatible(qs)
        new_eta = qs[0].eta
        new_Lambda = qs[0].Lambda
        for q in qs[1:]:
            new_eta = new_eta + q.eta
            new_Lambda = new_Lambda + q.Lambda

        return cls(eta=new_eta, Lambda=new_Lambda)

    @classmethod
    def from_quotient(cls, q1: Self, q2: Self) -> Self:
        cls.check_types_compatible([q1, q2])
        return cls(eta=q1.eta - q2.eta, Lambda=q1.Lambda - q2.Lambda)

    @classmethod
    def from_power(cls, q: Self, power: float) -> Self:
        cls.check_types_compatible([q])
        return cls(eta=power * q.eta, Lambda=power * q.Lambda)

    @classmethod
    def symmetrify_cov(cls, q: Self) -> Self:
        return cls(eta=q.eta, Lambda=0.5 * (q.Lambda + q.Lambda.T))

    def log_prob(self, x: Array, raw: bool = True) -> Array:
        """Compute log probability of a Gaussian distribution, up to some
        constants that do not depend on the `x`.
        """
        # When `x` and `eta` are vectors, the transposes have no
        # effect, but they are still correct, so we will leave them here.
        return self.eta.T @ x - 0.5 * x.T @ self.Lambda @ x

    def sample(self, key: KeyArray, n_samples=None) -> Array:
        shape = None if n_samples is None else (n_samples,)
        return jax.random.multivariate_normal(key, self.mu, self.Sigma, shape=shape)

    def _get_string_form(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(\u03B7={self.eta.shape}, \u039B={self.Lambda.shape})"
        )


class MultivariateT(eqx.Module):
    """
    Multivariate-t distribution.  See
    https://en.wikipedia.org/wiki/Multivariate_t-distribution
    """

    centre: Array  # the location parameter
    scale_inverse: Array  # inverse of the scale matrix
    C: Array  # (lower triangular) cholesky factor of the scale matrix
    df: int  # degree of freedom

    def __init__(self, centre: Array, scale_inverse: Array, df: int) -> None:
        self.centre = centre
        d = centre.size
        self.scale_inverse = scale_inverse
        # chol (upper triangular) of the inverse scale matrix
        U = jnp.linalg.cholesky(scale_inverse)
        # chol of the scale matrix
        self.C = jax.scipy.linalg.solve_triangular(U, jnp.eye(d))
        self.df = df

    @property
    def scale(self) -> Array:
        # scale matrix
        return self.C @ self.C.T

    def sample(self, key: KeyArray, n_samples=None) -> Array:
        key1, key2 = jax.random.split(key)
        d = self.centre.size
        normal_shape = (d,) if n_samples is None else (n_samples, d)
        normal_rv = jax.random.normal(key1, normal_shape)
        chi2_shape = None if n_samples is None else (n_samples,)
        chi2_rv = jax.random.chisquare(key2, self.df, chi2_shape)

        denominator = jnp.sqrt(chi2_rv / self.df)
        res = self.centre + (self.C @ normal_rv.T / denominator).T
        return res

    def log_prob(self, x: Array, raw: bool = True) -> Array:
        # return unnormalized log density if raw is true.
        if not raw:
            raise NotImplementedError
        d = self.centre.size
        quadratic = (x - self.centre).T @ self.scale_inverse @ (x - self.centre)
        res = -0.5 * (self.df + d) * jnp.log1p(quadratic / self.df)
        return res


def get_initial_site_gs(global_mu, global_Sigma, n_sites):
    # In: global mu and Sigma. Return: a list of FullGaussian objects. The
    # product of these objects is a Gaussian(global_mu, global_Sigma).

    site_Lambda = jnp.linalg.inv(global_Sigma) / n_sites
    site_eta = site_Lambda @ global_mu
    return [FullGaussian(eta=site_eta, Lambda=site_Lambda) for _ in range(n_sites)]


def get_cavity_dist(global_dist, site_dist):
    # In: global beta, site beta. Return: beta of cavity

    cavity_dist = global_dist.from_quotient(global_dist, site_dist)
    return cavity_dist


def is_converge(site_deltas, tol=1e-8):
    # Stop when the l2 norm of etas are moving less than tol
    return all([np.linalg.norm(delta.eta, ord=2) < tol for delta in site_deltas])


def is_pd(matrix):
    try:
        return np.all(np.linalg.eigvalsh(matrix) > 0)
    except np.linalg.LinAlgError:
        return False


def adjust_damping_factor(site_deltas, global_g, damping_factor):
    """
    Reduce the damping factor to keep the precision of global_g
    positive-definite after updating with delta.
    """
    class_g = global_g.__class__
    i = 0
    while i < 50:
        delta = class_g.from_products(site_deltas)
        adjusted_damping_factor = damping_factor * (0.9**i)
        damped_delta = class_g.from_power(delta, adjusted_damping_factor)
        candidate_g = class_g.from_products([global_g, damped_delta])
        if is_pd(candidate_g.Lambda):
            logging.info(f"Proportion of the damping factor: {0.9**i:.2f}")
            break
        i += 1
    return adjusted_damping_factor


def laplace_approximation(
    target: Callable,
    init_x0: Array,
    minimizer: Callable,
    **minimizer_kwargs,
) -> tuple[Gaussian, utils.OptDiagnostic]:
    """
    Laplace Gaussian approximation of the target density.

    :param target: The log of the target density.
    :param init_x0: The initial value for the minimizer.
    :param minimizer: A minimizer algorithm from utils.py.
    :param max_iter: The maximum number of iterations, to be passed to the miminizer.
    :param tol: The tolerance, to be passed to the miminizer.
    """

    loss = lambda z: -target(z)
    mu_hat, opt_diagnostics = minimizer(loss, init_x0, **minimizer_kwargs)

    # Compute the negative Hessian of the log target at mu_hat
    Lambda_hat = jax.hessian(loss)(mu_hat)
    Lambda_hat = (Lambda_hat + Lambda_hat.T) / 2  # ensure it is symmetry
    eta_hat = Lambda_hat @ mu_hat

    return FullGaussian(eta=eta_hat, Lambda=Lambda_hat), opt_diagnostics


def precision_from_scatter(scatter: Array, df: float) -> Array:
    # A stable way to compute the precision matrix. See A.3 of Vehtari 2019.
    p = scatter.shape[1]
    _, upper_tri = qr(scatter, mode="economic")
    chex.assert_shape(upper_tri, (p, p))
    precision = cho_solve((upper_tri, False), jnp.eye(p)) * df
    return precision


class ISDiagnostic(eqx.Module):
    """
    Diagnostic information for the importance sampling
    """

    ess: float  # effective sample size
    converged: bool  # whether the samples are good enough (e.g., larger than some ess threshold)
    w: Array  # weights of each samples


def snis(
    target: Callable,
    key: KeyArray,
    proposal: Gaussian,
    n_samples: int,
    threshold: float = 0.5,
) -> tuple[Gaussian, ISDiagnostic]:
    """
    A Gaussian approximation of the target, with mean and covariance computed
    with self-normalized importance sampling.

    :param proposal: A Gaussian object representing the proposal distribution.
    :param target: A callable that takes in a z and returns the unnormalized log
        density of the target distribution.
    :param n_samples: The number of samples to draw.
    :param key: A random key.
    :param threshold: A number between [0, 1] to set the threshold, so that when
        ess > threshold * n_samples the sampler is counted as "converged".

    :return: A Gaussian object and diagnostics.
    """
    # Compute samples
    sample_key, key = jax.random.split(key)
    samples = proposal.sample(sample_key, n_samples)

    # Compute unnormalized importance weights
    log_w = jax.vmap(lambda s: target(s) - proposal.log_prob(s, raw=True))(samples)
    log_w = log_w - jnp.mean(log_w)  # center the weights to prevent overflow
    w = jnp.exp(log_w)
    sum_w = jnp.exp(jax.nn.logsumexp(log_w))

    mu = jnp.sum(w[:, None] * samples, axis=0) / sum_w

    weighted_scatter = (samples - mu) * jnp.sqrt(w)[:, None]
    tilde_Lambda = precision_from_scatter(weighted_scatter, sum_w)
    tilde_Lambda = (tilde_Lambda + tilde_Lambda.T) / 2  # ensure it's symmetry
    tilde_eta = tilde_Lambda @ mu

    # ESS computed from 9.13 of Section 9.3, Monte Carlo book from Art Owen.
    # This is not the only definition. A poor ESS implies poor samples, but a
    # good ESS does not guarantee good samples.
    ess = n_samples * jnp.mean(w) ** 2 / jnp.mean(w**2)

    # Define convergence as having ess of at least 90% of the actual sample size
    diagnostics = ISDiagnostic(ess=ess, converged=ess > threshold * n_samples, w=w)
    return FullGaussian(eta=tilde_eta, Lambda=tilde_Lambda), diagnostics


@eqx.filter_jit
def get_kl_projection_mcmc(
    key: KeyArray,
    cavity_dist: Gaussian,
    idx: int,
    f_term: Callable[[Array, int], float],
    method: str = "nuts",
    n_samples: int = 1000,
    debug: bool = False,
) -> tuple[Gaussian, Any]:
    """KL projection of tilted dist to a FullGaussian through moment matching.

    Project the tilted distribution at the k_th site to a FullGaussian using the
    moment matching.  Tilted distribution is defined as

    g_{\k}(z) \propto g_{-k}(z) * f_k(z)

    where g_{-k}(z) is the cavity distribution, and f_k(z) is the raw
    (unnormalised) density.

    :param key: A random key.
    :param cavity_dist: g_{-k}(z), a Gaussian object specifying the cavity
        distribution.
    :param idx: The index of the site.
    :param f_term: A callable that takes in a z and an index i, and returns the
        unnormalized log density of the i-th term in the target density.
    :param method: The method to use for the projection.  Either "laplace",
        "nuts", "rmh" MH-algorithm with an independent proposal.
    :param n_samples: The number of samples to draw.
    :param debug: Whether to run in debug mode.

    :return: a tuple of the a FullGaussian object representing the KL projection
             of the tilted distribution, and whatever diagnostics that are
             relevant.
    """
    site_f = partial(f_term, idx=idx)  # density at the idx-th site
    log_tilted = lambda z: cavity_dist.log_prob(z, raw=True) + site_f(z)

    if method == "nuts" or method == "rmh":
        # exact = cavity_dist.from_products([cavity_dist, site_f])

        # Assuming cavity is Gaussian
        assert isinstance(cavity_dist, Gaussian)

        if method == "rmh":
            # MH with fixed proposal
            def proposal_density(prev_z, next_z):
                # conditional proposal density, conditioned on prev_z
                return cavity_dist.raw_logpdf(next_z.position)

            mcmc = blackjax.irmh(
                log_tilted, cavity_dist.sample, proposal_density
            )  # using cavity as proposal
            key, subkey = jax.random.split(key)
            init_state = (subkey, mcmc.init(cavity_dist.mu))  # start from cavity mean
        else:
            # NUTS adaptation
            adapt = blackjax.window_adaptation(
                blackjax.nuts, log_tilted, initial_step_size=1e-2
            )
            key, subkey = jax.random.split(key)
            adapt_res, _ = adapt.run(subkey, cavity_dist.mu, num_steps=500)

            # NUTS sampling
            mcmc = blackjax.nuts(log_tilted, **adapt_res.parameters)
            key, subkey = jax.random.split(key)
            init_state = (subkey, adapt_res.state)  # start from adapted state

        def mcmc_step(state, _):
            key, mcmc_state = state
            key, subkey = jax.random.split(key)
            next_mcmc_state, mcmc_info = mcmc.step(subkey, mcmc_state)
            return (key, next_mcmc_state), (
                mcmc_info.acceptance_rate,
                next_mcmc_state.position,
            )

        _, (accept_p, samples) = jax.lax.scan(
            mcmc_step, init_state, None, length=n_samples
        )

        mu = jnp.mean(samples, axis=0)
        scatter = samples - mu
        tilde_Lambda = precision_from_scatter(scatter, n_samples - 1)
        tilde_eta = tilde_Lambda @ mu

        mcmc_diagnostics = {
            "accept_p": jnp.mean(accept_p),
            "ess": blackjax.diagnostics.effective_sample_size(
                jnp.expand_dims(samples, 0)
            ),
        }
        return FullGaussian(eta=tilde_eta, Lambda=tilde_Lambda), mcmc_diagnostics


def expectation_propagation(
    key: KeyArray,
    f_term: Callable[[Array, int], float],
    n_terms: int,
    init_g: tuple[Array, Covariance],
    laplace_iter: int,
    snis_samples: int,
    damping_factor: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 50,
    verbose: bool = False,
):
    """
    Expectation propagation, projecting to a Gaussian object.  It approximates a
    target density that can be factorized into many sub-components.

    :param key: A random key.
    :param f_term: A callable that takes in a z and an index i, and returns the
        unnormalized log density of the i-th term in the target density.
    :param n_terms: The number of sites.  This should be consistent with the
        number of terms available in f_term.
    :param init_g: An initial guess of the mean and covariance of the
        approximate density.
    :param laplace_iter: The number of starting iterations when the tilted is
        approximated by Laplace only and no importance sampling.
    :param snis_samples: The number of samples used in importance sampling.
    :param damping_factor: The damping factor for the site deltas.
    :param tol: The tolerance for EP convergence.
    :param max_iter: The maximum number of EP iterations.
    :param verbose: Whether to print out the progress.

    :return: A tuple of output.  The first element is the global Gaussian
             approximation of the target density.  The second element is a list
             of Gaussian objects, the site approximations.  The third element is
             the number of iterations.  The fourth element is the wall-clock
             time of each iteration.
    """

    assert init_g[0].ndim == 1

    site_gs = get_initial_site_gs(init_g[0], init_g[1], n_terms)
    assert all(isinstance(g, type(site_gs[0])) for g in site_gs)
    class_g = site_gs[0].__class__

    time_ls = [0.0]  # wall-clock time of each iteration

    # Initial gs at standard Gaussians
    global_g = class_g.from_products(site_gs)
    global_g_ls = [global_g]  # global Gaussian at each iteration

    iteration = 0

    @jit
    def jit_snis(idx, cavity, proposal, key):
        log_tilted = lambda z: cavity.log_prob(z, raw=True) + f_term(z, idx)
        return snis(log_tilted, key, proposal, snis_samples)

    @jit
    def jit_laplace(idx, cavity, init_x0):
        log_tilted = lambda z: cavity.log_prob(z, raw=True) + f_term(z, idx)
        return laplace_approximation(
            log_tilted, init_x0, utils.newton_descent, max_iter=1000, tol=1e-10
        )

    jit_snis = jit_snis.lower(0, global_g, global_g, key).compile()
    jit_laplace = jit_laplace.lower(0, global_g, global_g.mu).compile()

    while iteration < max_iter:
        logging.info(f"-------------------- iter: {iteration} --------------------")
        logging.info(
            f"Eigenvalues of global Lambda: {np.linalg.eigvals(global_g.Lambda)}"
        )
        if verbose:
            logging.info(f"Global eta: {global_g.eta}")
            logging.info(f"Global Lambda: \n {global_g.Lambda}")
            logging.info(f"Global mu: {global_g.mu}")
            logging.info(f"Global Sigma: \n {global_g.Sigma}")

        start = timer()
        next_site_gs = []
        site_deltas = []

        diverge = []
        for k, site_g in enumerate(site_gs):
            cavity = get_cavity_dist(global_g, site_g)
            init_x0 = cavity.mu
            tilde_g, diagnostics = jit_laplace(k, cavity, init_x0)
            tilde_g_pd = is_pd(tilde_g.Lambda)
            if iteration >= laplace_iter or not tilde_g_pd or not diagnostics.converged:
                if tilde_g_pd:
                    proposal = tilde_g
                elif is_pd(cavity.Lambda):
                    logging.info(f"cavity as SNIS proposal at site {k}")
                    proposal = cavity
                else:
                    logging.info(f"global_g as SNIS proposal at site {k}")
                    proposal = global_g
                key, subkey = jax.random.split(key)
                tilde_g, diagnostics = jit_snis(k, cavity, proposal, subkey)

            if verbose:
                logging.info(f"Tilde eta: {tilde_g.eta}")
                logging.info(f"Tilde Lambda: \n {tilde_g.Lambda}")

            if np.all(np.isfinite(tilde_g.Lambda)):
                site_delta = class_g.from_quotient(tilde_g, global_g)
            else:
                # This could happen when one of the w in snis dominates, and end
                # up with unstable tilde_g calculation
                logging.info(f"kl projection of tilde_g is infite at site {k}")
                site_delta = class_g.from_quotient(global_g, global_g)
            site_deltas.append(site_delta)
            if not diagnostics.converged:
                if hasattr(diagnostics, "ess"):
                    ess = diagnostics.ess
                    logging.info(
                        f"ESS at site {k}: {ess:.4f}, {100 * ess / snis_samples:.2f}%"
                    )
                diverge.append(k)

        if len(diverge) > 0:
            logging.info(f"Diverge: {diverge}")
        adjusted_damping_factor = adjust_damping_factor(
            site_deltas, global_g, damping_factor
        )

        for site_g, site_delta in zip(site_gs, site_deltas):
            damped_site_delta = class_g.from_power(site_delta, adjusted_damping_factor)
            next_site_g = class_g.from_products([site_g, damped_site_delta])
            next_site_gs.append(next_site_g)

        global_g = class_g.from_products(next_site_gs)
        global_g = class_g.symmetrify_cov(global_g)
        jax.block_until_ready(global_g)
        time_ls.append(timer() - start)
        site_gs = next_site_gs
        iteration += 1
        global_g_ls.append(global_g)  # track global_g across iterations

        if is_converge(site_deltas, tol=tol):
            break

    return global_g_ls, site_gs, iteration, time_ls


if __name__ == "__main__":
    # A testing run of EP with a product of 6 Gaussians as the target distribution
    dim_z = 2

    site_fs = [
        FullGaussian(mu=jnp.zeros((dim_z,)), Sigma=jnp.eye(dim_z)),
        FullGaussian(mu=0.5 * jnp.ones((dim_z,)), Sigma=0.8 * jnp.eye(dim_z)),
        FullGaussian(mu=jnp.ones((dim_z,)), Sigma=1.2 * jnp.eye(dim_z)),
        FullGaussian(mu=jnp.zeros((dim_z,)), Sigma=jnp.eye(dim_z)),
        FullGaussian(mu=-1 * jnp.ones((dim_z,)), Sigma=0.8 * jnp.eye(dim_z)),
        FullGaussian(mu=-1 * jnp.ones((dim_z,)), Sigma=1.1 * jnp.eye(dim_z)),
    ]
    truth_f = FullGaussian.from_products(site_fs)
    print(f"truth eta: {truth_f.eta}")
    print(f"truth Lambda: \n {truth_f.Lambda}")

    site_gs = get_initial_site_gs(jnp.array([0.0, 0.0]), jnp.eye(dim_z), len(site_fs))

    key = jax.random.key(42398)

    def f_term(x, idx):
        return site_fs[idx].log_prob(x, raw=True)

    global_g_ls, _, _, _ = expectation_propagation(key, f_term, 6, jnp.zeros((dim_z,)))
    print(f"final eta: {global_g_ls[-1].eta}")
    print(f"final Lambda: \n {global_g_ls[-1].Lambda}")
