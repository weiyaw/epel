import jax
import jax.numpy as jnp
from jax import vmap

from functools import partial

from typing import Any
from collections.abc import Callable

import utils
import numpy as np

PyTree = Any
Array = jax.Array


def llog(x, eps):
    """
    For x < eps, evaluate with the quadratic expansion of log(x) at eps.
    Otherwise for x > eps, evaluate at the original log(x).
    """
    quad = lambda x: jnp.log(eps) - 1.5 + 2 * x / eps - 0.5 * (x / eps) ** 2
    return jax.lax.cond(x < eps, quad, jnp.log, x)


def dllog(x, eps):
    """
    First order derivative of llog(x).  It is 1/x for x > eps, and 2/eps -
    x/eps^2 otherwise.  In other words, use the linearization of 1/x at eps when
    x < eps.
    """
    dquad = lambda x: (2 - x / eps) / eps
    return jax.lax.cond(x < eps, dquad, lambda x: 1 / x, x)


def calc_log_w(lbd: Array, H: Array, use_llog: bool = False) -> Array:
    """Produce the set of weights that corresponds to the profile empirical
    likelihood, given a set of constraints and lambda.

    :param lbd: a 1D array of lambda that gives the profile empirical
        likelihood.
    :param H: a 2D array of constraints at a theta, where each row is a
              constraint vector at a data point.
    :param use_llog: if True, use llog to prevent log_weights blowing up to -inf
        when the weights are close to 0.

    :return: a vector of weight, with length equal to the number of row in H.
    """
    # n_data = utils.get_tree_lead_dim(data)
    n_data = H.shape[0]

    def calc_log_wi(hi):
        # Output the EL weight for each data point, without scaling with n_data.
        # #
        # hi = h(dp, theta)
        if use_llog:
            log_unscaled_wi = llog(1 / (jnp.dot(lbd, hi) + 1), 1 / n_data)
        else:
            log_unscaled_wi = -jnp.log(jnp.dot(lbd, hi) + 1)

        # Sum of unscaled_wi over i equals to n_data.
        return -jnp.log(n_data) + log_unscaled_wi

    return vmap(calc_log_wi)(H)


def calc_log_pel(
    h: Callable[[dict[str, Array], PyTree], Array],
    data: dict[str, Array],
    theta: PyTree,
    tol: float = 1e-7,
    max_iter: int = 200,
    check_sum_to_1: bool = True,
    use_llog: bool = False,
    return_opt_state: bool = False,
) -> float | tuple[float, Any]:
    """Compute the log of profile empirical likelihood.

    This will compute the log of profile empirical likelihood, max(sum_i log
    w_i), where the weight w_i is subject to constraints, sum_i w_i h(x_i,
    theta) = 0, on the data x.

    The gradient of this function wrt theta can be sped up by manually defining
    the gradient, because of grad(lbd)_theta * sum_i{w_i h(x_i, theta)} = 0, so
    so we don't need to compute grad(lbd)_theta.

    :param h: a Callable with a signature h(x, theta), where x is an input data
              point, and produce a 1D array.  This is the constraint function,
              i.e. sum_i w_i h(x_i, theta) = 0 for all x in data.
    :param data: a Pytree of array of data, where each entry in the leading
        dimension is the input of h.
    :param theta: a 1D array of parameters of interest.  This must be compatible
        with data and h, such that h will produce a 1D array.
    :param tol: tolerance for the convergence of the Newton-Raphson.  1e-7 is
        roughly the limit for float32.
    :param max_iter: maximum number of iterations.
    :param check_sum_to_1: if True, check if the sum of weights equals to 1.  If
        False, then the sum of weights will not be checked.
    :param use_llog: if True, use llog to prevent log_weights blowing up to -inf
        when the weights are close to 0.
    :param return_opt_state: if True, return the optimization state when solving
        for lambda.

    :return: a scalar value of the logarithm of profile empical likelihood under
             the given theta, constraint h and data.  If return_opt_state is
             True, then also return a tuple with the value in the first element
             and the optimization state as the second element.
    """

    H = vmap(lambda x: h(x, theta))(data)

    # DO NOT USE stop_gradient(calc_lambda), because the second-order partial
    # derivative will be wrong.
    #
    # lbd, opt_state = calc_lambda(H, tol, max_iter)
    # We are not returning opt_state, due to a bug in jax.lax.custom_root. See
    # https://github.com/jax-ml/jax/issues/24295
    lbd = calc_lambda(H, tol, max_iter)
    opt_state = None
    all_log_w = calc_log_w(lbd, H, use_llog=use_llog)

    if check_sum_to_1:
        # When theta is outside of the support, then the sum of weights will not
        # equal to 1. In this case we set log_pel to -inf, replicating the
        # behaviour in elhmc where they set theta outside of the support to be
        # some very large value
        log_sum_w = jax.scipy.special.logsumexp(all_log_w)
        sum_to_1 = jnp.allclose(log_sum_w, jnp.zeros(()), atol=1e-3)
        log_pel = jax.lax.select(sum_to_1, jnp.sum(all_log_w, axis=0), -np.inf)
    else:
        # If theta falls outside of the boundary, this will result in a
        # reasonably large, but not -inf value.
        log_pel = jnp.sum(all_log_w, axis=0)

    return (log_pel, opt_state) if return_opt_state else log_pel


def calc_log_ael(
    h: Callable[[dict[str, Array], PyTree], Array],
    data: dict[str, Array],
    theta: PyTree,
    a_n: float,
    tol: float = 1e-7,
    max_iter: int = 200,
    check_sum_to_1: bool = True,
    use_llog: bool = False,
    return_opt_state: bool = False,
) -> float | tuple[float, Any]:
    """Compute the log of adjusted empirical likelihood.

    a_n should be the order of 1 / n_data for nice theoretical properties.

    """

    # Add the constraint for the pseudo data point
    H = vmap(lambda x: h(x, theta))(data)
    assert H.ndim == 2  # H should be 2D
    h_pseudo = -a_n * jnp.mean(H, axis=0)
    H = jnp.vstack([H, h_pseudo])

    # lbd, opt_state = calc_lambda(H, tol, max_iter)
    # We are not returning opt_state, due to a bug in jax.lax.custom_root. See
    # https://github.com/jax-ml/jax/issues/24295
    lbd = calc_lambda(H, tol, max_iter)
    opt_state = None

    all_log_w = calc_log_w(lbd, H, use_llog=use_llog)

    if check_sum_to_1:
        # Check if lbd converges to the solution
        log_sum_w = jax.scipy.special.logsumexp(all_log_w)
        sum_to_1 = jnp.allclose(log_sum_w, jnp.zeros(()), atol=1e-3)
        log_ael = jax.lax.select(sum_to_1, jnp.sum(all_log_w, axis=0), -np.inf)
    else:
        log_ael = jnp.sum(all_log_w, axis=0)

    return (log_ael, opt_state) if return_opt_state else log_ael


def calc_lambda(H: Array, tol: float = 1e-7, max_iter: int = 50) -> tuple[Array, Any]:
    # Constraint matrix with shape n_data x n_constraints
    """Maximize the empirical likelihood, given a set of constraints.

    Estimate the lambda that maximizes the empirical likelihood using
    Newton-Raphson, given a set of constraints.

    :param H: a 2D array of constraints evaulated at a theta on a dataset.  Each
              row is a constraint vector at a data point.
    :param tol: tolerance for the convergence of the Newton-Raphson.  1e-7 is
        roughly the limit for float32.
    :param max_iter: maximum number of iterations.

    :return: a tuple with two elements: the vector of lambda that maximizes the
             empirical likelihood, and the diagnostic information from the
             optimizer.
    """

    tangent_solver = lambda g, y: jnp.linalg.solve(jax.jacrev(g)(y), y)

    assert H.ndim == 2  # H should be 2D
    n_data = H.shape[0]

    def dlogEL_ratio(lbd):
        # The gradient of llogEL_ratio. The weight has been transformed by llog
        # to stabalize the computation of very small dot(lbd, hi) + 1. The root
        # of this gradient maximizes EL. This is same as the implementation in
        # the emplik R package, more specifically the computation of wts1 in the
        # el.test routine. See also Section 2, Tang and Wu (2014).
        dlogEL_ratio_per_data = lambda h: h * dllog(jnp.dot(lbd, h) + 1, 1 / n_data)
        return jnp.sum(vmap(dlogEL_ratio_per_data)(H), axis=0)

    lbd0 = jnp.zeros((H.shape[1],))

    # The root of grad(logEL_ratio) is the lambda
    return jax.lax.custom_root(
        dlogEL_ratio,
        lbd0,
        utils.newton,
        tangent_solver,
    )


# @partial(jax.custom_jvp, nondiff_argnums=(0, 1))
# def calc_lambda2(g, data, theta, tol=1e-7, max_iter=200):
#     # Constraint matrix with shape n_data x n_constraints
#     G = vmap(lambda x: g(x, theta))(data)
#     assert len(G.shape) == 2  # G should be 2D

#     lbd_0 = jnp.zeros((G.shape[1],))

#     def F(lbd):
#         # This is the loss function.
#         # F_per_data = lambda g: g / (jnp.dot(lbd, g) + n_data)
#         F_per_data = lambda g: g / (jnp.dot(lbd, g) + 1)
#         return jnp.sum(vmap(F_per_data)(G), axis=0)

#     def loop_body(state):
#         # Newton-Raphson to find the root of F
#         k, lbd = state

#         # Jacobian of F wrt lambda
#         dF = jax.jacrev(F)(lbd)
#         # dF = grad_F(lbd)  # or
#         chex.assert_shape(dF, (lbd.size, lbd.size))  # has to be square by definition

#         # if G.ndim == 1:
#         if dF.shape == ():
#             delta_lbd = -F(lbd) / dF
#         else:
#             delta_lbd, resid, rank, s = jnp.linalg.lstsq(dF, -F(lbd))

#         return (k + 1, lbd + delta_lbd)

#     def loop_cond(state):
#         # Terminate when the norm of F is small enough or when max_iter is
#         # reached
#         k, lbd = state
#         return jnp.logical_and(jnp.linalg.norm(F(lbd)) > tol, k < max_iter)

#     init_state = (0, lbd_0)
#     total_k, lbd_k = jax.lax.while_loop(loop_cond, loop_body, init_state)
#     # jax.debug.print("k: {}, lbd: {}, -F: {}", total_k, lbd_k, -F(lbd_k))
#     return lbd_k


# @calc_lambda2.defjvp
# def calc_lambda2_jvp(g, data, primals, tangents):
#     """
#     This is the JVP of lambda wrt theta, derived manually with implicit
#     differentiation.
#     """
#     (theta,) = primals  # this is where the Jacobian is evaluated
#     (theta_dot,) = tangents  # this is the v in JVP

#     lbd = calc_lambda(g, data, theta)

#     G = vmap(lambda x: g(x, theta))(data)  # constraint matrix (N x m)

#     N = G.shape[0]
#     m = 1 if G.ndim == 1 else G.shape[1]
#     d = 1 if type(theta) is float else theta.size

#     calc_wi = lambda dp: jnp.exp(calc_log_wi(N, lbd, g, dp, theta))
#     W = vmap(calc_wi)(data)  # weight of each data point
#     chex.assert_shape(W, (N,))

#     grad_g = jax.jacrev(g, argnums=1)  # because the jacobian most likely be wide

#     # A x = b, where x is the JVP
#     # xi: the i^th data point
#     # wi: the i^th weight
#     # gi: the i^th contraint vector

#     # LHS of the linear equation (per data point)
#     lhs = lambda wi, gi: (wi**2) * jnp.outer(gi, gi)
#     LHS = jnp.sum(vmap(lhs)(W, G), axis=0)
#     chex.assert_shape(LHS, (m, m))

#     # RHS of the linear equation (per data point)
#     def rhs(xi, wi, gi):
#         p1 = wi * (jnp.eye(m) / N - wi * jnp.outer(lbd, gi))
#         p2 = grad_g(xi, theta)
#         chex.assert_shape(p1, (m, m))
#         chex.assert_shape(p2, (m, d))
#         return p1 @ p2 @ theta_dot

#     RHS = jnp.sum(vmap(rhs)(data, W, G), axis=0)
#     chex.assert_shape(theta_dot, (d,))
#     chex.assert_shape(RHS, (m,))

#     # Solve the linear equation
#     jvp_out = jnp.linalg.solve(LHS, RHS)
#     return lbd, jvp_out
