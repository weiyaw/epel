import jax
import jax.numpy as jnp
from jax import Array
import chex
import subprocess
import os

import pickle
import logging
import equinox as eqx

from omegaconf import DictConfig


class OptDiagnostic(eqx.Module):
    """
    Diagnostic information for optimization
    """

    steps: int  # total number of steps taken
    converged: bool  # whether the optimization converged
    final_value: Array = None  # final value of the objective function
    path: Array = None  # path from the initial value to the solution


def newton(f, x0, step=1, tol=1e-3, max_iter=32):
    # Newton-Rahpson to find the solution x such that f(x) = 0. Stop at
    # convergence.
    def loop_body(state):
        k, x = state
        df = jax.jacrev(f)(x)  # Jacobian of f wrt x
        delta_x, resid, rank, s = jnp.linalg.lstsq(df, -f(x))
        return (k + 1, x + delta_x)

    def loop_cond(state):
        # Terminate when the norm of f is small enough or when max_iter is
        # reached
        k, x = state
        return jnp.logical_and(jnp.linalg.norm(f(x)) > tol, k < max_iter)

    init_state = (0, x0)
    total_k, final_x = jax.lax.while_loop(loop_cond, loop_body, init_state)

    # We use this optimizer for lambda computation. Specifically, this is used
    # as a solver in jax.lax.custom_root. However, there is a bug that will
    # throw error if we compute the grad of a function that contains
    # custom_root, if its solver returns an aux value that are not float
    # (https://github.com/jax-ml/jax/issues/24295). As a workaround, don't
    # return OptDiagnostic until the bug is fixed.
    # return final_x, OptDiagnostic(total_k, total_k < max_iter, f(final_x))
    return final_x


def newton2(f, x0, step=1, tol=1e-3, max_iter=32):
    # Newton-Rahpson to find the solution x such that f(x) = 0. Run this for a
    # fixed number of iterations
    def loop_body(state, _):
        x, x_best, norm_f_best = state
        df = jax.jacrev(f)(x)  # Jacobian of f wrt x
        delta_x, resid, rank, s = jnp.linalg.lstsq(df, -f(x))
        x_next = x + delta_x

        norm_f_next = jnp.linalg.norm(f(x_next))
        x_best, norm_f_best = jax.lax.cond(
            norm_f_next < norm_f_best,
            lambda: (x_next, norm_f_next),
            lambda: (x_best, norm_f_best),
        )
        return (x_next, x_best, norm_f_best), x_next

    init_state = (x0, x0, jnp.linalg.norm(f(x0)))
    (x_final, x_best, norm_f_best), all_x = jax.lax.scan(
        loop_body, init_state, None, length=max_iter
    )

    is_converged = norm_f_best < tol
    return x_best, OptDiagnostic(max_iter, is_converged, f(x_best), all_x)


def newton_descent(f, x0, step=1, tol=1e-3, max_iter=32):
    # Newton-Rahpson to find the argmin of f(x). Stop at convergence.
    grad_f = jax.grad(f)

    def loop_body(state):
        k, x, x_best, f_best = state
        hess_f = jax.hessian(f)(x)  # Jacobian of f wrt x
        delta_x, resid, rank, s = jnp.linalg.lstsq(hess_f, -grad_f(x))
        x_next = x + delta_x
        x_best, f_best = jax.lax.cond(
            f(x_next) < f_best,
            lambda: (x_next, f(x_next)),
            lambda: (x_best, f_best),
        )
        return (k + 1, x_next, x_best, f_best)

    def loop_cond(state):
        # Terminate when x is no longer moving or when max_iter is reached
        k, x, _, _ = state
        return jnp.logical_and(jnp.linalg.norm(grad_f(x)) > tol, k < max_iter)

    init_state = (0, x0, x0, f(x0))
    total_k, x_final, x_best, f_best = jax.lax.while_loop(
        loop_cond, loop_body, init_state
    )

    return x_best, OptDiagnostic(total_k, total_k < max_iter, f_best, None)


def newton_descent2(f, x0, step=1, tol=1e-3, max_iter=32):
    # Newton-Rahpson to find an argmin of f(x). Run this for a fixed number of
    # iterations
    grad_f = jax.grad(f)

    def loop_body(state, _):
        x, x_best, f_best = state
        hess_f = jax.hessian(f)(x)  # Jacobian of f wrt x
        delta_x, resid, rank, s = jnp.linalg.lstsq(hess_f, -grad_f(x))
        x_next = x + delta_x
        x_best, f_best = jax.lax.cond(
            f(x_next) < f_best,
            lambda: (x_next, f(x_next)),
            lambda: (x_best, f_best),
        )
        return (x_next, x_best, f_best), x_next

    init_state = (x0, x0, f(x0))
    (x_final, x_best, f_best), all_x = jax.lax.scan(
        loop_body, init_state, None, length=max_iter
    )
    is_converged = jnp.linalg.norm(grad_f(x_best)) < tol
    return x_best, OptDiagnostic(max_iter, is_converged, f_best, all_x)


def gradient_descent(f, x0, step=1, tol=1e-3, max_iter=32):
    # Gradient descent to find an argmin of f(x). Stop at convergence.
    def loop_body(state):
        k, x, x_best, f_best = state
        fx, grad_fx = jax.value_and_grad(f)(x)
        x_best, f_best = jax.lax.cond(
            fx < f_best, lambda: (x, fx), lambda: (x_best, f_best)
        )
        x_next = x - step * grad_fx
        return (k + 1, x_next, x_best, f_best)

    def loop_cond(state):
        k, x, _, _ = state
        return jnp.logical_and(jnp.linalg.norm(jax.grad(f)(x)) > tol, k < max_iter)

    init_state = (0, x0, x0, f(x0))
    total_k, x_final, x_best, f_best = jax.lax.while_loop(
        loop_cond, loop_body, init_state
    )
    x_best, f_best = jax.lax.cond(
        f(x_final) < f_best, lambda: (x_final, f(x_final)), lambda: (x_best, f_best)
    )
    return x_best, OptDiagnostic(max_iter, total_k < max_iter, f_best)


def gradient_descent2(f, x0, step=1, tol=1e-3, max_iter=32):
    # Gradient descent to find an argmin of f(x). Run this for a fixed number of
    # iterations.
    def loop_body(state, _):
        x, x_best, f_best = state
        fx, grad_fx = jax.value_and_grad(f)(x)
        x_best, f_best = jax.lax.cond(
            fx < f_best, lambda: (x, fx), lambda: (x_best, f_best)
        )
        x_next = x - step * grad_fx
        return (x_next, x_best, f_best), x_next

    init_state = (x0, x0, f(x0))
    (x_final, x_best, f_best), all_x = jax.lax.scan(
        loop_body, init_state, None, length=max_iter
    )
    x_best, f_best = jax.lax.cond(
        f(x_final) < f_best, lambda: (x_final, f(x_final)), lambda: (x_best, f_best)
    )
    return x_best, OptDiagnostic(max_iter, None, f_best, all_x)


def get_tree_lead_dim(tree):
    """Return the leading dimensions of a PyTree, assuming that all leaves have
    the same leading dimension
    """

    leaves = jax.tree_util.tree_leaves(tree)
    chex.assert_equal_shape_prefix(leaves, 1)
    return leaves[0].shape[0]


def flatten_dict(nested_dict, prefix=""):
    result = {}
    for key, value in nested_dict.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, DictConfig):
            result.update(flatten_dict(value, new_key))
        else:
            result[new_key] = value
    return result


def dict_to_str(d):
    flat_dict = flatten_dict(d)
    return " ".join(f"{k}={v}" for k, v in flat_dict.items())


def print_algo(d):
    algo = list(d.keys())[0]
    res = "algo="
    if algo == "hmc":
        return res + f"hmc n_samples={d['hmc']['n_samples']}"
    else:
        return res + algo


def githash():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def write_to_local(path, obj, verbose=False):
    # write to local
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        logging.info(f"Write to local:/{path}")


def write_to(path, obj, verbose=False):
    write_to_local(path, obj, verbose=verbose)


def read_from_local(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_from(path):
    return read_from_local(path)


def get_matching_dirs(directory: str, regex: str) -> list[str]:
    """
    Find subdirectories in a given directory that match a regex pattern.

    Args:
        directory (str): The directory to search in.
        regex (str): The regular expression pattern to search for in directory names.

    Returns:
        list[str]: A sorted list of full paths to the matching directories.
    """
    matching_dirs = []
    if os.path.exists(directory):
        for entry in os.scandir(directory):
            if entry.is_dir() and re.search(regex, entry.name):
                matching_dirs.append(entry.path)
    matching_dirs.sort()
    return matching_dirs


def get_matching_files(directory: str, regex: str) -> list[str]:
    matching_files = []
    if os.path.exists(directory):
        for entry in os.scandir(directory):
            if entry.is_file() and re.search(regex, entry.name):
                matching_files.append(entry.path)
    matching_files.sort()
    return matching_files