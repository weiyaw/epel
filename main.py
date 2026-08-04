# %%
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

import hydra
import equinox as eqx
from omegaconf import DictConfig, OmegaConf
from functools import partial
import logging
import os
import optax
from tqdm import tqdm


import model
import utils
import vi
from mcmc import hmc, nuts, random_walk
import ep

np.set_printoptions(linewidth=np.inf, formatter={"float": "{:.4g}".format})

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_default_matmul_precision", "float32")  # control for 32bit input
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))
    outdir = os.path.relpath(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    # outdir = "./outputs/debug"

    if "rw" in cfg.algorithm:
        assert (
            len(cfg.algorithm) == 1
        ), "Random walk does not compute the same posterior as other algorithms. Run it seperately."

    # Use exact score for rw since it can handle non-differentiable h
    bayes_model = model.build_model(
        cfg.model,
        cfg.prior_sd,
        n_points=cfg.algorithm.ep.n_points if "ep" in cfg.algorithm else 1,
        exact_score="rw" in cfg.algorithm,
    )

    logging.info(f"dim of theta: {bayes_model.dim_theta}")
    n_data = utils.get_tree_lead_dim(bayes_model.data)
    logging.info(f"data size: {n_data}")
    logging.info(
        f"Initial theta: {bayes_model.init_theta(jax.random.PRNGKey(cfg.seed))}"
    )

    key = jax.random.PRNGKey(cfg.seed)
    if any(algo in ["nuts", "hmc", "rw"] for algo in cfg.algorithm):
        log_posterior = partial(
            bayes_model.log_posterior, check_sum_to_1=True, use_llog=False
        )

        key, init_key = jax.random.split(key)
        initial_theta = bayes_model.init_theta(init_key)
        logging.info(initial_theta)

        if "nuts" in cfg.algorithm:
            cfg_algo = cfg.algorithm.nuts
            key, subkey = jax.random.split(key)
            mcmc_state, mcmc_info, time_ls = nuts(
                subkey,
                log_posterior,
                initial_theta,
                n_samples=cfg_algo.n_samples,
                n_warmup=cfg_algo.n_warmup,
                initial_step_size=cfg_algo.initial_step_size,
            )
        if "hmc" in cfg.algorithm:
            cfg_algo = cfg.algorithm.hmc
            key, subkey = jax.random.split(key)
            mcmc_state, mcmc_info, time_ls = hmc(
                subkey,
                log_posterior,
                initial_theta,
                n_samples=cfg_algo.n_samples,
                n_warmup=cfg_algo.n_warmup,
                step_size=cfg_algo.step_size,
                inverse_mass_matrix=jnp.ones((bayes_model.dim_theta,)),
                num_integration_steps=cfg_algo.num_integration_steps,
            )
        if "rw" in cfg.algorithm:
            cfg_algo = cfg.algorithm.rw
            key, subkey = jax.random.split(key)
            mcmc_state, mcmc_info, time_ls = random_walk(
                subkey,
                log_posterior,
                initial_theta,
                sd_shrink_factor=cfg_algo.sd_shrink_factor,
                n_samples=cfg_algo.n_samples,
                n_warmup=cfg_algo.n_warmup,
                thinning=cfg_algo.thinning,
            )

        utils.write_to(
            f"{outdir}/output.pickle",
            {"state": mcmc_state, "info": mcmc_info, "time": time_ls},
            verbose=True,
        )
        # save as pure numpy array and built-in structure, to be readable by reticulate
        utils.write_to(
            f"{outdir}/pure.pickle",
            {"samples": np.asarray(mcmc_state.position), "time": time_ls},
        )

        mcmc_mean = jnp.mean(mcmc_state.position, axis=0)
        mcmc_cov = jnp.cov(mcmc_state.position, rowvar=False)
        logging.info(f"MCMC mu :{mcmc_mean}")
        logging.info(f"MCMC Sigma : \n {mcmc_cov}")
        logging.info(f"Acceptance prob: {mcmc_info.is_accepted.mean()}")

    if "vi" in cfg.algorithm:
        cfg_algo = cfg.algorithm.vi

        log_ael_posterior = partial(
            bayes_model.log_posterior, check_sum_to_1=True, use_llog=False, use_ael=True
        )

        if cfg_algo.opt == "sgd":
            opt = optax.sgd(cfg_algo.learning_rate)
        elif cfg_algo.opt == "adam":
            opt = optax.adam(cfg_algo.learning_rate)
        else:
            raise ValueError("Invalid optimizer")
        loss_fn = vi.ElboLoss(log_ael_posterior, num_samples=cfg_algo.kl_samples)

        # Use Laplace as the initial distribution
        init_q, diagnostic = ep.laplace_approximation(
            bayes_model.log_posterior,
            bayes_model.init_theta(key),
            utils.newton_descent,
            max_iter=100,
            tol=1e-10,
        )
        L = jnp.linalg.cholesky(init_q.Sigma)
        idx = jnp.tril_indices(init_q.Sigma.shape[0], -1)
        log_diag_L = jnp.log(jnp.diag(L))
        off_L = L[idx]
        init_q = vi.FullGaussian(mu=init_q.mu, log_diag_L=log_diag_L, off_L=off_L)

        key, subkey = jax.random.split(key)
        q_ls, final_q, loss_ls, time_ls = vi.fit_to_variational_target(
            key=subkey,
            dist=init_q,
            loss_fn=loss_fn,
            optimizer=opt,
            iters=cfg_algo.iters,
            save_freq=cfg_algo.save_freq,
            show_progress=True,
        )

        utils.write_to(
            f"{outdir}/output.pickle",
            {"state": q_ls, "time": time_ls},
            verbose=True,
        )

        # save as pure numpy array and built-in structure, to be readable by reticulate
        utils.write_to(
            f"{outdir}/pure.pickle",
            {
                "state": {
                    "mean": np.asarray([q.mu for q in q_ls]),
                    "cov": np.asarray([q.Sigma for q in q_ls]),
                },
                "time": time_ls,
            },
        )

        logging.info(f"final mu: {final_q.mu}")
        logging.info(f"final Sigma: \n {final_q.Sigma}")

    if "ep" in cfg.algorithm:
        cfg_algo = cfg.algorithm.ep
        # Expectation propagation
        if cfg.model == "job":
            raise NotImplementedError("the SingleEL data should be region specific.")

        init_g, diagnostic = ep.laplace_approximation(
            bayes_model.log_posterior,
            bayes_model.init_theta(key),
            utils.newton_descent,
            max_iter=100,
            tol=1e-10,
        )
        logging.info(
            f"Laplace converged: {diagnostic.converged}, steps taken: {diagnostic.steps}"
        )
        # According to theory, precision of init_g must be sufficiently small
        init_g = (init_g.mu, init_g.Sigma * cfg_algo.init_cov_factor)

        key, algo_key, init_key = jax.random.split(key, 3)
        global_g_ls, final_gs, iterations, time_ls = ep.expectation_propagation(
            algo_key,
            bayes_model.log_posterior_term,
            bayes_model.n_terms,
            init_g,
            laplace_iter=cfg_algo.laplace_iter,
            snis_samples=cfg_algo.snis_samples,
            damping_factor=cfg_algo.damping_factor,
            tol=cfg_algo.tol,
            max_iter=cfg_algo.max_iter,
            verbose=cfg.verbose,
        )

        utils.write_to(
            f"{outdir}/output.pickle",
            {"state": global_g_ls, "time": time_ls},
            verbose=True,
        )

        # save as pure numpy array and built-in structure, to be readable by reticulate
        utils.write_to(
            f"{outdir}/pure.pickle",
            {
                "state": {
                    "mean": np.asarray([g.mu for g in global_g_ls]),
                    "cov": np.asarray([g.Sigma for g in global_g_ls]),
                },
                "time": time_ls,
            },
        )
        logging.info(f"final mu: {global_g_ls[-1].mu}")
        logging.info(f"final Sigma: \n {global_g_ls[-1].Sigma}")


# %%
if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    OmegaConf.register_new_resolver("dict_to_str", utils.dict_to_str)
    OmegaConf.register_new_resolver("print_algo", utils.print_algo)
    main()


# %%
