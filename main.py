# %%
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
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


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

    n_terms = cfg.algorithm.ep.n_terms if "ep" in cfg.algorithm else None

    if cfg.model == "kyphosis":
        bayes_model = model.Kyphosis(prior_sd=cfg.prior_sd, n_terms=n_terms)
    elif cfg.model == "regression":
        bayes_model = model.Regression(prior_sd=cfg.prior_sd, n_terms=n_terms)
    elif cfg.model == "quantregression":
        # Use exact score if the algorithm can handle non-differentiable h
        if cfg.algorithm == "rw":
            bayes_model = model.QuantRegression(
                tau=0.7, exact_score=True, prior_sd=cfg.prior_sd, n_terms=n_terms
            )
        else:
            bayes_model = model.QuantRegression(
                tau=0.7, exact_score=False, prior_sd=cfg.prior_sd, n_terms=n_terms
            )
    elif cfg.model == "quantregression3":
        if cfg.algorithm == "rw":
            # Use exact score if the algorithm can handle non-differentiable h
            bayes_model = model.QuantRegression3(
                tau=0.7, exact_score=True, prior_sd=cfg.prior_sd, n_terms=n_terms
            )
        else:
            bayes_model = model.QuantRegression3(
                tau=0.7, exact_score=False, prior_sd=cfg.prior_sd, n_terms=n_terms
            )
    elif cfg.model == "regression10":
        bayes_model = model.Regression10(prior_sd=cfg.prior_sd, n_terms=n_terms)
    elif cfg.model == "job":
        bayes_model = model.Job()
    elif cfg.model == "gee":
        bayes_model = model.GEE(prior_sd=cfg.prior_sd, n_terms=n_terms)
    else:
        raise ValueError("Invalid dataset")

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
            )

        utils.write_to(
            f"{outdir}/output.pickle",
            {"state": mcmc_state, "info": mcmc_info, "time": time_ls},
            verbose=True,
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

        # key, subkey = jax.random.split(key)
        # q = flowjax.flows.masked_autoregressive_flow(
        #     subkey,
        #     base_dist=StandardNormal((bayes_model.dim_theta,)),
        #     transformer=Affine(),
        #     nn_depth=2,
        #     # flow_layers=8,
        #     flow_layers=32,
        #     invert=True,
        # )
        # # logging.info(f"initial: {q}")

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
        logging.info(f"final mu: {global_g_ls[-1].mu}")
        logging.info(f"final Sigma: \n {global_g_ls[-1].Sigma}")


# %%
if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    OmegaConf.register_new_resolver("dict_to_str", utils.dict_to_str)
    OmegaConf.register_new_resolver("print_algo", utils.print_algo)
    main()


# %%
# OmegaConf.register_new_resolver("githash", utils.githash)
# OmegaConf.register_new_resolver("dict_to_str", utils.dict_to_str)
# with hydra.initialize(version_base=None, config_path="conf", job_name="test_app"):
#     cfg = hydra.compose(
#         config_name="main",
#         overrides=[
#             "hydra.run.dir=./outputs/debug",
#             "algorithm=[hmc, ep]",
#             "algorithm.hmc.n_samples=4000",
#             "algorithm.hmc.n_warmup=2000",
#             "algorithm.hmc.step_size=0.01",
#             "algorithm.hmc.num_integration_steps=50",
#             "algorithm.ep.damping_factor=0.1",
#             "algorithm.ep.max_iter=50",
#             "algorithm.ep.tol=1e-3",
#             "prior_sd=100",
#             "seed=200",
#             "verbose=True",
#             "model=gee",
#         ],
#     )
#     main(cfg)
