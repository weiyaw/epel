import sys
import logging
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

import model
import ep
import utils

jax.config.update("jax_enable_x64", True)
logging.basicConfig(level=logging.INFO)


def main(experiment_dir: str, n_skew_samples: int):
    cfg = OmegaConf.load(f"{experiment_dir}/.hydra/config.yaml")
    key = jax.random.PRNGKey(cfg.seed)

    if "ep" not in cfg.algorithm:
        raise ValueError(
            f"Only EP experiment dirs are supported. Got algorithm keys: {list(cfg.algorithm.keys())}"
        )

    bayes_model = model.build_model(
        cfg.model, cfg.prior_sd, n_points=cfg.algorithm.ep.n_points
    )

    output = utils.read_from(f"{experiment_dir}/output.pickle")
    global_g_ls = output["state"]

    skew_approx_ls = [
        ep.SkewSymmetricApproximation(
            symmetric_dist=g,
            log_posterior=bayes_model.log_posterior,
            theta_star=g.mu,
        )
        for g in global_g_ls
    ]

    skew_samples = []
    key, skew_key = jax.random.split(key)
    for i, skew_approx in enumerate(skew_approx_ls):
        skew_subkey = jax.random.fold_in(skew_key, i + 71)
        skew_samples.append(skew_approx.sample(skew_subkey, n_skew_samples))

    logging.info(
        f"Skew-EP: drew {n_skew_samples} samples for each of {len(global_g_ls)} iterates"
    )

    pure = utils.read_from(f"{experiment_dir}/pure.pickle")

    pure["skew_samples"] = np.asarray(skew_samples)
    utils.write_to(f"{experiment_dir}/pure.pickle", pure, verbose=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <experiment_dir> <n_skew_samples>")
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]))
