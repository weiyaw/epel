# Expectation-propagation for Bayesian empirical likelihood inference
This repository contains the code used to reproduce the plots in the paper. The
file `emplik.py` implements empirical likelihood, which may be of independent
interest to users seeking to implement empirical likelihood in JAX.

The workflow of this experiment is as follows:
1. Run the algorithm to obtain the samples or fitted object.
2. Run `compute-nbp.R` to compute the NBP statistics.
3. Run `visual-nbp.R` to reproduce the plots in the paper.

## Examples

The following code performs one repetition of fitting and NBP computation. It
saves the fitted output in `outputs/2025-01-18` and the NBP computation results
in `outputs/nbp/2025-01-18`.


### GEE with Expectation-Propagation (EPEL)
``` bash
python main.py model=gee seed=856501 date=2025-01-18 prior_sd=10 \
    algorithm=ep \
    algorithm.ep.damping_factor=0.1 \
    algorithm.ep.max_iter=120 \
    algorithm.ep.laplace_iter=50 \
    algorithm.ep.n_terms=6 \
    algorithm.ep.init_cov_factor=1.0

Rscript compute-nbp.R --model=gee --seed=18021 --input_dir outputs/2025-01-18 --save_dir outputs/nbp/2025-01-18 --cores 50
```

### Kyphosis with HMC
``` bash
python main.py model=kyphosis seed=815401 date=2025-01-18 prior_sd=10 \
    algorithm=hmc \
    algorithm.hmc.n_samples=40000 \
    algorithm.hmc.step_size=0.01 \
    algorithm.hmc.num_integration_steps=50

Rscript compute-nbp.R --model=kyphosis --seed=18021 --input_dir outputs/2025-01-18 --save_dir outputs/nbp/2025-01-18 --cores 50
```

### High-dimensional Regression with Random Walk
``` bash
python main.py model=regression10 seed=991101 date=2025-01-18 prior_sd=10 \
    algorithm=rw \
    algorithm.rw.n_samples=500000 \
    algorithm.rw.n_warmup=10000 \
    algorithm.rw.sd_shrink_factor=0.5

Rscript compute-nbp.R --model=regression10 --seed=18021 --input_dir outputs/2025-01-18 --save_dir outputs/nbp/2025-01-18 --cores 50
```

### Quantile Regression with Variational Bayes
``` bash
python main.py model=quantile seed=544301 date=2025-01-18  prior_sd=10 \
    algorithm=vi \
    algorithm.vi.opt=adam \
    algorithm.vi.learning_rate=1e-3 \
    algorithm.vi.kl_samples=1 \
    algorithm.vi.iters=800000 \
    algorithm.vi.save_freq=2000

Rscript compute-nbp.R --model=quantile --seed=18021 --input_dir outputs/2025-01-18 --save_dir outputs/nbp/2025-01-18 --cores 50
```
