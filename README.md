# Expectation propagation for Bayesian empirical likelihood inference

This repository contains the code used to reproduce the experiments and plots
for the paper. The file [`emplik.py`](src/emplik.py) implements empirical
likelihood in JAX and may also be useful independently of the experiments.

The experiment workflow is:

1. Run an inference algorithm to produce samples or a fitted approximation.
2. Run `compute-stats.R` after all algorithm and gold-standard runs are
   available.
3. Run `visual-nbp.R` to reproduce the comparison plots.

## Repository structure

```text
.
├── main.py                  # Hydra entry point for inference experiments
├── skew-approx.py           # Adds skew-EP samples to completed EP runs
├── src/                     # Empirical likelihood, models, and algorithms
├── conf/                    # Hydra model-independent and algorithm settings
├── data/                    # Input datasets used by the fitted models
├── tests/                   # Python unit and regression smoke tests
├── run-{model}.sh           # Docker launchers for the eight fitted models
├── run-skew-approx.sh       # Batch skew-EP post-processing launcher
├── run-experiments.sh       # Full experiment and analysis orchestrator
├── compute-stats.R          # Computes comparison statistics
├── hypothesis-tests.R       # Runs hypothesis-test analyses
├── visual-nbp.R             # Produces comparison plots
├── Dockerfile               # Reproducible container environment
├── .dockerignore            # Docker build-context allowlist
├── pyproject.toml           # Python project and dependency metadata
├── uv.lock                  # Locked Python dependency versions
└── LICENSE                  # MIT license
```

## Setup

The Python environment is managed with
[`uv`](https://docs.astral.sh/uv/). The repository selects Python 3.13; R is
also required for the post-processing and plotting scripts.

Install the locked Python dependencies from the repository root:

```bash
uv sync --locked
```

The batch scripts run experiments in Docker. Build the image they expect from
the repository root:

```bash
docker build -t bayes-el .
```

R packages used by `compute-stats.R` and `visual-nbp.R` must be installed
separately.

## Examples

The following commands each run one fit with Hydra configuration overrides.
They write their results below `outputs/example-run/`; replace the `id` and seed
values as needed.

### GEE with expectation propagation (EPEL)

```bash
uv run python main.py model=gee seed=856501 id=example-run prior_sd=10 \
    algorithm=ep \
    algorithm.ep.damping_factor=0.1 \
    algorithm.ep.max_iter=120 \
    algorithm.ep.laplace_iter=50 \
    algorithm.ep.n_points=10 \
    algorithm.ep.init_cov_factor=1.0
```

### Kyphosis with HMC

```bash
uv run python main.py model=kyphosis seed=815401 id=example-run prior_sd=10 \
    algorithm=hmc \
    algorithm.hmc.n_samples=40000 \
    algorithm.hmc.step_size=0.01 \
    algorithm.hmc.num_integration_steps=50
```

### High-dimensional regression with random walk

```bash
uv run python main.py model=regression10 seed=991101 id=example-run prior_sd=10 \
    algorithm=rw \
    algorithm.rw.n_samples=500000 \
    algorithm.rw.n_warmup=10000 \
    algorithm.rw.sd_shrink_factor=0.5
```

### Quantile regression with variational inference

```bash
uv run python main.py model=quantregression seed=544301 id=example-run prior_sd=10 \
    algorithm=vi \
    algorithm.vi.opt=adam \
    algorithm.vi.learning_rate=1e-3 \
    algorithm.vi.kl_samples=1 \
    algorithm.vi.iters=800000 \
    algorithm.vi.save_freq=2000
```

## Batch experiment scripts

Eight model-specific scripts launch the main experiment batches:

| Script | Model |
| --- | --- |
| `run-breastfeed.sh` | Breastfeeding data |
| `run-gee.sh` | Generalized estimating equations |
| `run-kyphosis.sh` | Kyphosis data |
| `run-mroz.sh` | Mroz labour-force data |
| `run-orings.sh` | O-ring data |
| `run-quantregression.sh` | Quantile regression |
| `run-regression.sh` | Regression |
| `run-regression10.sh` | High-dimensional regression |

They share the following interface, where `output-id` becomes the directory
immediately below `outputs/`:

```bash
./run-<model>.sh <output-id> [--gold]
```

Normal mode launches EP, HMC, random-walk, and variational-inference jobs for
50 seeds (`991100` through `991149`). Gold mode launches 10 longer reference
runs (`304900` through `304909`): it uses HMC for every model except quantile
regression, which uses random walk. The scripts run containers in detached
mode, mount the repository at `/home`, allocate four CPUs per normal job or six
per gold job, and limit the respective concurrency to 15 or 10 containers.
Model-specific algorithm settings are defined directly in each launcher.

For example, run the GEE experiment batch and its gold-standard batch with:

```bash
./run-gee.sh main-exp
./run-gee.sh gold --gold
```

After EP runs are complete, `run-skew-approx.sh` finds matching EP output
directories and adds skew-EP samples to their `pure.pickle` files:

```bash
./run-skew-approx.sh --input_dir outputs/main-exp orings breastfeed
```

The input directory and at least one model name are required. Its concurrency,
CPU allocation, and samples per fitted approximation can be overridden with
the `MAX_CONTAINERS`, `CORES_PER_CONTAINER`, and `N_SKEW_SAMPLES` environment
variables; their defaults are 15, 1, and 1000, respectively.

`run-experiments.sh` runs all gold-standard and main batches, waits between
model groups, performs skew-EP post-processing, and invokes `compute-stats.R`.
It launches the complete computational workload, waits for all running Docker
containers on the host, and removes successfully stopped containers. Run it
only on a Docker host dedicated to these experiments:

```bash
./run-experiments.sh
```

## Post-processing

`compute-stats.R` compares the completed experiment runs with separate
gold-standard samples. For example, if the experiment outputs are in
`outputs/example-run/` and the gold-standard outputs are in
`outputs/gold-standard/`, run:

```bash
Rscript compute-stats.R \
    --model=gee \
    --seed=18021 \
    --input_dir=outputs/example-run \
    --gold_dir=outputs/gold-standard \
    --save_dir=outputs/nbp/example-run \
    --cores=50
```

Repeat the command for each model to analyse. Computed statistics are written
beneath `outputs/nbp/<id>/`. The plotting workflow is implemented in
`visual-nbp.R`; set its `input_dir` to the statistics directory before running
it with `Rscript visual-nbp.R`.

## License

This project is available under the [MIT License](LICENSE).
