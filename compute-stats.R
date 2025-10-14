library(reticulate)
library(magrittr)

library(doParallel)

rm(list = ls())
use_condaenv("el")

r_mcmc_posterior <- function(posterior, n_samples) {
  ## Sample from an empirical distribution constructed from MCMC samples
  n_mcmc_total <- NROW(posterior)
  posterior[sample.int(n_mcmc_total, n_samples, replace = TRUE), ]
}

r_ep_posterior <- function(posterior, n_samples) {
  ## Sample from a Gaussian VI approximation
  mean <- posterior$mean
  cov <- posterior$cov
  cov <- (cov + t(cov)) / 2 # symmetrify the covariance matrix
  mvtnorm::rmvnorm(n_samples, mean, cov)
}


convert_py_to_r_gaussian <- function(posterior) {
  ## Convert a Gaussian posterior from Python to R.
  np <- reticulate::import("numpy")
  mean <- as.numeric(np$array(posterior$mu))
  cov <- np$array(posterior$Sigma)
  list(mean = mean, cov = cov)
}

convert_py_to_r_vi_output <- function(output_dir) {
  ## output is a list of state and time, and state itself is a list of Gaussian
  ## python objects
  output <- reticulate::py_load_object(output_dir)
  list(
    state = lapply(output$state, convert_py_to_r_gaussian),
    time = as.numeric(output$time)
  )
}

convert_py_to_r_mcmc_output <- function(output_dir) {
  ## output is a list of state and time, and state itself is blackjax mcmc state
  ## object. The 'position' of the state object is the parameter samples.
  output <- reticulate::py_load_object(output_dir)
  list(
    samples = as.matrix(output$state$position),
    time = as.numeric(output$time)
  )
}

compute_nbp_matches <- function(x, y) {
  ## Given two sets of samples x and y, compute the number of non-bipartite
  ## matches
  stopifnot(NROW(x) == NROW(y))

  all_samples <- rbind(x, y)
  n_samples <- NROW(all_samples)
  distobj <- nbpMatching::gendistance(as.data.frame(all_samples))
  matches <- nbpMatching::nonbimatch(nbpMatching::distancematrix(distobj))$halves

  ## For each pair, check if they are from the same set or not
  pair1_in_x <- matches$Group1.Row <= (n_samples / 2)
  pair2_in_x <- matches$Group2.Row <= (n_samples / 2)
  sum(pair1_in_x != pair2_in_x) # number of pairs coming from different sets
}

compute_mmd <- function(x, y, sigma = 1) {
  ## Given two sets of samples x and y, compute the number of non-bipartite
  ## matches
  stopifnot(NROW(x) == NROW(y))
  Ecume::mmd_test(x, y, kernel = "rbfdot", type = "unbiased", sigma = sigma)$statistic
}


# Create a parser object
parser <- argparse::ArgumentParser(description = "Compute non-bipartite matches between the gold standard and various algorithms.")
parser$add_argument("--model", type = "character", default = "regression", help = "model")
parser$add_argument("--statistic", type = "character", default = "nbp", help = "nbp or mmd")
parser$add_argument("--n_samples", type = "integer", default = 1000, help = "number of samples to draw from the posterior to compute statistic.")
parser$add_argument("--seed", type = "integer", default = 18021, help = "seed for reproducibility.")
parser$add_argument("--cores", type = "integer", default = 2, help = "run on how many cpu cores?")
parser$add_argument("--input_dir", type = "character", default = paste0("outputs/", lubridate::today()), help = "directory of the algorithm output.")
parser$add_argument("--save_dir", type = "character", default = paste0("outputs/nbp/", lubridate::today()), help = "directory to save nbp results.")
parser$add_argument("--timeline", type = "integer", default = 50, help = "number of breaks in plot timeline to compute statistic")
parser$add_argument("--maxtime", type = "numeric", default = 0.95, help = "the length of the plot timeline, as proportion to the experiment wall-time.")
parser$add_argument("--mmd_sigma", type = "numeric", default = 0.95, help = "sigma of the RBF kernel when computing MMD.")

cfg <- parser$parse_args()
registerDoParallel(cores = cfg$cores)

if (!dir.exists(cfg$save_dir)) dir.create(cfg$save_dir, recursive = TRUE)

if (cfg$statistic == "nbp") {
  compute_statistic <- compute_nbp_matches
} else if (cfg$statistic == "mmd") {
  compute_statistic <- \(x, y) compute_mmd(x, y, sigma = cfg$mmd_sigma)
}

## Get gold standard samples as a big matrix
gold <- list.files("outputs/2024-12-19", full.names = TRUE, recursive = TRUE) %>%
  stringr::str_subset(paste0("model=", cfg$model, "\\b.+output.pickle$")) %>%
  purrr::map(\(x) py_load_object(x)$state$position[-1, ]) %>%
  do.call(rbind, .)


# Load the data set
## future::plan(future::multisession, workers = cfg$cores)
future::plan(future::sequential) # For some reason, the parallel version doesn't work anymore
all_hmc <- list.files(cfg$input_dir, full.names = TRUE, recursive = TRUE) %>%
  stringr::str_subset(paste0("model=", cfg$model, " algo=hmc.+output.pickle$")) %>%
  furrr::future_map(convert_py_to_r_mcmc_output, .options = furrr::furrr_options(seed = TRUE))

all_ep <- list.files(cfg$input_dir, full.names = TRUE, recursive = TRUE) %>%
  stringr::str_subset(paste0("model=", cfg$model, " algo=ep.+output.pickle$")) %>%
  furrr::future_map(convert_py_to_r_vi_output, .options = furrr::furrr_options(seed = TRUE))

all_vi <- list.files(cfg$input_dir, full.names = TRUE, recursive = TRUE) %>%
  stringr::str_subset(paste0("model=", cfg$model, " algo=vi.+output.pickle$")) %>%
  furrr::future_map(convert_py_to_r_vi_output, .options = furrr::furrr_options(seed = TRUE))

all_rw <- list.files(cfg$input_dir, full.names = TRUE, recursive = TRUE) %>%
  stringr::str_subset(paste0("model=", cfg$model, " algo=rw.+output.pickle$")) %>%
  furrr::future_map(convert_py_to_r_mcmc_output, .options = furrr::furrr_options(seed = TRUE))
future::plan(future::sequential)

## Define the time point to evaluate stats
runtime <- purrr::map_dbl(list(all_rw, all_vi, all_ep, all_hmc), \(x) sum(x[[1]]$time))
plot_timeline <- seq(0, min(runtime) * cfg$maxtime, length.out = cfg$timeline)

all_stats <- foreach(j = seq_along(all_ep)) %dopar% {
  ## Loop over each of the 50 reps
  cat(paste0(j, " "))
  one_rep <- list(ep = all_ep[[j]], vi = all_vi[[j]], hmc = all_hmc[[j]], rw = all_rw[[j]], laplace = NULL)
  timeline <- lapply(one_rep, \(x) cumsum(x$time))
  stats <- lapply(one_rep, \(x) setNames(rep(0, length(plot_timeline)), plot_timeline))

  withr::with_seed(
    cfg$seed * j,
    {
      for (i in seq_along(plot_timeline)) {
        upto <- plot_timeline[i]
        gold_samples <- r_mcmc_posterior(gold, cfg$n_samples)

        ## EP, up to each point of the plot_timeline
        ep_posterior <- one_rep$ep$state[[sum(timeline$ep <= upto)]]
        ep_samples <- r_ep_posterior(ep_posterior, cfg$n_samples)
        stats$ep[i] <- compute_statistic(gold_samples, ep_samples)

        ## VI, up to each point of the plot_timeline
        vi_posterior <- one_rep$vi$state[[sum(timeline$vi <= upto)]]
        vi_samples <- r_ep_posterior(vi_posterior, cfg$n_samples)
        stats$vi[i] <- compute_statistic(gold_samples, vi_samples)

        ## HMC, up to each point of the plot_timeline
        hmc_posterior <- one_rep$hmc$samples[timeline$hmc <= upto, , drop = FALSE]
        hmc_samples <- r_mcmc_posterior(hmc_posterior, cfg$n_samples)
        stats$hmc[i] <- compute_statistic(gold_samples, hmc_samples)

        ## Random walk, up to each point of the plot_timeline
        rw_posterior <- one_rep$rw$samples[timeline$rw <= upto, , drop = FALSE]
        rw_samples <- r_mcmc_posterior(rw_posterior, cfg$n_samples)
        stats$rw[i] <- compute_statistic(gold_samples, rw_samples)

        ## Laplace. The initial value of VI is Laplace.
        laplace_posterior <- one_rep$vi$state[[1]]
        laplace_samples <- r_ep_posterior(laplace_posterior, cfg$n_samples)
        stats$laplace[i] <- compute_statistic(gold_samples, laplace_samples)
      }
      stats
    },
    .rng_kind = "L'Ecuyer-CMRG"
  )
}
cat("\n")

saveRDS(all_stats, file.path(cfg$save_dir, paste0(cfg$model, "-all-", cfg$statistic, ".rds")))
