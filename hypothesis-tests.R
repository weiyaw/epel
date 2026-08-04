library(argparse)
library(tidyverse)

rm(list = ls())

sign_test <- function(x, mu = 0, alternative = "two.sided") {
  d <- x - mu
  d <- d[!is.na(d)]
  d <- d[d != 0]

  n <- length(d)
  if (n == 0) {
    return(list(
      p.value = NA_real_,
      estimate = NA_real_,
      statistic = NA_real_,
      n = 0L,
      alternative = alternative,
      method = "Sign test"
    ))
  }

  k <- sum(d > 0)

  p_value <- switch(
    alternative,
    greater = pbinom(k - 1, size = n, prob = 0.5, lower.tail = FALSE),
    less = pbinom(k, size = n, prob = 0.5, lower.tail = TRUE),
    two.sided = {
      p_left <- pbinom(k, size = n, prob = 0.5, lower.tail = TRUE)
      p_right <- pbinom(k - 1, size = n, prob = 0.5, lower.tail = FALSE)
      min(1, 2 * min(p_left, p_right))
    },
    stop("alternative must be one of: two.sided, greater, less")
  )

  list(
    p.value = p_value,
    estimate = k / n,
    statistic = as.numeric(k),
    n = as.integer(n),
    alternative = alternative,
    method = "Sign test"
  )
}

first_time_reached <- function(df, p_col, alpha) {
  df %>%
    group_by(model, method) %>%
    summarise(
      first_time = {
        ok <- time[!is.na(.data[[p_col]]) & .data[[p_col]] <= alpha]
        if (length(ok) == 0) NA_real_ else min(ok)
      },
      .groups = "drop"
    )
}

first_time_to_wide <- function(df, value_col) {
  df %>%
    pivot_wider(
      id_cols = model,
      names_from = method,
      values_from = all_of(value_col)
    ) %>%
    arrange(model)
}

# Create a parser object
parser <- ArgumentParser(description = "Read compute-stats outputs and run sign / Wilcoxon tests.")
parser$add_argument("--input_dir", type = "character", default = "outputs/nbp", help = "Directory containing *-all-nbp.rds files")
parser$add_argument("--pattern", type = "character", default = "-all-nbp\\.rds$", help = "Regex to match statistic files")
parser$add_argument("--threshold", type = "double", default = 474, help = "Threshold for one-sample tests")
parser$add_argument("--alpha", type = "double", default = 0.05, help = "Significance level")
parser$add_argument("--output_dir", type = "character", default = NULL, help = "Directory to save test results")

cfg <- parser$parse_args()

if (is.null(cfg$output_dir)) {
  cfg$output_dir <- file.path(cfg$input_dir, "hypothesis-tests")
}
if (!dir.exists(cfg$output_dir)) dir.create(cfg$output_dir, recursive = TRUE)

files <- list.files(cfg$input_dir, pattern = cfg$pattern, full.names = TRUE, recursive = TRUE)
if (length(files) == 0) {
  stop("No files matched pattern in input_dir. Check --input_dir and --pattern.")
}

cat(paste0("Found ", length(files), " files\n"))

raw_dat <- files %>%
  setNames(., stringr::str_extract(basename(.), ".*(?=-all-[^.]+\\.rds$)")) %>%
  map(readRDS) %>%
  tibble(stat = ., model = names(.)) %>%
  unnest_longer(stat, indices_to = "rep") %>%
  unnest_longer(stat, indices_to = "method") %>%
  unnest_longer(stat, indices_to = "time") %>%
  mutate(
    rep = as.integer(rep),
    method = as.character(method),
    time = as.numeric(time),
    stat = as.numeric(stat)
  ) %>%
  arrange(model, method, rep, time)

# Test 1: For each method and time point, test if stats exceed threshold.
threshold_tests <- raw_dat %>%
  group_by(model, method, time) %>%
  summarise(
    n_reps = dplyr::n(),
    n_ge_threshold = sum(stat >= cfg$threshold, na.rm = TRUE),
    prop_ge_threshold = n_ge_threshold / n_reps,
    sign = list(sign_test(stat, mu = cfg$threshold, alternative = "greater")),
    wilcox = list(wilcox.test(stat, mu = cfg$threshold, alternative = "greater", exact = FALSE, conf.int = FALSE)),
    .groups = "drop"
  ) %>%
  mutate(
    sign_p_value = map_dbl(sign, "p.value"),
    sign_statistic = map_dbl(sign, "statistic"),
    sign_n_non_ties = map_int(sign, "n"),
    sign_reject = sign_p_value <= cfg$alpha,
    wilcox_p_value = map_dbl(wilcox, "p.value"),
    wilcox_statistic = map_dbl(wilcox, "statistic"),
    wilcox_reject = wilcox_p_value <= cfg$alpha
  ) %>%
  select(-sign, -wilcox)

threshold_first_time_sign <- first_time_reached(threshold_tests, "sign_p_value", cfg$alpha)
threshold_first_time_wilcox <- first_time_reached(threshold_tests, "wilcox_p_value", cfg$alpha)
threshold_first_time_sign_wide <- first_time_to_wide(threshold_first_time_sign, "first_time")
threshold_first_time_wilcox_wide <- first_time_to_wide(threshold_first_time_wilcox, "first_time")

# Test 2: For each non-HMC method and time point, compare against HMC.
hmc_dat <- raw_dat %>%
  filter(method == "hmc") %>%
  select(model, rep, time, stat_hmc = stat)

if (nrow(hmc_dat) == 0) {
  stop("No method='hmc' found in the input files. Cannot run Test 2.")
}

paired_dat <- raw_dat %>%
  filter(method != "hmc") %>%
  inner_join(hmc_dat, by = c("model", "rep", "time")) %>%
  mutate(diff = stat - stat_hmc)

hmc_comparison_tests <- paired_dat %>%
  group_by(model, method, time) %>%
  summarise(
    n_pairs = dplyr::n(),
    n_ties = sum(diff == 0, na.rm = TRUE),
    mean_diff = mean(diff, na.rm = TRUE),
    median_diff = median(diff, na.rm = TRUE),
    sign = list(sign_test(diff, mu = 0, alternative = "two.sided")),
    wilcox = list(wilcox.test(stat, stat_hmc, paired = TRUE, alternative = "two.sided", exact = FALSE, conf.int = FALSE)),
    .groups = "drop"
  ) %>%
  mutate(
    sign_p_value = map_dbl(sign, "p.value"),
    sign_statistic = map_dbl(sign, "statistic"),
    sign_n_non_ties = map_int(sign, "n"),
    sign_reject_same = sign_p_value <= cfg$alpha,
    wilcox_p_value = map_dbl(wilcox, "p.value"),
    wilcox_statistic = map_dbl(wilcox, "statistic"),
    wilcox_reject_same = wilcox_p_value <= cfg$alpha
  ) %>%
  select(-sign, -wilcox)

hmc_first_time_notdiff_sign <- hmc_comparison_tests %>%
  group_by(model, method) %>%
  summarise(
    first_time_not_diff = {
      ok <- time[!is.na(sign_p_value) & sign_p_value > cfg$alpha]
      if (length(ok) == 0) NA_real_ else min(ok)
    },
    .groups = "drop"
  )

hmc_first_time_notdiff_wilcox <- hmc_comparison_tests %>%
  group_by(model, method) %>%
  summarise(
    first_time_not_diff = {
      ok <- time[!is.na(wilcox_p_value) & wilcox_p_value > cfg$alpha]
      if (length(ok) == 0) NA_real_ else min(ok)
    },
    .groups = "drop"
  )

hmc_first_time_notdiff_sign_wide <- first_time_to_wide(hmc_first_time_notdiff_sign, "first_time_not_diff")
hmc_first_time_notdiff_wilcox_wide <- first_time_to_wide(hmc_first_time_notdiff_wilcox, "first_time_not_diff")

write_csv(raw_dat, file.path(cfg$output_dir, "stats_long.csv"))
write_csv(threshold_tests, file.path(cfg$output_dir, "test1_threshold_sign_and_wilcox_by_time.csv"))
write_csv(threshold_first_time_sign_wide, file.path(cfg$output_dir, "test1_first_time_sign.csv"))
write_csv(threshold_first_time_wilcox_wide, file.path(cfg$output_dir, "test1_first_time_wilcox.csv"))
write_csv(hmc_comparison_tests, file.path(cfg$output_dir, "test2_vs_hmc_sign_and_wilcox_by_time.csv"))
write_csv(hmc_first_time_notdiff_sign_wide, file.path(cfg$output_dir, "test2_first_time_not_different_from_hmc_sign.csv"))
write_csv(hmc_first_time_notdiff_wilcox_wide, file.path(cfg$output_dir, "test2_first_time_not_different_from_hmc_wilcox.csv"))

cat("Saved outputs:\n")
cat(paste0("- ", file.path(cfg$output_dir, "stats_long.csv"), "\n"))
cat(paste0("- ", file.path(cfg$output_dir, "test1_threshold_sign_and_wilcox_by_time.csv"), "\n"))
cat(paste0("- ", file.path(cfg$output_dir, "test1_first_time_sign.csv"), "\n"))
cat(paste0("- ", file.path(cfg$output_dir, "test1_first_time_wilcox.csv"), "\n"))
cat(paste0("- ", file.path(cfg$output_dir, "test2_vs_hmc_sign_and_wilcox_by_time.csv"), "\n"))
cat(paste0("- ", file.path(cfg$output_dir, "test2_first_time_not_different_from_hmc_sign.csv"), "\n"))
cat(paste0("- ", file.path(cfg$output_dir, "test2_first_time_not_different_from_hmc_wilcox.csv"), "\n"))
