library(tidyverse)
theme_set(theme_bw() + theme(text = element_text(size = 10)))

# Load the data
## input_dir <- "outputs/nbp/2025-01-18"
input_dir <- "outputs/nbp/2026-01-04"


algo_names <- c("EPEL", "EPEL (skewed)", "HMC", "Laplace Approximation", "Metropolis-Hastings", "Variational Bayes")
names(algo_names) <- c("ep", "ep_skew", "hmc", "laplace", "rw", "vi")
algo_order <- c("ep", "ep_skew", "hmc", "laplace", "rw", "vi")
algo_labels <- unname(algo_names[algo_order])
algo_colors <- setNames(RColorBrewer::brewer.pal(9, "Set1")[c(1, 7, 2:5)], algo_order)
exclude <- c("cushings", "cushings-logistic")

# NBP
raw_dat <- dir(input_dir, full.names = TRUE) %>%
  str_extract(".*-all-nbp.rds$") %>%
  discard(is.na) %>%
  setNames(., str_extract(basename(.), ".*(?=-all-nbp.rds)")) %>%
  map(readRDS) %>%
  tibble(nbp = ., model = names(.)) %>%
  unnest_longer(nbp, indices_to = "rep") %>%
  unnest_longer(nbp, indices_to = "method") %>%
  unnest_longer(nbp, indices_to = "time") %>%
  mutate(
    time = zapsmall(as.numeric(time), 4),
    method = factor(method, levels = algo_order)
  )


plot_dat <- raw_dat %>%
  group_by(model, method, time) %>%
  filter(!model %in% exclude) %>%
  summarise(
    mean_nbp = mean(nbp),
    sd_nbp = sd(nbp),
    ci_upper = mean_nbp + qt(0.975, length(nbp) - 1) * sd_nbp,
    ci_lower = mean_nbp - qt(0.975, length(nbp) - 1) * sd_nbp,
    median_nbp = median(nbp),
    quantile_25 = quantile(nbp, 0.25),
    quantile_75 = quantile(nbp, 0.75),
  )

ep_baseline <- filter(raw_dat, method == "ep") %>%
  select(-method)

diff_plot_dat <- raw_dat %>%
  left_join(ep_baseline, by = c("model", "time", "rep"), suffix = c("", "_ep")) %>%
  mutate(diff_nbp = nbp - nbp_ep) %>%
  group_by(model, method, time) %>%
  summarise(
    mean_diff = mean(diff_nbp),
    sd_diff = sd(diff_nbp),
    median_diff = median(diff_nbp),
    quantile_25_diff = quantile(diff_nbp, 0.25),
    quantile_75_diff = quantile(diff_nbp, 0.75),
  )


## Median plot
ggplot(plot_dat) +
  geom_line(aes(time, median_nbp, color = method), alpha = 1) +
  geom_ribbon(aes(x = time, ymin = quantile_25, ymax = quantile_75, fill = method), alpha = 0.1) +
  geom_hline(yintercept = 474, linetype = "dashed") +
  facet_wrap(~model, scales = "free_x") +
  scale_color_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  scale_fill_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  coord_cartesian(ylim = c(300, 510)) +
  labs(title = "Median NBP. Ribbon = 25th-75th quantile", x = "time (seconds)")

## Mean plot
ggplot(plot_dat) +
  geom_line(aes(time, mean_nbp, color = method), alpha = 1) +
  geom_ribbon(aes(x = time, ymin = ci_lower, ymax = ci_upper, fill = method), alpha = 0.1) +
  geom_hline(yintercept = 474, linetype = "dashed") +
  facet_wrap(~model, scales = "free_x") +
  scale_color_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  scale_fill_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  coord_cartesian(ylim = c(300, 510)) +
  labs(title = "Mean NBP. Ribbon = 95% CI", x = "time (seconds)")

for (model in unique(plot_dat$model)) {
  ## Plot it individually
  p <- ggplot(plot_dat %>% filter(model == !!model)) +
    geom_line(aes(time, median_nbp, color = method), alpha = 1) +
    geom_ribbon(aes(x = time, ymin = quantile_25, ymax = quantile_75, fill = method), alpha = 0.2) +
    geom_hline(yintercept = 474, linetype = "dashed") +
    scale_color_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
    scale_fill_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
    coord_cartesian(ylim = c(300, 510)) +
    theme(
      legend.position = "inside",
      legend.position.inside = c(0.75, 0.25), # horizontal, vertical from bottom left corner
      legend.title = element_blank(),
      legend.margin = margin(c(0, 0, 0, 0)),
      legend.box.background = element_rect(color = "black"),
      plot.margin = margin(c(0, 0, 0, 5)),
      text = element_text(size = NULL),
    ) +
    labs(y = "NBP", x = "Time (seconds)")
  ggsave(p, filename = paste0("plots/nbp-median-", model, ".pdf"), width = 4, height = 3)
  print(p)
}








## Mean plot
ggplot(plot_dat) +
  geom_line(aes(time, mean_nbp, color = method), alpha = 1) +
  geom_ribbon(aes(time, ymin = mean_nbp - sd_nbp, ymax = mean_nbp + sd_nbp, fill = method), alpha = 0.1) +
  geom_hline(yintercept = 474, linetype = "dashed") +
  facet_wrap(~model, scales = "free_x") +
  scale_color_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  scale_fill_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  ## ylim(c(0.1, NA)) +
  labs(title = "Mean NBP. Ribbon = mean +/- SE", x = "time (seconds)")


ggplot(diff_plot_dat) +
  geom_line(aes(time, mean_diff, color = method), alpha = 1) +
  geom_ribbon(aes(x = time, ymin = mean_diff - sd_diff, ymax = mean_diff + sd_diff, fill = method), alpha = 0.1) +
  facet_wrap(~model, scales = "free_x") +
  scale_color_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  scale_fill_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  labs(title = "Mean of Difference in NBP from EP baseline", x = "time (seconds)") +
  coord_cartesian(ylim = c(-50, 50))

ggplot(diff_plot_dat) +
  geom_line(aes(time, mean_diff, color = method), alpha = 1) +
  geom_ribbon(aes(x = time, ymin = mean_diff - 1.96 * sd_diff, ymax = mean_diff + 1.96 * sd_diff, fill = method), alpha = 0.1) +
  facet_wrap(~model, scales = "free_x") +
  scale_color_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  scale_fill_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  labs(
    title = "Pairwise difference in NBP from EP",
    subtitle = "Solid line: Mean, Shaded area: SE", y = "NBP diff", x = "time (seconds)"
  ) +
  coord_cartesian(ylim = c(-100, 100))


ggplot(diff_plot_dat) +
  geom_line(aes(time, median_diff, color = method), alpha = 1) +
  geom_ribbon(aes(x = time, ymin = quantile_25_diff, ymax = quantile_75_diff, fill = method), alpha = 0.1) +
  facet_wrap(~model, scales = "free_x") +
  scale_color_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  scale_fill_manual(values = algo_colors, breaks = algo_order, labels = algo_labels) +
  labs(
    title = "Pairwise difference in NBP from EP",
    subtitle = "Solid line: Median, Shaded area: 25th-75th quantile", y = "NBP diff", x = "time (seconds)"
  ) +
  coord_cartesian(ylim = c(-100, 100))

# one-sided t-test over the threshold
#
raw_dat %>%
  filter(model == "gee") %>%
  group_by(model, method, time) %>%
  summarise(test = list(t.test(nbp, mu = 474, alternative = "greater"))) %>%
  mutate(
    p_value = map_dbl(test, "p.value"),
    mean_nbp = map_dbl(test, "estimate")
  )


  summarise(nbp_reps = list(nbp)) %>%



  mutate(test = map(nbp_reps, \(x) t.test(x, mu = 474, alternative = "greater")))





%>%
  mutate(nbp = list(nbp))

%>%
