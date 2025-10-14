library(tidyverse)
theme_set(theme_bw() + theme(text = element_text(size = 10)))

# Load the data
input_dir <- "outputs/nbp/2025-01-18"

# NBP
raw_dat <- dir(input_dir, full.names = TRUE) %>%
  setNames(., str_extract(basename(.), ".*(?=-all-nbp.rds)")) %>%
  map(readRDS) %>%
  tibble(nbp = ., model = names(.)) %>%
  unnest_longer(nbp, indices_to = "rep") %>%
  unnest_longer(nbp, indices_to = "method") %>%
  unnest_longer(nbp, indices_to = "time") %>%
  mutate(time = zapsmall(as.numeric(time), 4))

plot_dat <- raw_dat %>%
  group_by(model, method, time) %>%
  summarise(
    mean_nbp = mean(nbp),
    sd_nbp = sd(nbp),
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
  ## ylim(c(400, NA)) +
  labs(title = "NBP. Ribbon = 25th-75th quantile", x = "time (seconds)")


algo_names <- c("EPEL", "HMC", "Laplace Approximation", "Metropolis-Hastings", "Variational Bayes")
names(algo_names) <- c("ep", "hmc", "laplace", "rw", "vi")
## algo_names <- function(x) {
##   algo[x]
## }

for (model in unique(plot_dat$model)) {
  ## Plot it individually
  p <- ggplot(plot_dat %>% filter(model == !!model)) +
    geom_line(aes(time, median_nbp, color = method), alpha = 1) +
    geom_ribbon(aes(x = time, ymin = quantile_25, ymax = quantile_75, fill = method), alpha = 0.2) +
    geom_hline(yintercept = 474, linetype = "dashed") +
    scale_color_brewer(palette = "Set1", labels = algo_names) +
    scale_fill_brewer(palette = "Set1", labels = algo_names) +
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
  ## ylim(c(0.1, NA)) +
  labs(title = "Mean NBP. Ribbon = mean +/- SE", x = "time (seconds)")


ggplot(diff_plot_dat) +
  geom_line(aes(time, mean_diff, color = method), alpha = 1) +
  geom_ribbon(aes(x = time, ymin = mean_diff - sd_diff, ymax = mean_diff + sd_diff, fill = method), alpha = 0.1) +
  facet_wrap(~model, scales = "free_x") +
  labs(title = "Mean of Difference in NBP from EP baseline", x = "time (seconds)") +
  coord_cartesian(ylim = c(-50, 50))

ggplot(diff_plot_dat) +
  geom_line(aes(time, mean_diff, color = method), alpha = 1) +
  geom_ribbon(aes(x = time, ymin = mean_diff - 1.96 * sd_diff, ymax = mean_diff + 1.96 * sd_diff, fill = method), alpha = 0.1) +
  facet_wrap(~model, scales = "free_x") +
  labs(
    title = "Pairwise difference in NBP from EP",
    subtitle = "Solid line: Mean, Shaded area: SE", y = "NBP diff", x = "time (seconds)"
  ) +
  coord_cartesian(ylim = c(-100, 100))


ggplot(diff_plot_dat) +
  geom_line(aes(time, median_diff, color = method), alpha = 1) +
  geom_ribbon(aes(x = time, ymin = quantile_25_diff, ymax = quantile_75_diff, fill = method), alpha = 0.1) +
  facet_wrap(~model, scales = "free_x") +
  labs(
    title = "Pairwise difference in NBP from EP",
    subtitle = "Solid line: Median, Shaded area: 25th-75th quantile", y = "NBP diff", x = "time (seconds)"
  ) +
  coord_cartesian(ylim = c(-100, 100))
