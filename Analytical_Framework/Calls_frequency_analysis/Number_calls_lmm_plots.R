# Number of calls analysis - lmm and plots
# Antonella Torrisi

rm(list = ls())

library(extrafont)
if (requireNamespace("extrafont", quietly = TRUE)) {
  tryCatch(loadfonts(device = "pdf"), error = function(e) NULL)
}

library(plyr)
library(dplyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(emmeans)
library(readr)
library(tidyr)
library(effectsize)

# load data
setwd("/path/to/project")
#setwd("C:\Users\anton\VPA_vocalisations_project\Results_Clustering")
data_path <- file.path("data", "full_dataset_vpa_ctrl.csv")
t <- read_csv(data_path)
names(t)

# call counts per id x bin x condition x cluster
counts_full <- t %>%
  group_by(id, bin, condition, cluster_membership_2) %>%
  summarise(num_observations = n(), .groups = "drop") %>%
  filter(!is.na(bin), !is.na(condition), !is.na(cluster_membership_2))

counts_full$id                   <- as.factor(counts_full$id)
counts_full$bin                  <- as.factor(counts_full$bin)
counts_full$condition            <- as.factor(counts_full$condition)
counts_full$cluster_membership_2 <- as.factor(counts_full$cluster_membership_2)
counts_full$bin                  <- relevel(counts_full$bin, ref = "1")
counts_full                      <- na.omit(counts_full)

# plot: mean calls by bin and condition
summary_condition <- counts_full %>%
  group_by(id, bin, condition) %>%
  summarise(total_obs = sum(num_observations), .groups = "drop") %>%
  group_by(bin, condition) %>%
  summarise(
    mean_obs   = mean(total_obs, na.rm = TRUE),
    se_obs     = sd(total_obs, na.rm = TRUE) / sqrt(n()),
    n_subjects = n(),
    .groups    = "drop"
  )

plot_condition <- ggplot(summary_condition,
                         aes(x = as.numeric(as.character(bin)), y = mean_obs,
                             color = condition, fill = condition)) +
  geom_ribbon(aes(ymin = mean_obs - se_obs, ymax = mean_obs + se_obs),
              alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 4) +
  scale_color_manual(values = c("CTRL" = "#F8766D", "VPA" = "#00BFC4"),
                     labels = c("CTRL" = "CTRL", "VPA" = "VPA")) +
  scale_fill_manual(values  = c("CTRL" = "#F8766D", "VPA" = "#00BFC4"),
                    labels  = c("CTRL" = "CTRL", "VPA" = "VPA")) +
  scale_x_continuous(breaks = 1:6, labels = 1:6) +
  labs(x = "Time window", y = "Mean Number of Calls", color = "Condition", fill = "Condition") +
  theme_minimal(base_size = 18, base_family = "sans") +
  theme(
    plot.title        = element_blank(),
    axis.line         = element_line(color = "black", linewidth = 0.8),
    axis.ticks        = element_line(color = "black", linewidth = 0.6),
    axis.ticks.length = unit(0.2, "cm"),
    axis.title.x      = element_text(size = 20, face = "italic", margin = margin(t = 15)),
    axis.title.y      = element_text(size = 20, face = "italic", margin = margin(r = 15)),
    axis.text.x       = element_text(size = 18, color = "black"),
    axis.text.y       = element_text(size = 18, color = "black"),
    legend.title      = element_text(size = 18, face = "bold"),
    legend.text       = element_text(size = 16),
    legend.position   = "bottom",
    panel.grid.major  = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor  = element_blank(),
    panel.border      = element_blank(),
    panel.background  = element_rect(fill = "white", color = NA),
    plot.background   = element_rect(fill = "white", color = NA)
  )

# plot: mean calls by bin and cluster
summary_cluster <- counts_full %>%
  group_by(id, bin, cluster_membership_2) %>%
  summarise(total_obs = sum(num_observations), .groups = "drop") %>%
  group_by(bin, cluster_membership_2) %>%
  summarise(
    mean_obs   = mean(total_obs, na.rm = TRUE),
    se_obs     = sd(total_obs, na.rm = TRUE) / sqrt(n()),
    n_subjects = n(),
    .groups    = "drop"
  )

plot_cluster <- ggplot(summary_cluster,
                       aes(x = as.numeric(as.character(bin)), y = mean_obs,
                           color = cluster_membership_2, fill = cluster_membership_2)) +
  geom_ribbon(aes(ymin = mean_obs - se_obs, ymax = mean_obs + se_obs),
              alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 4) +
  scale_color_manual(values = c("0" = "#F8766D", "1" = "#00BFC4"),
                     labels = c("0" = "Cluster 0", "1" = "Cluster 1")) +
  scale_fill_manual(values  = c("0" = "#F8766D", "1" = "#00BFC4"),
                    labels  = c("0" = "Cluster 0", "1" = "Cluster 1")) +
  scale_x_continuous(breaks = 1:6, labels = 1:6) +
  labs(x = "Time window", y = "Mean Number of Calls", color = "Cluster", fill = "Cluster") +
  theme_minimal(base_size = 18, base_family = "sans") +
  theme(
    plot.title        = element_blank(),
    axis.line         = element_line(color = "black", linewidth = 0.8),
    axis.ticks        = element_line(color = "black", linewidth = 0.6),
    axis.ticks.length = unit(0.2, "cm"),
    axis.title.x      = element_text(size = 20, face = "italic", margin = margin(t = 15)),
    axis.title.y      = element_text(size = 20, face = "italic", margin = margin(r = 15)),
    axis.text.x       = element_text(size = 18, color = "black"),
    axis.text.y       = element_text(size = 18, color = "black"),
    legend.title      = element_text(size = 18, face = "bold"),
    legend.text       = element_text(size = 16),
    legend.position   = "bottom",
    panel.grid.major  = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor  = element_blank(),
    panel.border      = element_blank(),
    panel.background  = element_rect(fill = "white", color = NA),
    plot.background   = element_rect(fill = "white", color = NA)
  )

# plot: full interaction (condition x cluster)
summary_full <- counts_full %>%
  group_by(id, bin, condition, cluster_membership_2) %>%
  summarise(total_obs = sum(num_observations), .groups = "drop") %>%
  group_by(bin, condition, cluster_membership_2) %>%
  summarise(
    mean_obs   = mean(total_obs, na.rm = TRUE),
    se_obs     = sd(total_obs, na.rm = TRUE) / sqrt(n()),
    n_subjects = n(),
    .groups    = "drop"
  )

plot_interaction <- ggplot(summary_full,
                           aes(x = as.numeric(as.character(bin)), y = mean_obs,
                               color = condition, group = condition)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3.5) +
  geom_errorbar(aes(ymin = mean_obs - se_obs, ymax = mean_obs + se_obs),
                width = 0.1) +
  scale_color_manual(values = c("CTRL" = "#F8766D", "VPA" = "#00BFC4"),
                     labels = c("CTRL" = "CTRL", "VPA" = "VPA")) +
  scale_x_continuous(breaks = 1:6, labels = 1:6) +
  labs(x = "Time window", y = "Mean Number of Calls", color = "Condition") +
  facet_wrap(~ cluster_membership_2,
             labeller = labeller(cluster_membership_2 = c("0" = "Cluster 0", "1" = "Cluster 1"))) +
  theme_minimal(base_size = 18, base_family = "sans") +
  theme(
    plot.title        = element_blank(),
    axis.line         = element_line(color = "black", linewidth = 0.8),
    axis.ticks        = element_line(color = "black", linewidth = 0.6),
    axis.ticks.length = unit(0.2, "cm"),
    axis.title.x      = element_text(size = 20, face = "italic", margin = margin(t = 15)),
    axis.title.y      = element_text(size = 20, face = "italic", margin = margin(r = 15)),
    axis.text.x       = element_text(size = 18, color = "black"),
    axis.text.y       = element_text(size = 18, color = "black"),
    legend.title      = element_text(size = 18, face = "bold"),
    legend.text       = element_text(size = 16),
    legend.position   = "bottom",
    strip.text        = element_text(size = 20, face = "bold"),
    panel.grid.major  = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor  = element_blank(),
    panel.border      = element_blank(),
    panel.background  = element_rect(fill = "white", color = NA),
    plot.background   = element_rect(fill = "white", color = NA)
  )

# lmm - progressive simplification via likelihood ratio tests
# outcome: total calls per id x bin x condition x cluster
calls_for_stats <- counts_full %>%
  group_by(id, bin, condition, cluster_membership_2) %>%
  summarise(total_calls = sum(num_observations), .groups = "drop")

model_full          <- lmer(total_calls ~ bin * condition * cluster_membership_2 + (1 | id),
                            data = calls_for_stats)
model_no_cluster    <- lmer(total_calls ~ bin * condition + (1 | id),
                            data = calls_for_stats)
model_no_condition  <- lmer(total_calls ~ bin * cluster_membership_2 + (1 | id),
                            data = calls_for_stats)
model_no_bin        <- lmer(total_calls ~ condition * cluster_membership_2 + (1 | id),
                            data = calls_for_stats)
model_no_cond_clust <- lmer(total_calls ~ bin + condition + cluster_membership_2 +
                              bin:condition + bin:cluster_membership_2 + (1 | id),
                            data = calls_for_stats)

anova_cluster    <- anova(model_no_cluster,    model_full)
anova_condition  <- anova(model_no_condition,  model_full)
anova_bin        <- anova(model_no_bin,        model_full)
anova_cond_clust <- anova(model_no_cond_clust, model_full)

cat("cluster effect:\n");    print(anova_cluster)
cat("condition effect:\n");  print(anova_condition)
cat("bin effect:\n");        print(anova_bin)
cat("condition x cluster:\n"); print(anova_cond_clust)
cat("full model summary:\n");  print(summary(model_full))

# post-hoc if condition x cluster interaction is significant
if (anova_cond_clust$`Pr(>Chisq)`[2] < 0.05) {
  emm_interaction <- emmeans(model_full, ~ condition * cluster_membership_2)
  cat("cluster differences within condition:\n")
  print(pairs(emm_interaction, by = "condition", adjust = "tukey"))
  cat("condition differences within cluster:\n")
  print(pairs(emm_interaction, by = "cluster_membership_2", adjust = "tukey"))
}

# save
calls_output_dir <- file.path(getwd(), "calls_analysis_results")
if (!dir.exists(calls_output_dir)) dir.create(calls_output_dir, recursive = TRUE)

ggsave(file.path(calls_output_dir, "calls_by_condition.png"),
       plot_condition, width = 10, height = 6, dpi = 300, bg = "white")
ggsave(file.path(calls_output_dir, "calls_by_cluster.png"),
       plot_cluster, width = 10, height = 6, dpi = 300, bg = "white")
ggsave(file.path(calls_output_dir, "calls_interaction.png"),
       plot_interaction, width = 12, height = 7, dpi = 300, bg = "white")

write.csv(summary_condition, file.path(calls_output_dir, "summary_condition.csv"),   row.names = FALSE)
write.csv(summary_cluster,   file.path(calls_output_dir, "summary_cluster.csv"),     row.names = FALSE)
write.csv(summary_full,      file.path(calls_output_dir, "summary_interaction.csv"), row.names = FALSE)

calls_stats_results <- data.frame(
  Model_Comparison = c("Cluster_Effect", "Condition_Effect", "Bin_Effect", "Condition_x_Cluster"),
  Chi_Square = c(anova_cluster$Chisq[2],   anova_condition$Chisq[2],
                 anova_bin$Chisq[2],        anova_cond_clust$Chisq[2]),
  Df         = c(anova_cluster$Df[2],      anova_condition$Df[2],
                 anova_bin$Df[2],           anova_cond_clust$Df[2]),
  P_Value    = c(anova_cluster$`Pr(>Chisq)`[2],   anova_condition$`Pr(>Chisq)`[2],
                 anova_bin$`Pr(>Chisq)`[2],        anova_cond_clust$`Pr(>Chisq)`[2]),
  Significant = c(anova_cluster$`Pr(>Chisq)`[2]   < 0.05, anova_condition$`Pr(>Chisq)`[2] < 0.05,
                  anova_bin$`Pr(>Chisq)`[2]        < 0.05, anova_cond_clust$`Pr(>Chisq)`[2] < 0.05),
  stringsAsFactors = FALSE
)
write.csv(calls_stats_results, file.path(calls_output_dir, "calls_statistical_results.csv"), row.names = FALSE)