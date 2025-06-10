# Clear the workspace to ensure a fresh environment
rm(list = ls())

# Load required libraries
library(readr)
library(dplyr)
library(ggplot2)
library(plyr)
library(lme4)
library(xtable)
library(tidyr)

# Define base and output folders
base_path <- "C:/Users/anton/VPA_vocalisations_project"
output_stats_folder <- file.path(base_path, "Results_univariate_analysis_46")
output_plots_folder <- file.path(output_stats_folder, "Plots")
if (!dir.exists(output_stats_folder)) dir.create(output_stats_folder, recursive = TRUE)
if (!dir.exists(output_plots_folder)) dir.create(output_plots_folder, recursive = TRUE)

# Function to assign significance stars according to p-value thresholds
get_stars <- function(p) {
  if (p < 0.001) "***"
  else if (p < 0.01) "**"
  else if (p < 0.05) "*"
  else ""
}

# Function to prettify feature names for plot labels (replace _ or - with space)
prettify_feature <- function(feat) {
  gsub("_", " ", feat)
}

# Import clustering membership data for both groups and assign condition labels
ctrl_membership <- read_csv(
  file.path(base_path, "Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv"),
  show_col_types = FALSE) %>%
  mutate(condition = "CTRL")

vpa_membership <- read_csv(
  file.path(base_path, "Results_Clustering/hierarchical_clustering_vpa_46_3_clusters/hierarchical_clustering_3_distance_membership.csv"),
  show_col_types = FALSE) %>%
  mutate(condition = "VPA")

# Combine the two datasets into one dataframe
combined_membership <- bind_rows(ctrl_membership, vpa_membership)

# Clean column names by replacing spaces and hyphens with underscores
clean_names <- function(x) {
  x %>%
    gsub(" ", "_", .) %>%
    gsub("-", "_", .)
}
colnames(combined_membership) <- clean_names(colnames(combined_membership))

# Save the cleaned and combined dataset for future reference
output_data_folder <- file.path(base_path, "Results_R", "Features_univariate_analysis_46")
if (!dir.exists(output_data_folder)) dir.create(output_data_folder, recursive = TRUE)
write.csv(combined_membership, file.path(output_data_folder, "combined_dataset.csv"), row.names = FALSE)

# Prepare the data for subsequent analyses
t <- combined_membership %>%
  rename(id = recording, group = condition)
t$group <- as.factor(t$group)

# List of features to be analysed (column names must match those in the dataframe)
features <- c("Duration_call", "F0_Mean", "F0_Std", "F0_Skewness", "F0_Kurtosis",
              "F0_Bandwidth", "F0_1st_Order_Diff", "F0_Slope", "F0_Mag_Mean",
              "F1_Mag_Mean", "F2_Mag_Mean", "F1_F0_Ratio", "F2_F0_Ratio",
              "Spectral_Centroid_Mean", "Spectral_Centroid_Std", "RMS_Mean", "RMS_Std",
              "Slope", "Attack_magnitude", "Attack_time")

# Initialise dataframes to store results for each test
t_test_results <- data.frame()
wilcoxon_results <- data.frame()
lmm_results <- data.frame()

# Loop over each feature to conduct statistical tests and generate plots
for (feature in features) {
  message("Processing: ", feature)
  if (!(feature %in% colnames(t))) {
    message("Feature ", feature, " not found in dataset. Skipping...")
    next
  }
  
  # average for chick and group
  summary_data <- ddply(t, .(id, group), summarise,
                        mean_feature = mean(get(feature), na.rm = TRUE),
                        sd_feature = sd(get(feature), na.rm = TRUE),
                        N = sum(!is.na(get(feature)))
  )
  
  
  # Calculate group mean, standard error, and 95% confidence interval
  summary_plot <- ddply(summary_data, .(group), summarise,
                        mean_group = mean(mean_feature, na.rm = TRUE),
                        se_group   = sd(mean_feature, na.rm = TRUE) / sqrt(length(na.omit(mean_feature))),
                        N          = length(na.omit(mean_feature))
  )
  
  # Calculate the critical t-value for 95% confidence interval (degrees of freedom = N-1 per group)
  summary_plot$ci_low  <- summary_plot$mean_group - qt(0.975, summary_plot$N - 1) * summary_plot$se_group
  summary_plot$ci_high <- summary_plot$mean_group + qt(0.975, summary_plot$N - 1) * summary_plot$se_group
  
  #print(summary_plot)
  
  # Conduct t-test (comparing means between groups, using chick means)
  t_test_result <- t.test(mean_feature ~ group, data = summary_data)
  
  # Conduct Wilcoxon rank-sum test (non-parametric alternative)
  wilcoxon_result <- wilcox.test(mean_feature ~ group, data = summary_data, exact = FALSE)
  
  # Fit a linear mixed-effects model with random intercept for individual
  mod <- lmer(as.formula(paste(feature, "~ group + (1 | id)")), data = t, REML = FALSE)
  mod_null <- lmer(as.formula(paste(feature, "~ (1 | id)")), data = t, REML = FALSE)
  model_summary <- summary(mod)
  anova_result <- anova(mod, mod_null)
  model_estimate <- model_summary$coefficients[2,1]
  model_se <- model_summary$coefficients[2,2]
  model_t <- model_summary$coefficients[2,3]
  model_p <- anova_result$`Pr(>Chisq)`[2]
  model_AIC <- AIC(mod)
  
  # Store test results in the respective dataframes
  t_test_results <- rbind(t_test_results, data.frame(
    Feature = feature,
    Mean_CTRL = summary_plot$mean_group[summary_plot$group == "CTRL"], 
    Mean_VPA = summary_plot$mean_group[summary_plot$group == "VPA"],
    SD_CTRL = sd(summary_data$mean_feature[summary_data$group == "CTRL"], na.rm=TRUE),
    SD_VPA = sd(summary_data$mean_feature[summary_data$group == "VPA"], na.rm=TRUE),
    t_value = t_test_result$statistic,
    df = t_test_result$parameter,
    p_value = t_test_result$p.value,
    CI_lower = t_test_result$conf.int[1],
    CI_upper = t_test_result$conf.int[2],
    Significance = get_stars(t_test_result$p.value)
  ))
  
  wilcoxon_results <- rbind(wilcoxon_results, data.frame(
    Feature = feature,
    Median_CTRL = median(summary_data$mean_feature[summary_data$group == "CTRL"], na.rm=TRUE),
    Median_VPA = median(summary_data$mean_feature[summary_data$group == "VPA"], na.rm=TRUE),
    W_value = wilcoxon_result$statistic,
    p_value = wilcoxon_result$p.value,
    Significance = get_stars(wilcoxon_result$p.value)
  ))
  
  lmm_results <- rbind(lmm_results, data.frame(
    Feature = feature,
    Formula = paste(feature, "~ group + (1 | id)"),
    Estimate = model_estimate,
    SE = model_se,
    t_value = model_t,
    p_value = model_p,
    AIC = model_AIC,
    Significance = get_stars(model_p)
  ))
  
  # ---- PLOT ----
  fill_col <- c("CTRL" = "lightgray", "VPA" = "cornflowerblue")
  point_col <- c("CTRL" = "#636363", "VPA" = "deepskyblue4")
  
  # FDR-corrected p-value per questa feature (t-test)
  p_fdr <- p.adjust(t_test_result$p.value, method = "fdr", n = length(features))
  significance_label <- get_stars(p_fdr)
  
  y_max <- max(summary_plot$ci_high, na.rm = TRUE)
  y_sig <- y_max * 1.10
  pretty_label <- prettify_feature(feature)
  
  # PLOT con IC
  p <- ggplot(summary_plot, aes(x = group, y = mean_group, fill = group)) +
    geom_bar(stat = "identity", width = 0.35, alpha = 0.85) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.12, linewidth = 0.5) +
    geom_jitter(data = summary_data, aes(x = group, y = mean_feature, color = group),
                width = 0.10, size = 0.6, alpha = 0.95, inherit.aes = FALSE) +
    scale_fill_manual(values = fill_col) +
    scale_color_manual(values = point_col) +
    labs(x = "Group", y = paste(pretty_label,"Average"), 
         title = paste("Mean", pretty_label, "by Group (±95% CI)")) +
    theme_minimal(base_size = 7) +
    theme(legend.position = "none",
          plot.title = element_text(size = 6, hjust = 0.5, face="bold"),
          axis.title.x = element_text(size = 5, face="bold"),
          axis.title.y = element_text(size = 5, face="bold"),
          axis.text = element_text(size = 4)) +
    # Add significance annotation if present
    {if (significance_label != "") geom_text(aes(x = 1.5, y = y_sig, label = significance_label), size = 4.5, vjust = 0)}
  
  ggsave(p, filename = file.path(output_plots_folder, paste0("plot_", feature, ".png")),
         width = 2.1, height = 2.3, dpi = 350)
}

# Apply False Discovery Rate correction to all p-values and assign significance stars
t_test_results$p_value_fdr <- p.adjust(t_test_results$p_value, method = "fdr")
t_test_results$Significance_fdr <- sapply(t_test_results$p_value_fdr, get_stars)

wilcoxon_results$p_value_fdr <- p.adjust(wilcoxon_results$p_value, method = "fdr")
wilcoxon_results$Significance_fdr <- sapply(wilcoxon_results$p_value_fdr, get_stars)

lmm_results$p_value_fdr <- p.adjust(lmm_results$p_value, method = "fdr")
lmm_results$Significance_fdr <- sapply(lmm_results$p_value_fdr, get_stars)

# Export results to CSV for further reporting or publication
write.csv(t_test_results, file.path(output_stats_folder, "t_test_results.csv"), row.names = FALSE)
write.csv(wilcoxon_results, file.path(output_stats_folder, "wilcoxon_results.csv"), row.names = FALSE)
write.csv(lmm_results, file.path(output_stats_folder, "lmm_results.csv"), row.names = FALSE)

# Export each results table to LaTeX for direct inclusion in a thesis or manuscript
print(xtable(t_test_results), type = "latex", file = file.path(output_stats_folder, "t_test_results.tex"), include.rownames = FALSE)
print(xtable(wilcoxon_results), type = "latex", file = file.path(output_stats_folder, "wilcoxon_results.tex"), include.rownames = FALSE)
print(xtable(lmm_results), type = "latex", file = file.path(output_stats_folder, "lmm_results.tex"), include.rownames = FALSE)

print("Analysis, plotting and LaTeX export completed. All results and plots saved in Results_univariate_analysis_46.")