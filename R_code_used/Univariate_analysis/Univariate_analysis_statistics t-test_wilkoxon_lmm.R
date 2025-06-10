# Load required libraries
library(readxl)
library(ggplot2)
library(plyr)
library(lme4)
library(readr)
library(dplyr)

# Clear the workspace
rm(list = ls())

# Set base and output folders
base_path <- "C:/Users/anton/VPA_vocalisations_project"
output_stats_folder <- file.path(base_path, "Results_univariate_analysis_46")
if (!dir.exists(output_stats_folder)) dir.create(output_stats_folder, recursive = TRUE)

# Function to assign significance stars
get_stars <- function(p) {
  if (p < 0.001) "***"
  else if (p < 0.01) "**"
  else if (p < 0.05) "*"
  else ""
}

# Import clustering membership data and add condition labels
ctrl_membership <- read_csv(file.path(base_path, "Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv"), show_col_types = FALSE) %>%
  mutate(condition = "CTRL")

vpa_membership <- read_csv(file.path(base_path, "Results_Clustering/hierarchical_clustering_vpa_46_3_clusters/hierarchical_clustering_3_distance_membership.csv"), show_col_types = FALSE) %>%
  mutate(condition = "VPA")

# Combine datasets
combined_membership <- bind_rows(ctrl_membership, vpa_membership)

names(combined_membership)

# Clean column names (replace spaces and hyphens with underscores)
clean_names <- function(x) {
  x %>%
    gsub(" ", "_", .) %>%
    gsub("-", "_", .)
}
colnames(combined_membership) <- clean_names(colnames(combined_membership))

# Save the combined dataset for future use
output_data_folder <- file.path(base_path, "Results_R", "Features_univariate_analysis_46")
if (!dir.exists(output_data_folder)) dir.create(output_data_folder, recursive = TRUE)
write.csv(combined_membership, file.path(output_data_folder, "combined_dataset.csv"), row.names = FALSE)

# Prepare data for analysis
t <- combined_membership %>%
  rename(id = recording, group = condition)

names(t)

# Ensure group is a factor
t$group <- as.factor(t$group)

View(t)
# List of features to analyse
features <- c("Duration_call", "F0_Mean", "F0_Std", "F0_Skewness", "F0_Kurtosis",            
              "F0_Bandwidth", "F0_1st_Order_Diff", "F0_Slope", "F0_Mag_Mean",            
              "F1_Mag_Mean", "F2_Mag_Mean", "F1_F0_Ratio", "F2_F0_Ratio",            
              "Spectral_Centroid_Mean", "Spectral_Centroid_Std", "RMS_Mean", "RMS_Std",                
              "Slope", "Attack_magnitude", "Attack_time")

# Initialise the results dataframe
statistical_results <- tibble(
  Feature = character(),
  t_test_t = numeric(),
  t_test_df = numeric(),
  t_test_p = numeric(),
  t_test_ci_lower = numeric(),
  t_test_ci_upper = numeric(),
  t_test_ctrl_mean = numeric(),
  t_test_vpa_mean = numeric(),
  t_test_sig = character(),
  wilcoxon_p = numeric(),
  wilcoxon_sig = character(),
  model_estimate = numeric(),
  model_se = numeric(),
  model_t = numeric(),
  model_p = numeric(),
  model_sig = character(),
  model_AIC = numeric()
)

for (feature in features) {
  if (!(feature %in% colnames(t))) {
    message("Feature '", feature, "' not found in dataset. Skipping...")
    next
  }
  message("Processing: ", feature)
  
  # Summarise by id and group (mean per chick per group)
  summary_data_reg <- t %>%
    group_by(id, group) %>%
    summarise(
      mean_feature = mean(.data[[feature]], na.rm = TRUE),
      se_feature = sd(.data[[feature]], na.rm = TRUE) / sqrt(sum(!is.na(.data[[feature]]))),
      N = sum(!is.na(.data[[feature]])),
      .groups = "drop"
    )
  
  # Two-sample t-test (assuming equal variance not guaranteed)
  t_test_result <- t.test(mean_feature ~ group, data = summary_data_reg)
  
  # Wilcoxon test
  wilcoxon_result <- wilcox.test(mean_feature ~ group, data = summary_data_reg, exact = FALSE)
  
  # Mixed-effects model (random intercept for individual chick)
  mod <- lmer(as.formula(paste(feature, "~ group + (1 | id)")), data = t, REML = FALSE)
  mod_null <- lmer(as.formula(paste(feature, "~ (1 | id)")), data = t, REML = FALSE)
  anova_result <- anova(mod, mod_null)
  model_summary <- summary(mod)
  
  # Store results
  statistical_results <- statistical_results %>%
    add_row(
      Feature = feature,
      t_test_t = t_test_result$statistic,
      t_test_df = t_test_result$parameter,
      t_test_p = t_test_result$p.value,
      t_test_ci_lower = t_test_result$conf.int[1],
      t_test_ci_upper = t_test_result$conf.int[2],
      t_test_ctrl_mean = t_test_result$estimate[1],
      t_test_vpa_mean = t_test_result$estimate[2],
      t_test_sig = get_stars(t_test_result$p.value),
      wilcoxon_p = wilcoxon_result$p.value,
      wilcoxon_sig = get_stars(wilcoxon_result$p.value),
      model_estimate = model_summary$coefficients[2, "Estimate"],
      model_se = model_summary$coefficients[2, "Std. Error"],
      model_t = model_summary$coefficients[2, "t value"],
      model_p = anova_result$`Pr(>Chisq)`[2],
      model_sig = get_stars(anova_result$`Pr(>Chisq)`[2]),
      model_AIC = AIC(mod)
    )
}

# Export statistical results
write.csv(statistical_results, file.path(output_stats_folder, "statistical_results.csv"), row.names = FALSE)


# Correction for multiple testing (FDR)
statistical_results <- statistical_results %>%
  mutate(
    t_test_p_adj = p.adjust(t_test_p, method = "fdr"),
    wilcoxon_p_adj = p.adjust(wilcoxon_p, method = "fdr"),
    model_p_adj = p.adjust(model_p, method = "fdr"),
    t_test_sig_adj = sapply(t_test_p_adj, get_stars),
    wilcoxon_sig_adj = sapply(wilcoxon_p_adj, get_stars),
    model_sig_adj = sapply(model_p_adj, get_stars)
  )

# Export statistical results
write.csv(statistical_results, file.path(output_stats_folder, "statistical_results_corrected.csv"), row.names = FALSE)
