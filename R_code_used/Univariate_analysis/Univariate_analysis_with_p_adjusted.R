# Load required libraries
library(readxl)
library(ggplot2)
library(lme4)
library(readr)
library(dplyr)
library(car)  # Per test diagnostici

# Clear the workspace
rm(list = ls())

# Set base and output folders
base_path <- "C:/Users/anton/VPA_vocalisations_project"
output_stats_folder <- file.path(base_path, "Results_univariate_analysis_46")
if (!dir.exists(output_stats_folder)) dir.create(output_stats_folder, recursive = TRUE)

# Function to assign significance stars
get_stars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) "***"
  else if (p < 0.01) "**"
  else if (p < 0.05) "*"
  else ""
}

# Function to calculate effect size (Cohen's d)
cohens_d <- function(x, y) {
  pooled_sd <- sqrt(((length(x) - 1) * var(x, na.rm = TRUE) + 
                       (length(y) - 1) * var(y, na.rm = TRUE)) / 
                      (length(x) + length(y) - 2))
  d <- (mean(y, na.rm = TRUE) - mean(x, na.rm = TRUE)) / pooled_sd
  return(d)
}

# Import and prepare data
ctrl_membership <- read_csv(file.path(base_path, "Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv"), show_col_types = FALSE) %>%
  mutate(condition = "CTRL")

vpa_membership <- read_csv(file.path(base_path, "Results_Clustering/hierarchical_clustering_vpa_46_3_clusters/hierarchical_clustering_3_distance_membership.csv"), show_col_types = FALSE) %>%
  mutate(condition = "VPA")

combined_membership <- bind_rows(ctrl_membership, vpa_membership)

# Clean column names
clean_names <- function(x) {
  x %>% gsub(" ", "_", .) %>% gsub("-", "_", .)
}
colnames(combined_membership) <- clean_names(colnames(combined_membership))

# Prepare data
t <- combined_membership %>%
  rename(id = recording, group = condition) %>%
  mutate(group = as.factor(group))

# Data summary
cat("Dataset Summary:\n")
cat("Total calls:", nrow(t), "\n")
cat("Subjects per group:\n")
print(table(t$group))

calls_per_subject <- t %>% 
  group_by(id, group) %>% 
  summarise(n_calls = n(), .groups = "drop")
cat("\nCalls per subject (min, median, max):\n")
print(summary(calls_per_subject$n_calls))

# Features to analyze
features <- c("Duration_call", "F0_Mean", "F0_Std", "F0_Skewness", "F0_Kurtosis",            
              "F0_Bandwidth", "F0_1st_Order_Diff", "F0_Slope", "F0_Mag_Mean",            
              "F1_Mag_Mean", "F2_Mag_Mean", "F1_F0_Ratio", "F2_F0_Ratio",            
              "Spectral_Centroid_Mean", "Spectral_Centroid_Std", "RMS_Mean", "RMS_Std",                
              "Slope", "Attack_magnitude", "Attack_time")

# Initialize results dataframe with comprehensive metrics
statistical_results <- tibble(
  Feature = character(),
  # Sample sizes
  n_ctrl_subjects = numeric(),
  n_vpa_subjects = numeric(),
  n_ctrl_calls = numeric(),
  n_vpa_calls = numeric(),
  # Descriptive statistics
  ctrl_mean = numeric(),
  vpa_mean = numeric(),
  ctrl_sd = numeric(),
  vpa_sd = numeric(),
  # Test results on aggregated data (subject-level means)
  t_test_t = numeric(),
  t_test_df = numeric(),
  t_test_p = numeric(),
  t_test_effect_size = numeric(),
  t_test_ci_lower = numeric(),
  t_test_ci_upper = numeric(),
  wilcoxon_p = numeric(),
  # Mixed-effects model results (call-level data)
  lmer_estimate = numeric(),
  lmer_se = numeric(),
  lmer_t = numeric(),
  lmer_p = numeric(),
  lmer_AIC = numeric(),
  # ICC and variance components
  icc = numeric(),
  between_subject_var = numeric(),
  within_subject_var = numeric(),
  # Diagnostic flags
  normality_assumption = character(),
  variance_assumption = character()
)

# Analysis loop
for (feature in features) {
  if (!(feature %in% colnames(t))) {
    message("Feature '", feature, "' not found. Skipping...")
    next
  }
  
  message("Processing: ", feature)
  
  # Remove missing values for this feature
  t_clean <- t %>% filter(!is.na(.data[[feature]]))
  
  # Aggregate data by subject (for t-test and descriptive stats)
  subject_summary <- t_clean %>%
    group_by(id, group) %>%
    summarise(
      mean_feature = mean(.data[[feature]], na.rm = TRUE),
      n_calls = n(),
      .groups = "drop"
    ) %>%
    filter(!is.na(mean_feature))
  
  # Separate groups for descriptive stats
  ctrl_subjects <- subject_summary %>% filter(group == "CTRL")
  vpa_subjects <- subject_summary %>% filter(group == "VPA")
  
  # Skip if insufficient data
  if (nrow(ctrl_subjects) < 3 || nrow(vpa_subjects) < 3) {
    message("  Insufficient subjects. Skipping...")
    next
  }
  
  # Descriptive statistics
  ctrl_mean <- mean(ctrl_subjects$mean_feature)
  vpa_mean <- mean(vpa_subjects$mean_feature)
  ctrl_sd <- sd(ctrl_subjects$mean_feature)
  vpa_sd <- sd(vpa_subjects$mean_feature)
  
  # Sample sizes
  n_ctrl_subjects <- nrow(ctrl_subjects)
  n_vpa_subjects <- nrow(vpa_subjects)
  n_ctrl_calls <- sum(ctrl_subjects$n_calls)
  n_vpa_calls <- sum(vpa_subjects$n_calls)
  
  # Test assumptions
  # Normality test (on subject-level means)
  shapiro_ctrl <- if(n_ctrl_subjects >= 3 && n_ctrl_subjects <= 5000) {
    shapiro.test(ctrl_subjects$mean_feature)$p.value > 0.05
  } else NA
  
  shapiro_vpa <- if(n_vpa_subjects >= 3 && n_vpa_subjects <= 5000) {
    shapiro.test(vpa_subjects$mean_feature)$p.value > 0.05
  } else NA
  
  normality_ok <- if(is.na(shapiro_ctrl) || is.na(shapiro_vpa)) {
    "Not tested"
  } else if(shapiro_ctrl && shapiro_vpa) {
    "OK"
  } else {
    "Violated"
  }
  
  # Variance homogeneity test
  levene_p <- tryCatch({
    leveneTest(mean_feature ~ group, data = subject_summary)$`Pr(>F)`[1]
  }, error = function(e) NA)
  
  variance_ok <- if(is.na(levene_p)) {
    "Not tested"
  } else if(levene_p > 0.05) {
    "OK"
  } else {
    "Violated"
  }
  
  # T-test on subject-level means
  t_test_result <- t.test(mean_feature ~ group, data = subject_summary, 
                          var.equal = (variance_ok == "OK"))
  
  # Effect size
  effect_size <- cohens_d(ctrl_subjects$mean_feature, vpa_subjects$mean_feature)
  
  # Wilcoxon test on subject-level means
  wilcoxon_result <- tryCatch({
    wilcox.test(mean_feature ~ group, data = subject_summary, exact = FALSE)
  }, error = function(e) list(p.value = NA))
  
  # Mixed-effects model on call-level data
  tryCatch({
    lmer_model <- lmer(as.formula(paste(feature, "~ group + (1 | id)")), 
                       data = t_clean, REML = FALSE)
    lmer_null <- lmer(as.formula(paste(feature, "~ (1 | id)")), 
                      data = t_clean, REML = FALSE)
    
    # Likelihood ratio test
    lrt_result <- anova(lmer_model, lmer_null)
    lmer_summary <- summary(lmer_model)
    
    # Variance components and ICC
    var_comp <- as.data.frame(VarCorr(lmer_model))
    between_var <- var_comp$vcov[1]  # Random intercept variance
    within_var <- var_comp$vcov[2]   # Residual variance
    icc_val <- between_var / (between_var + within_var)
    
    # Store results
    statistical_results <- statistical_results %>%
      add_row(
        Feature = feature,
        n_ctrl_subjects = n_ctrl_subjects,
        n_vpa_subjects = n_vpa_subjects,
        n_ctrl_calls = n_ctrl_calls,
        n_vpa_calls = n_vpa_calls,
        ctrl_mean = ctrl_mean,
        vpa_mean = vpa_mean,
        ctrl_sd = ctrl_sd,
        vpa_sd = vpa_sd,
        t_test_t = t_test_result$statistic,
        t_test_df = t_test_result$parameter,
        t_test_p = t_test_result$p.value,
        t_test_effect_size = effect_size,
        t_test_ci_lower = t_test_result$conf.int[1],
        t_test_ci_upper = t_test_result$conf.int[2],
        wilcoxon_p = wilcoxon_result$p.value,
        lmer_estimate = lmer_summary$coefficients[2, "Estimate"],
        lmer_se = lmer_summary$coefficients[2, "Std. Error"],
        lmer_t = lmer_summary$coefficients[2, "t value"],
        lmer_p = lrt_result$`Pr(>Chisq)`[2],
        lmer_AIC = AIC(lmer_model),
        icc = icc_val,
        between_subject_var = between_var,
        within_subject_var = within_var,
        normality_assumption = normality_ok,
        variance_assumption = variance_ok
      )
    
  }, error = function(e) {
    message("  Error in mixed-effects model: ", e$message)
    
    # Store partial results (without mixed-effects)
    statistical_results <<- statistical_results %>%
      add_row(
        Feature = feature,
        n_ctrl_subjects = n_ctrl_subjects,
        n_vpa_subjects = n_vpa_subjects,
        n_ctrl_calls = n_ctrl_calls,
        n_vpa_calls = n_vpa_calls,
        ctrl_mean = ctrl_mean,
        vpa_mean = vpa_mean,
        ctrl_sd = ctrl_sd,
        vpa_sd = vpa_sd,
        t_test_t = t_test_result$statistic,
        t_test_df = t_test_result$parameter,
        t_test_p = t_test_result$p.value,
        t_test_effect_size = effect_size,
        t_test_ci_lower = t_test_result$conf.int[1],
        t_test_ci_upper = t_test_result$conf.int[2],
        wilcoxon_p = wilcoxon_result$p.value,
        lmer_estimate = NA,
        lmer_se = NA,
        lmer_t = NA,
        lmer_p = NA,
        lmer_AIC = NA,
        icc = NA,
        between_subject_var = NA,
        within_subject_var = NA,
        normality_assumption = normality_ok,
        variance_assumption = variance_ok
      )
  })
}

# Add significance indicators (raw p-values)
statistical_results <- statistical_results %>%
  mutate(
    t_test_sig = sapply(t_test_p, get_stars),
    wilcoxon_sig = sapply(wilcoxon_p, get_stars),
    lmer_sig = sapply(lmer_p, get_stars)
  )

# Multiple testing correction (FDR)
statistical_results <- statistical_results %>%
  mutate(
    t_test_p_adj = p.adjust(t_test_p, method = "fdr"),
    wilcoxon_p_adj = p.adjust(wilcoxon_p, method = "fdr"),
    lmer_p_adj = p.adjust(lmer_p, method = "fdr"),
    # Adjusted significance indicators
    t_test_sig_adj = sapply(t_test_p_adj, get_stars),
    wilcoxon_sig_adj = sapply(wilcoxon_p_adj, get_stars),
    lmer_sig_adj = sapply(lmer_p_adj, get_stars)
  )

# Export results
write.csv(statistical_results, 
          file.path(output_stats_folder, "comprehensive_statistical_results.csv"), 
          row.names = FALSE)

# Summary of results
cat("\n=== ANALYSIS SUMMARY ===\n")
cat("Features analyzed:", nrow(statistical_results), "\n")
cat("Significant t-tests (raw p < 0.05):", sum(statistical_results$t_test_p < 0.05, na.rm = TRUE), "\n")
cat("Significant t-tests (FDR adjusted):", sum(statistical_results$t_test_p_adj < 0.05, na.rm = TRUE), "\n")
cat("Significant mixed-effects (raw p < 0.05):", sum(statistical_results$lmer_p < 0.05, na.rm = TRUE), "\n")
cat("Significant mixed-effects (FDR adjusted):", sum(statistical_results$lmer_p_adj < 0.05, na.rm = TRUE), "\n")
cat("Average ICC:", round(mean(statistical_results$icc, na.rm = TRUE), 3), "\n")
cat("Average effect size:", round(mean(abs(statistical_results$t_test_effect_size), na.rm = TRUE), 3), "\n")

# Show significant results (FDR corrected)
significant_results <- statistical_results %>%
  filter(t_test_p_adj < 0.05 | lmer_p_adj < 0.05) %>%
  select(Feature, t_test_p_adj, lmer_p_adj, t_test_effect_size, icc, 
         normality_assumption, variance_assumption)

if (nrow(significant_results) > 0) {
  cat("\nSignificant features (FDR corrected):\n")
  print(significant_results)
} else {
  cat("\nNo features remain significant after FDR correction.\n")
}

message("Analysis completed successfully!")
