# -------------------------------------
# Mixed-Effects and Aggregated ANOVA Analysis of Acoustic Features 
# -------------------------------------
rm(list = ls())
library(dplyr)
library(readr)
library(lme4)
library(lmerTest)
library(multcomp)
library(xtable)
library(plyr)   # For ddply

# Function for significance stars
get_stars <- function(p) {
  if (p < 0.001) return("***")
  else if (p < 0.01) return("**")
  else if (p < 0.05) return("*")
  else return("")
}

# Clean workspace and set output folder
rm(list = ls())
setwd("C:/Users/anton/VPA_vocalisations_project/Results_Analysis/Statistical_plots_between")

output_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis_new/Statistical_plots_46_files_bt3_condition_anova"
if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)

# Import and combine datasets
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv") %>%
  mutate(condition = "CTRL")
vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_46_3_clusters/hierarchical_clustering_3_distance_membership.csv") %>%
  mutate(condition = "VPA")

combined_membership <- bind_rows(ctrl_membership, vpa_membership) %>%
  mutate(
    id = factor(recording),
    condition = factor(condition),
    cluster_membership = factor(cluster_membership)
  )

cols_to_remove <- c("onsets_sec", "offsets_sec", "recording", "call_id", "distance_to_center")
combined_membership <- combined_membership[ , !(names(combined_membership) %in% cols_to_remove)]


names(combined_membership)

features <- c("Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis",
              "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean",
              "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio",
              "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
              "Slope", "Attack_magnitude", "Attack_time")



# Initialise results dataframes
anova_mixed_df <- data.frame()
mixed_results_df <- data.frame()
anova_agg_df <- data.frame()
tukey_agg_df <- data.frame()

# Loop over each feature and run both analyses
for (feature in features) {
  if (!(feature %in% colnames(combined_membership))) {
    cat("Feature", feature, "not found. Skipping...\n")
    next
  }
  cat("Analysing feature:", feature, "\n")
  
  # --- Mixed-effects model (random effect for ID) ---
  mixed_formula <- as.formula(paste0("`", feature, "` ~ condition * cluster_membership + (1 | id)"))
  mixed_model <- lmer(mixed_formula, data = combined_membership)
  anova_tab <- anova(mixed_model, type = 3)
  anova_results <- data.frame(
    Feature = feature,
    Term = rownames(anova_tab),
    F_value = anova_tab$`F value`,
    p_value = anova_tab$`Pr(>F)`
  )
  anova_mixed_df <- bind_rows(anova_mixed_df, anova_results)
  
  fixed_effects <- summary(mixed_model)$coefficients
  fixed_results <- data.frame(
    Feature = feature,
    Term = rownames(fixed_effects),
    Estimate = fixed_effects[, "Estimate"],
    Std_Error = fixed_effects[, "Std. Error"],
    t_value = fixed_effects[, "t value"],
    p_value = fixed_effects[, "Pr(>|t|)"]
  )
  mixed_results_df <- bind_rows(mixed_results_df, fixed_results)
  
  # --- Aggregated data by id, condition and cluster_membership ---
  summary_stats <- ddply(combined_membership, .(id, condition, cluster_membership), summarise,
                         mean_value = mean(get(feature), na.rm = TRUE),
                         sem_value = sd(get(feature), na.rm = TRUE) / sqrt(length(na.omit(get(feature)))),
                         n = length(na.omit(get(feature))))
  
  # Classic ANOVA on aggregated data
  aov_formula <- as.formula("mean_value ~ condition * cluster_membership")
  aov_model <- aov(aov_formula, data = summary_stats)
  aov_tab <- summary(aov_model)[[1]]
  
  # Calculate eta squared for each term (no DescTools needed)
  ss_total <- sum(aov_tab$"Sum Sq", na.rm = TRUE)
  eta2 <- aov_tab$"Sum Sq" / ss_total
  
  anova_agg_out <- data.frame(
    Feature = feature,
    Term = rownames(aov_tab),
    Df = aov_tab$Df,
    Sum_Sq = aov_tab$"Sum Sq",
    Mean_Sq = aov_tab$"Mean Sq",
    F_value = aov_tab$"F value",
    p_value = aov_tab$"Pr(>F)",
    eta2 = eta2
  )
  anova_agg_df <- bind_rows(anova_agg_df, anova_agg_out)
  
  
  # Tukey HSD post-hoc test on aggregated data
  tukey_result <- TukeyHSD(aov_model, "cluster_membership", conf.level = 0.95)
  tukey_df <- as.data.frame(tukey_result$cluster_membership)
  tukey_df$Feature <- feature
  tukey_df$Comparison <- rownames(tukey_df)
  tukey_df$Significance <- sapply(tukey_df$`p adj`, get_stars)
  # Reorder and select relevant columns for output
  tukey_df <- tukey_df[, c("diff", "lwr", "upr", "p adj", "Comparison", "Feature", "Significance")]
  tukey_agg_df <- bind_rows(tukey_agg_df, tukey_df)
}

# Save all results as CSV
write.csv(anova_mixed_df, file.path(output_folder, "anova_results_mixed.csv"), row.names = FALSE)
write.csv(mixed_results_df, file.path(output_folder, "mixed_results.csv"), row.names = FALSE)
write.csv(anova_agg_df, file.path(output_folder, "anova_results_aggregated.csv"), row.names = FALSE)
write.csv(tukey_agg_df, file.path(output_folder, "tukey_results_aggregated.csv"), row.names = FALSE)

# Export as LaTeX tables
print(xtable(anova_mixed_df), include.rownames = FALSE, file = file.path(output_folder, "anova_results_mixed.tex"))
print(xtable(mixed_results_df), include.rownames = FALSE, file = file.path(output_folder, "mixed_results.tex"))
print(xtable(anova_agg_df), include.rownames = FALSE, file = file.path(output_folder, "anova_results_aggregated.tex"))
print(xtable(tukey_agg_df), include.rownames = FALSE, file = file.path(output_folder, "tukey_results_aggregated.tex"))

cat("Analysis completed and results saved!\n")
