# ================================================================================
# CONSOLIDATED BETWEEN-CLUSTERS ANALYSIS: VPA vs CTRL
# Acoustic feature analysis using aggregated data approach
# ================================================================================

# STEP 1: SETUP AND LIBRARIES ====================================================
rm(list = ls())

# Load required libraries
library(dplyr)         # Data manipulation
library(readr)         # CSV import
library(multcomp)      # Post-hoc tests (Tukey)
library(xtable)        # Export tables to LaTeX
library(plyr)          # For ddply aggregation

# Function for significance stars
get_stars <- function(p) {
  if (p < 0.001) return("***")
  else if (p < 0.01) return("**")
  else if (p < 0.05) return("*")
  else return("")
}

# Set working directory
setwd("C:/Users/anton/VPA_vocalisations_project/Results_Analysis/Statistical_plots_between")

# Create output folder
output_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Consolidated_Analysis_Results"
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# STEP 2: DATA IMPORT AND PREPARATION ============================================

cat("=== STEP 2: Data import ===\n")

# Import cluster membership data
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv") %>%
  mutate(condition = "CTRL")

vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_46_3_clusters/hierarchical_clustering_3_distance_membership.csv") %>%
  mutate(condition = "VPA")

# Combine datasets
combined_membership <- bind_rows(ctrl_membership, vpa_membership) %>%
  mutate(
    id = factor(recording),
    condition = factor(condition),
    cluster_membership = factor(cluster_membership)
  )

# Remove unwanted columns
cols_to_remove <- c("onsets_sec", "offsets_sec", "recording", "call_id", "distance_to_center")
combined_membership <- combined_membership[, !(names(combined_membership) %in% cols_to_remove)]

# Display dataset summary
cat("Dataset structure:\n")
cat("- Total observations: ", nrow(combined_membership), "\n")
cat("- Number of individuals: ", n_distinct(combined_membership$id), "\n")
cat("- Distribution by group: \n"); print(table(combined_membership$condition))
cat("- Distribution by cluster: \n"); print(table(combined_membership$cluster_membership))
cat("- Contingency table:\n")
print(table(combined_membership$condition, combined_membership$cluster_membership))

# STEP 3: DEFINE ACOUSTIC FEATURES ==============================================

features <- c(
  "Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis",
  "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean",
  "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio",
  "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
  "Slope", "Attack_magnitude", "Attack_time"
)

# Check missing features
missing_features <- features[!features %in% colnames(combined_membership)]
if (length(missing_features) > 0) {
  cat("WARNING: Missing features: ", paste(missing_features, collapse = ", "), "\n")
  features <- features[features %in% colnames(combined_membership)]
}
cat("Features to analyse: ", length(features), "\n")

# STEP 4: INITIALISE RESULTS STORAGE ============================================

results_summary <- data.frame()      # Overall summary for each feature
anova_results <- data.frame()        # ANOVA results
tukey_results <- data.frame()        # Tukey post-hoc results
glm_results <- data.frame()          # GLM coefficients

# STEP 5: MAIN ANALYSIS LOOP ====================================================

cat("\n=== STEP 5: Feature analysis using aggregated data ===\n")

for (i in seq_along(features)) {
  feature <- features[i]
  
  cat(sprintf("Analysing feature %d/%d: %s\n", i, length(features), feature))
  
  tryCatch({
    # ---- 5.1: AGGREGATE DATA BY ID, CONDITION AND CLUSTER ----
    # Calculate mean, SEM and n for each individual in each condition/cluster combination
    summary_stats <- ddply(combined_membership, .(id, condition, cluster_membership), 
                           summarise,
                           mean_value = mean(get(feature), na.rm = TRUE),
                           sem_value = sd(get(feature), na.rm = TRUE) / sqrt(length(na.omit(get(feature)))),
                           n = length(na.omit(get(feature))))
    
    # ---- 5.2: CLASSIC ANOVA ON AGGREGATED DATA ----
    aov_formula <- as.formula("mean_value ~ condition * cluster_membership")
    aov_model <- aov(aov_formula, data = summary_stats)
    aov_tab <- summary(aov_model)[[1]]
    
    # Calculate eta squared
    ss_total <- sum(aov_tab$"Sum Sq", na.rm = TRUE)
    eta2 <- aov_tab$"Sum Sq" / ss_total
    
    # Store ANOVA results
    anova_row <- data.frame(
      Feature = feature,
      Term = rownames(aov_tab),
      Df = aov_tab$Df,
      Sum_Sq = aov_tab$"Sum Sq",
      Mean_Sq = aov_tab$"Mean Sq",
      F_value = aov_tab$"F value",
      p_value = aov_tab$"Pr(>F)",
      eta2 = eta2,
      Significance = sapply(aov_tab$"Pr(>F)", get_stars)
    )
    anova_results <- rbind(anova_results, anova_row)
    
    # ---- 5.3: GLM COEFFICIENTS FROM ANOVA MODEL ----
    glm_summary <- summary.lm(aov_model)$coefficients
    glm_row <- data.frame(
      Feature = feature,
      Term = rownames(glm_summary),
      Estimate = glm_summary[, "Estimate"],
      Std_Error = glm_summary[, "Std. Error"],
      t_value = glm_summary[, "t value"],
      p_value = glm_summary[, "Pr(>|t|)"],
      Significance = sapply(glm_summary[, "Pr(>|t|)"], get_stars)
    )
    glm_results <- rbind(glm_results, glm_row)
    
    # ---- 5.4: TUKEY POST-HOC TESTS ----
    # Tukey HSD for cluster comparisons
    tukey_cluster <- TukeyHSD(aov_model, "cluster_membership", conf.level = 0.95)
    if (!is.null(tukey_cluster$cluster_membership)) {
      tukey_cluster_df <- as.data.frame(tukey_cluster$cluster_membership)
      tukey_cluster_df$Feature <- feature
      tukey_cluster_df$Test_Type <- "Cluster_Comparison"
      tukey_cluster_df$Comparison <- rownames(tukey_cluster_df)
      tukey_cluster_df$Significance <- sapply(tukey_cluster_df$`p adj`, get_stars)
      
      # Rename columns for consistency
      names(tukey_cluster_df)[names(tukey_cluster_df) == "diff"] <- "Estimate"
      names(tukey_cluster_df)[names(tukey_cluster_df) == "p adj"] <- "p_adjusted"
      
      tukey_results <- rbind(tukey_results, tukey_cluster_df[, c("Feature", "Test_Type", "Comparison", 
                                                                 "Estimate", "lwr", "upr", "p_adjusted", "Significance")])
    }
    
    # Tukey HSD for condition comparisons (if more than 2 levels)
    if (nlevels(summary_stats$condition) > 2) {
      tukey_condition <- TukeyHSD(aov_model, "condition", conf.level = 0.95)
      if (!is.null(tukey_condition$condition)) {
        tukey_condition_df <- as.data.frame(tukey_condition$condition)
        tukey_condition_df$Feature <- feature
        tukey_condition_df$Test_Type <- "Condition_Comparison"
        tukey_condition_df$Comparison <- rownames(tukey_condition_df)
        tukey_condition_df$Significance <- sapply(tukey_condition_df$`p adj`, get_stars)
        
        names(tukey_condition_df)[names(tukey_condition_df) == "diff"] <- "Estimate"
        names(tukey_condition_df)[names(tukey_condition_df) == "p adj"] <- "p_adjusted"
        
        tukey_results <- rbind(tukey_results, tukey_condition_df[, c("Feature", "Test_Type", "Comparison", 
                                                                     "Estimate", "lwr", "upr", "p_adjusted", "Significance")])
      }
    }
    
    # ---- 5.5: SUMMARY FOR CURRENT FEATURE ----
    # Extract main effects and interaction p-values
    condition_idx <- which(rownames(aov_tab) == "condition")
    cluster_idx <- which(rownames(aov_tab) == "cluster_membership")
    interaction_idx <- which(rownames(aov_tab) == "condition:cluster_membership")
    
    summary_row <- data.frame(
      Feature = feature,
      Condition_F = if(length(condition_idx) > 0) aov_tab[condition_idx, "F value"] else NA,
      Condition_p = if(length(condition_idx) > 0) aov_tab[condition_idx, "Pr(>F)"] else NA,
      Condition_Significant = if(length(condition_idx) > 0) aov_tab[condition_idx, "Pr(>F)"] < 0.05 else FALSE,
      Condition_eta2 = if(length(condition_idx) > 0) eta2[condition_idx] else NA,
      Cluster_F = if(length(cluster_idx) > 0) aov_tab[cluster_idx, "F value"] else NA,
      Cluster_p = if(length(cluster_idx) > 0) aov_tab[cluster_idx, "Pr(>F)"] else NA,
      Cluster_Significant = if(length(cluster_idx) > 0) aov_tab[cluster_idx, "Pr(>F)"] < 0.05 else FALSE,
      Cluster_eta2 = if(length(cluster_idx) > 0) eta2[cluster_idx] else NA,
      Interaction_F = if(length(interaction_idx) > 0) aov_tab[interaction_idx, "F value"] else NA,
      Interaction_p = if(length(interaction_idx) > 0) aov_tab[interaction_idx, "Pr(>F)"] else NA,
      Interaction_Significant = if(length(interaction_idx) > 0) aov_tab[interaction_idx, "Pr(>F)"] < 0.05 else FALSE,
      Interaction_eta2 = if(length(interaction_idx) > 0) eta2[interaction_idx] else NA
    )
    results_summary <- rbind(results_summary, summary_row)
    
  }, error = function(e) {
    cat("ERROR in analysis for", feature, ":", e$message, "\n")
  })
}

# STEP 6: MULTIPLE TESTING CORRECTION ==========================================

cat("\n=== STEP 6: Multiple testing correction ===\n")

# Apply FDR correction to main and interaction p-values
results_summary$Condition_p_FDR <- p.adjust(results_summary$Condition_p, method = "fdr")
results_summary$Cluster_p_FDR <- p.adjust(results_summary$Cluster_p, method = "fdr")
results_summary$Interaction_p_FDR <- p.adjust(results_summary$Interaction_p, method = "fdr")

# Update significance flags after FDR correction
results_summary$Condition_Significant_FDR <- results_summary$Condition_p_FDR < 0.05
results_summary$Cluster_Significant_FDR <- results_summary$Cluster_p_FDR < 0.05
results_summary$Interaction_Significant_FDR <- results_summary$Interaction_p_FDR < 0.05

# Add significance stars
results_summary$Condition_Stars <- sapply(results_summary$Condition_p_FDR, get_stars)
results_summary$Cluster_Stars <- sapply(results_summary$Cluster_p_FDR, get_stars)
results_summary$Interaction_Stars <- sapply(results_summary$Interaction_p_FDR, get_stars)

# STEP 7: SAVE ALL RESULTS ======================================================

cat("\n=== STEP 7: Saving results ===\n")

# Write results to CSV
write.csv(results_summary, file.path(output_folder, "01_Summary_Results.csv"), row.names = FALSE)
write.csv(anova_results, file.path(output_folder, "02_ANOVA_Results.csv"), row.names = FALSE)
write.csv(glm_results, file.path(output_folder, "03_GLM_Coefficients.csv"), row.names = FALSE)
write.csv(tukey_results, file.path(output_folder, "04_Tukey_PostHoc.csv"), row.names = FALSE)

# Export as LaTeX tables
print(xtable(results_summary), include.rownames = FALSE, 
      file = file.path(output_folder, "01_Summary_Results.tex"))
print(xtable(anova_results), include.rownames = FALSE, 
      file = file.path(output_folder, "02_ANOVA_Results.tex"))
print(xtable(glm_results), include.rownames = FALSE, 
      file = file.path(output_folder, "03_GLM_Coefficients.tex"))
print(xtable(tukey_results), include.rownames = FALSE, 
      file = file.path(output_folder, "04_Tukey_PostHoc.tex"))

# STEP 8: FINAL SUMMARY REPORT ==================================================

cat("\n=== STEP 8: Final report ===\n")

# Count significant findings
n_condition_sig <- sum(results_summary$Condition_Significant, na.rm = TRUE)
n_cluster_sig <- sum(results_summary$Cluster_Significant, na.rm = TRUE)
n_interaction_sig <- sum(results_summary$Interaction_Significant, na.rm = TRUE)

n_condition_sig_fdr <- sum(results_summary$Condition_Significant_FDR, na.rm = TRUE)
n_cluster_sig_fdr <- sum(results_summary$Cluster_Significant_FDR, na.rm = TRUE)
n_interaction_sig_fdr <- sum(results_summary$Interaction_Significant_FDR, na.rm = TRUE)

cat("FINAL RESULTS:\n")
cat("==================\n")
cat("Features analysed: ", nrow(results_summary), "\n")
cat("\nSignificant effects (p < 0.05):\n")
cat("- Condition (VPA vs CTRL): ", n_condition_sig, " / ", nrow(results_summary), "\n")
cat("- Cluster: ", n_cluster_sig, " / ", nrow(results_summary), "\n")
cat("- Interaction: ", n_interaction_sig, " / ", nrow(results_summary), "\n")
cat("\nAfter FDR correction:\n")
cat("- Condition: ", n_condition_sig_fdr, " / ", nrow(results_summary), "\n")
cat("- Cluster: ", n_cluster_sig_fdr, " / ", nrow(results_summary), "\n")
cat("- Interaction: ", n_interaction_sig_fdr, " / ", nrow(results_summary), "\n")

# List significant features after correction
if (n_condition_sig_fdr > 0) {
  sig_condition_features <- results_summary$Feature[results_summary$Condition_Significant_FDR]
  cat("\nFeatures with significant CONDITION effect (after FDR):\n")
  cat(paste(sig_condition_features, collapse = ", "), "\n")
}
if (n_cluster_sig_fdr > 0) {
  sig_cluster_features <- results_summary$Feature[results_summary$Cluster_Significant_FDR]
  cat("\nFeatures with significant CLUSTER effect (after FDR):\n")
  cat(paste(sig_cluster_features, collapse = ", "), "\n")
}
if (n_interaction_sig_fdr > 0) {
  sig_interaction_features <- results_summary$Feature[results_summary$Interaction_Significant_FDR]
  cat("\nFeatures with significant INTERACTION effect (after FDR):\n")
  cat(paste(sig_interaction_features, collapse = ", "), "\n")
}

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("All results saved in: ", output_folder, "\n")

# Show first rows of summary
cat("\nFirst 5 rows of the summary table:\n")
print(head(results_summary[, c("Feature", "Condition_p_FDR", "Cluster_p_FDR", "Interaction_p_FDR", 
                               "Condition_Significant_FDR", "Cluster_Significant_FDR")], 5))