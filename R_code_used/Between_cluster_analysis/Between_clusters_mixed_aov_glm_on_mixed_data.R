# ================================================================================
# BETWEEN-CLUSTERS ANALYSIS: VPA vs CTRL
# Acoustic feature analysis between groups and clusters
# ================================================================================

# STEP 1: SETUP AND LIBRARIES ====================================================
# Clear the workspace to ensure a clean analysis environment
rm(list = ls())

# Load all required libraries for statistical analysis and reporting
library(dplyr)         # Data manipulation (tidyverse)
library(readr)         # Fast and reliable CSV import
library(lme4)          # Linear mixed models
library(lmerTest)      # P-values and ANOVA tables for mixed models
library(multcomp)      # Post-hoc tests (Tukey, etc.)
library(xtable)        # Export tables to LaTeX format
library(effectsize)    # Calculation of effect sizes
library(broom.mixed)   # Tidying of mixed model outputs

# Set the working directory for reading/writing files
setwd("C:/Users/anton/VPA_vocalisations_project/Results_Analysis/Statistical_plots_between")

# Create output folder if it does not exist
output_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Unified_Analysis_Results"
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# STEP 2: DATA IMPORT AND PREPARATION ============================================

cat("=== STEP 2: Data import ===\n")

# Import the cluster membership data for CTRL group
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv") %>%
  mutate(condition = "CTRL")

# Import the cluster membership data for VPA group
vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_46_3_clusters/hierarchical_clustering_3_distance_membership.csv") %>%
  mutate(condition = "VPA")

# Combine both datasets and tidy factor/variable types
combined_membership <- bind_rows(ctrl_membership, vpa_membership) %>%
  mutate(
    id = factor(recording),                     # Unique individual identifier
    condition = factor(condition),              # Experimental group: CTRL or VPA
    cluster_membership = factor(cluster_membership) # Cluster assignment (1, 2, or 3)
  )

# Remove unwanted columns not needed for statistical analysis
cols_to_remove <- c("onsets_sec", "offsets_sec", "recording", "call_id", "distance_to_center")
combined_membership <- combined_membership[ , !(names(combined_membership) %in% cols_to_remove)]

# Display structure and key summaries for quality control
cat("Dataset structure:\n")
cat("- Total observations: ", nrow(combined_membership), "\n")
cat("- Number of individuals: ", n_distinct(combined_membership$id), "\n")
cat("- Distribution by group: \n"); print(table(combined_membership$condition))
cat("- Distribution by cluster: \n"); print(table(combined_membership$cluster_membership))
cat("- Contingency table:\n")
print(table(combined_membership$condition, combined_membership$cluster_membership))

# STEP 3: DEFINE ACOUSTIC FEATURES ==============================================

# List all acoustic features to analyse (as per your protocol)
features <- c(
  "Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis",
  "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean",
  "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio",
  "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
  "Slope", "Attack_magnitude", "Attack_time"
)

# Check if any requested features are missing from the dataset
missing_features <- features[!features %in% colnames(combined_membership)]
if (length(missing_features) > 0) {
  cat("WARNING: Missing features: ", paste(missing_features, collapse = ", "), "\n")
  features <- features[features %in% colnames(combined_membership)] # Only analyse present features
}
cat("Features to analyse: ", length(features), "\n")

# STEP 4: INITIALISE RESULTS STORAGE ============================================

# Data frames to save results at each stage
results_summary   <- data.frame()      # Overall summary for each feature
anova_results     <- data.frame()      # Full ANOVA tables
glm_coefficients  <- data.frame()      # Model coefficients
tukey_results     <- data.frame()      # Post-hoc test results (Tukey)
effect_sizes      <- data.frame()      # Effect sizes

# STEP 5: MAIN ANALYSIS LOOP ====================================================

cat("\n=== STEP 5: Feature analysis ===\n")

for (i in seq_along(features)) {
  feature <- features[i]
  
  cat(sprintf("Analysing feature %d/%d: %s\n", i, length(features), feature))
  
  tryCatch({
    # ---- 5.1: FIT MIXED MODEL ----
    # Linear mixed model: feature ~ condition * cluster_membership + (1 | id)
    # Random intercept for individual to account for repeated measures
    formula_mixed <- as.formula(paste0("`", feature, "` ~ condition * cluster_membership + (1 | id)"))
    mixed_model <- lmer(formula_mixed, data = combined_membership)
    
    # ---- 5.2: ANOVA ON MIXED MODEL ----
    # Type III sums of squares for accurate main effect and interaction inference
    anova_table <- anova(mixed_model, type = 3)
    
    # Save ANOVA results
    anova_row <- data.frame(
      Feature  = feature,
      Term     = rownames(anova_table),
      Sum_Sq   = anova_table$`Sum Sq`,
      Mean_Sq  = anova_table$`Mean Sq`,
      NumDF    = anova_table$NumDF,
      DenDF    = anova_table$DenDF,
      F_value  = anova_table$`F value`,
      p_value  = anova_table$`Pr(>F)`
    )
    anova_results <- rbind(anova_results, anova_row)
    
    # ---- 5.3: EXTRACT GLM COEFFICIENTS ----
    glm_summary <- summary(mixed_model)$coefficients
    glm_row <- data.frame(
      Feature   = feature,
      Term      = rownames(glm_summary),
      Estimate  = glm_summary[, "Estimate"],
      Std_Error = glm_summary[, "Std. Error"],
      df        = glm_summary[, "df"],
      t_value   = glm_summary[, "t value"],
      p_value   = glm_summary[, "Pr(>|t|)"]
    )
    glm_coefficients <- rbind(glm_coefficients, glm_row)
    
    # ---- 5.4: EFFECT SIZE CALCULATION ----
    eta_sq <- eta_squared(mixed_model, partial = TRUE)
    effect_row <- data.frame(
      Feature        = feature,
      Term           = eta_sq$Parameter,
      Eta2_partial   = eta_sq$Eta2_partial,
      CI_low         = eta_sq$CI_low,
      CI_high        = eta_sq$CI_high
    )
    effect_sizes <- rbind(effect_sizes, effect_row)
    
    # ---- 5.5: POST-HOC TUKEY TESTS ----
    # If the cluster main effect is significant, perform post-hoc for cluster
    if (anova_table["cluster_membership", "Pr(>F)"] < 0.05) {
      tukey_cluster <- glht(mixed_model, linfct = mcp(cluster_membership = "Tukey"))
      tukey_summary <- summary(tukey_cluster)
      tukey_row <- data.frame(
        Feature    = feature,
        Test_Type  = "Cluster_Comparison",
        Comparison = names(tukey_summary$test$coefficients),
        Estimate   = tukey_summary$test$coefficients,
        Std_Error  = tukey_summary$test$sigma,
        z_value    = tukey_summary$test$tstat,
        p_value    = tukey_summary$test$pvalues,
        p_adjusted = tukey_summary$test$pvalues   # Already Tukey-corrected
      )
      tukey_results <- rbind(tukey_results, tukey_row)
    }
    # If there are >2 groups, you may also wish to compare 'condition'
    if (nlevels(combined_membership$condition) > 2 && 
        anova_table["condition", "Pr(>F)"] < 0.05) {
      tukey_condition <- glht(mixed_model, linfct = mcp(condition = "Tukey"))
      tukey_summary_cond <- summary(tukey_condition)
      tukey_row_cond <- data.frame(
        Feature    = feature,
        Test_Type  = "Condition_Comparison", 
        Comparison = names(tukey_summary_cond$test$coefficients),
        Estimate   = tukey_summary_cond$test$coefficients,
        Std_Error  = tukey_summary_cond$test$sigma,
        z_value    = tukey_summary_cond$test$tstat,
        p_value    = tukey_summary_cond$test$pvalues,
        p_adjusted = tukey_summary_cond$test$pvalues
      )
      tukey_results <- rbind(tukey_results, tukey_row_cond)
    }
    
    # ---- 5.6: SUMMARY FOR CURRENT FEATURE ----
    summary_row <- data.frame(
      Feature                  = feature,
      Condition_F              = anova_table["condition", "F value"],
      Condition_p              = anova_table["condition", "Pr(>F)"],
      Condition_Significant    = anova_table["condition", "Pr(>F)"] < 0.05,
      Cluster_F                = anova_table["cluster_membership", "F value"],
      Cluster_p                = anova_table["cluster_membership", "Pr(>F)"],
      Cluster_Significant      = anova_table["cluster_membership", "Pr(>F)"] < 0.05,
      Interaction_F            = anova_table["condition:cluster_membership", "F value"],
      Interaction_p            = anova_table["condition:cluster_membership", "Pr(>F)"],
      Interaction_Significant  = anova_table["condition:cluster_membership", "Pr(>F)"] < 0.05,
      Condition_Effect_Size    = eta_sq[eta_sq$Parameter == "condition", "Eta2_partial"],
      Cluster_Effect_Size      = eta_sq[eta_sq$Parameter == "cluster_membership", "Eta2_partial"],
      Interaction_Effect_Size  = eta_sq[eta_sq$Parameter == "condition:cluster_membership", "Eta2_partial"]
    )
    results_summary <- rbind(results_summary, summary_row)
    
  }, error = function(e) {
    cat("ERROR in analysis for", feature, ":", e$message, "\n")
  })
}

# STEP 6: MULTIPLE TESTING CORRECTION ==========================================

cat("\n=== STEP 6: Multiple testing correction ===\n")

# Apply FDR correction to main and interaction p-values
results_summary$Condition_p_FDR   <- p.adjust(results_summary$Condition_p, method = "fdr")
results_summary$Cluster_p_FDR     <- p.adjust(results_summary$Cluster_p, method = "fdr")
results_summary$Interaction_p_FDR <- p.adjust(results_summary$Interaction_p, method = "fdr")

# Update significance flags after FDR correction
results_summary$Condition_Significant_FDR   <- results_summary$Condition_p_FDR < 0.05
results_summary$Cluster_Significant_FDR     <- results_summary$Cluster_p_FDR < 0.05
results_summary$Interaction_Significant_FDR <- results_summary$Interaction_p_FDR < 0.05

# STEP 7: SAVE ALL RESULTS ======================================================

cat("\n=== STEP 7: Saving results ===\n")

# Write results to CSV for reproducibility and reporting
write.csv(results_summary,  file.path(output_folder, "01_Summary_Results.csv"),     row.names = FALSE)
write.csv(anova_results,    file.path(output_folder, "02_ANOVA_Results.csv"),      row.names = FALSE)
write.csv(glm_coefficients, file.path(output_folder, "03_GLM_Coefficients.csv"),   row.names = FALSE)
write.csv(effect_sizes,     file.path(output_folder, "04_Effect_Sizes.csv"),       row.names = FALSE)
write.csv(tukey_results,    file.path(output_folder, "05_Tukey_PostHoc.csv"),      row.names = FALSE)

# Export key tables as LaTeX for publication-ready reporting
print(xtable(results_summary),  include.rownames = FALSE, 
      file = file.path(output_folder, "01_Summary_Results.tex"))
print(xtable(anova_results),    include.rownames = FALSE, 
      file = file.path(output_folder, "02_ANOVA_Results.tex"))
print(xtable(glm_coefficients), include.rownames = FALSE, 
      file = file.path(output_folder, "03_GLM_Coefficients.tex"))

# STEP 8: FINAL SUMMARY REPORT ==================================================

cat("\n=== STEP 8: Final report ===\n")

# Count and print the number of significant findings before and after correction
n_condition_sig      <- sum(results_summary$Condition_Significant,      na.rm = TRUE)
n_cluster_sig        <- sum(results_summary$Cluster_Significant,        na.rm = TRUE)
n_interaction_sig    <- sum(results_summary$Interaction_Significant,    na.rm = TRUE)

n_condition_sig_fdr  <- sum(results_summary$Condition_Significant_FDR,  na.rm = TRUE)
n_cluster_sig_fdr    <- sum(results_summary$Cluster_Significant_FDR,    na.rm = TRUE)
n_interaction_sig_fdr<- sum(results_summary$Interaction_Significant_FDR,na.rm = TRUE)

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

# List features with significant main or cluster effects after correction
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

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("All results saved in: ", output_folder, "\n")

# Show first five rows of summary for a quick check
cat("\nFirst 5 rows of the summary table:\n")
print(head(results_summary[, c("Feature", "Condition_p", "Cluster_p", "Interaction_p", 
                               "Condition_Significant_FDR", "Cluster_Significant_FDR")], 5))
