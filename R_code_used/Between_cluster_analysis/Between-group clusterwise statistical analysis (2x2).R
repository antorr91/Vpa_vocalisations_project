# ---------------------------------------------------------------
#   Statistical Comparison Between CTRL and VPA Groups by Cluster
#               (Two-cluster solution)
# ---------------------------------------------------------------

# ------------- Set Working Directory and Clear Workspace -------------
setwd("C:/Users/anton/VPA_vocalisations_project/Results_Analysis/Statistical_plots_between")
rm(list = ls())  # Remove all objects for a clean environment

# ------------- Load Required Libraries -------------
library(readr)    # For fast CSV import
library(plyr)     # For ddply and data grouping
library(dplyr)    # For modern tidy data manipulation
library(ggplot2)  # (Optional, useful for plotting)

# ------------- Import Data -------------
# Load cluster assignments for CTRL and VPA groups
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_2/hierarchical_clustering_2_distance_membership.csv")
ctrl_membership$condition <- "CTRL"  # Add group label

vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_2/hierarchical_clustering_2_distance_membership.csv")
vpa_membership$condition <- "VPA"    # Add group label

# Merge datasets for joint analysis
combined_membership <- bind_rows(ctrl_membership, vpa_membership)
t <- combined_membership
t$id <- t$recording  # Unique identifier for each chick/recording

# ------------- Significance Star Helper Function -------------
# Returns British significance stars according to p-value thresholds
get_stars <- function(p_value) {
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("")
}

# ------------- Output Directories for Results -------------
output_plot_folder  <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_plots_bt_condition"
output_stats_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_plots_bt_condition"

# Create output directories if they do not already exist
if (!dir.exists(output_plot_folder))  dir.create(output_plot_folder, recursive = TRUE)
if (!dir.exists(output_stats_folder)) dir.create(output_stats_folder, recursive = TRUE)

# ------------- List of Features for Analysis -------------
features <- c(
  "Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis", 
  "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean", 
  "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio", 
  "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
  "Slope", "Attack_magnitude", "Attack_time"
)

# ------------- Master Dataframe to Collect All Results -------------
all_statistical_results <- data.frame()

# =========== MAIN ANALYSIS LOOP: PER-FEATURE ===========
for (feature in features) {
  print(paste("Processing:", feature))
  
  # Check that the feature exists in the dataset
  if (!(feature %in% colnames(t))) {
    print(paste("Feature", feature, "not found in dataset. Skipping..."))
    next
  }
  
  # ----- Aggregate Data Per Animal, Condition, Cluster -----
  # Compute mean and SEM for each feature per individual (to account for individual variability)
  summary_stats <- ddply(
    t, .(id, condition, cluster_membership), summarise,
    mean_value = mean(get(feature), na.rm = TRUE),
    sem_value  = sd(get(feature), na.rm = TRUE) / sqrt(length(get(feature))),
    n          = length(get(feature))
  )
  
  # ----- Prepare an Empty Dataframe to Store Results -----
  statistical_results_between <- data.frame(
    Comparison           = character(),
    T_test_statistic     = numeric(),
    T_test_df            = numeric(),
    T_test_p_value       = numeric(),
    T_test_significance  = character(),
    Wilcox_statistic     = numeric(),
    Wilcox_p_value       = numeric(),
    Wilcox_significance  = character(),
    stringsAsFactors     = FALSE
  )
  
  # ----- Extract Data for Each Group/Cluster Combination -----
  ctrl_cluster1 <- summary_stats[summary_stats$condition == "CTRL" & summary_stats$cluster_membership == 1, ]
  ctrl_cluster0 <- summary_stats[summary_stats$condition == "CTRL" & summary_stats$cluster_membership == 0, ]
  vpa_cluster1  <- summary_stats[summary_stats$condition == "VPA"  & summary_stats$cluster_membership == 1, ]
  vpa_cluster0  <- summary_stats[summary_stats$condition == "VPA"  & summary_stats$cluster_membership == 0, ]
  
  # ----- Helper Function: Add Test Results to Dataframe -----
  add_test_results <- function(df, comp_name, t_test, wilcox_test) {
    new_row <- data.frame(
      Comparison           = comp_name,
      T_test_statistic     = t_test$statistic,
      T_test_df            = t_test$parameter,
      T_test_p_value       = t_test$p.value,
      T_test_significance  = get_stars(t_test$p.value),
      Wilcox_statistic     = wilcox_test$statistic,
      Wilcox_p_value       = wilcox_test$p.value,
      Wilcox_significance  = get_stars(wilcox_test$p.value)
    )
    return(rbind(df, new_row))
  }
  
  # ----- Define the Specific Comparisons to Perform -----
  comparisons <- list(
    "CTRL Cluster 1 vs VPA Cluster 1" = list(ctrl_cluster1, vpa_cluster1),
    "CTRL Cluster 1 vs VPA Cluster 0" = list(ctrl_cluster1, vpa_cluster0),
    "CTRL Cluster 0 vs VPA Cluster 1" = list(ctrl_cluster0, vpa_cluster1),
    "CTRL Cluster 0 vs VPA Cluster 0" = list(ctrl_cluster0, vpa_cluster0)
  )
  
  # ========== PERFORM STATISTICAL TESTS FOR EACH COMPARISON ==========
  for (comp_name in names(comparisons)) {
    data1 <- comparisons[[comp_name]][[1]]
    data2 <- comparisons[[comp_name]][[2]]
    
    # Must have at least two animals in each group to perform statistical tests
    if (nrow(data1) < 2 | nrow(data2) < 2) {
      print(paste("Skipping comparison", comp_name, "due to insufficient data"))
      next
    }
    
    # --- Student's t-test (Welch's, unequal variance, robust to non-homogeneity)
    t_test <- t.test(data1$mean_value, data2$mean_value)
    # --- Wilcoxon rank sum test (non-parametric, for robustness)
    wilcox_test <- wilcox.test(data1$mean_value, data2$mean_value)
    
    # Add results to results table
    statistical_results_between <- add_test_results(
      statistical_results_between, 
      comp_name, 
      t_test, 
      wilcox_test
    )
  }
  
  # ----- Tidy and Add Results to Master Table -----
  statistical_results_between$Feature <- feature
  statistical_results_between <- statistical_results_between %>% 
    select(Feature, everything())  # Move Feature to front
  
  # Bind to master dataframe
  all_statistical_results <- rbind(all_statistical_results, statistical_results_between)
  
  # Print results for current feature
  print(statistical_results_between)
  
  # ----- Write Results for This Feature to CSV -----
  write.csv(
    statistical_results_between, 
    file.path(output_stats_folder, paste0("statistical_results_between_", feature, ".csv")), 
    row.names = FALSE
  )
  
  # ----- Optionally: Save Results as Readable TXT -----
  sink(file.path(output_stats_folder, paste0("statistical_results_between_", feature, ".txt")))
  for (comp in unique(statistical_results_between$Comparison)) {
    cat("\nResults for", comp, "(", feature, "):\n")
    results <- statistical_results_between[statistical_results_between$Comparison == comp,]
    cat("T-test p-value:", results$T_test_p_value, 
        "Significance:", results$T_test_significance, "\n")
    cat("Wilcoxon test p-value:", results$Wilcox_p_value, 
        "Significance:", results$Wilcox_significance, "\n")
    cat("\n-----------------------------------\n")
  }
  sink()
}

# ========== FINAL AGGREGATED OUTPUT ==========
# Save all results together for downstream analyses or tables
write.csv(
  all_statistical_results,
  file.path(output_stats_folder, "all_statistical_results_between.csv"), 
  row.names = FALSE
)

