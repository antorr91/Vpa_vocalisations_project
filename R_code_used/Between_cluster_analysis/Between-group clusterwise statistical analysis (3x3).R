# -----------------------------------------------------------
#        Between-group Cluster Comparison (CTRL vs VPA)
#      Statistical Tests for Acoustic Call Features
#                (Three-cluster solution)
# -----------------------------------------------------------

# --------- Set Working Directory and Clean Environment ---------
setwd("C:/Users/anton/VPA_vocalisations_project/Results_Analysis/Statistical_plots_between")
rm(list = ls())  # Remove all objects to start fresh

# --------- Load Required Libraries ---------
library(readr)    # For importing .csv files
library(plyr)     # For ddply and data grouping
library(dplyr)    # For tidy data manipulation
library(ggplot2)  # For plotting (if needed, not used here)

# --------- Import and Prepare Data ---------
# Read cluster membership for CTRL group and label
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv")
ctrl_membership$condition <- "CTRL"

# Read cluster membership for VPA group and label
vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_3/hierarchical_clustering_3_distance_membership.csv")
vpa_membership$condition <- "VPA"

# Combine both datasets for joint processing
combined_membership <- bind_rows(ctrl_membership, vpa_membership)
t <- combined_membership
t$id <- t$recording  # Unique animal identifier

# --------- Helper Function for British Significance Stars ---------
get_stars <- function(p_value) {
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("")
}

# --------- Output Folders for Results ---------
output_plot_folder  <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_bt3_condition"
output_stats_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_bt3_condition"

# Create output folders if they do not already exist
if (!dir.exists(output_plot_folder))  dir.create(output_plot_folder, recursive = TRUE)
if (!dir.exists(output_stats_folder)) dir.create(output_stats_folder, recursive = TRUE)

# --------- List of Features to be Analysed ---------
features <- c(
  "Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis", 
  "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean", 
  "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio", 
  "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
  "Slope", "Attack_magnitude", "Attack_time"
)

# --------- Main Analysis: Loop Over All Features ---------
all_statistical_results <- data.frame()  # Master dataframe for all results

for (feature in features) {
  cat("\nProcessing:", feature, "\n")
  
  # Skip if the feature does not exist in the dataset
  if (!(feature %in% colnames(t))) {
    cat("Feature", feature, "not found in dataset. Skipping...\n")
    next
  }
  
  # --------- Aggregate Per-Recording (Individual), Condition and Cluster ---------
  summary_stats <- ddply(
    t, .(id, condition, cluster_membership), summarise,
    mean_value = mean(get(feature), na.rm = TRUE),
    sem_value  = sd(get(feature), na.rm = TRUE) / sqrt(length(get(feature))),
    n          = length(get(feature))
  )
  
  # --------- Dataframe for Pairwise Comparison Results (for This Feature) ---------
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
  
  # --------- Extract Data for Each Cluster in Both Groups ---------
  # This explicit approach helps prevent errors if any cluster is missing
  ctrl_cluster0 <- summary_stats[summary_stats$condition == "CTRL" & summary_stats$cluster_membership == 0, ]
  ctrl_cluster1 <- summary_stats[summary_stats$condition == "CTRL" & summary_stats$cluster_membership == 1, ]
  ctrl_cluster2 <- summary_stats[summary_stats$condition == "CTRL" & summary_stats$cluster_membership == 2, ]
  vpa_cluster0  <- summary_stats[summary_stats$condition == "VPA"  & summary_stats$cluster_membership == 0, ]
  vpa_cluster1  <- summary_stats[summary_stats$condition == "VPA"  & summary_stats$cluster_membership == 1, ]
  vpa_cluster2  <- summary_stats[summary_stats$condition == "VPA"  & summary_stats$cluster_membership == 2, ]
  
  # --------- Define Comparison Pairs: All Possible CTRL Cluster vs VPA Cluster ---------
  comparisons <- list(
    "CTRL Cluster 0 vs VPA Cluster 0" = list(ctrl_cluster0, vpa_cluster0),
    "CTRL Cluster 0 vs VPA Cluster 1" = list(ctrl_cluster0, vpa_cluster1),
    "CTRL Cluster 0 vs VPA Cluster 2" = list(ctrl_cluster0, vpa_cluster2),
    "CTRL Cluster 1 vs VPA Cluster 0" = list(ctrl_cluster1, vpa_cluster0),
    "CTRL Cluster 1 vs VPA Cluster 1" = list(ctrl_cluster1, vpa_cluster1),
    "CTRL Cluster 1 vs VPA Cluster 2" = list(ctrl_cluster1, vpa_cluster2),
    "CTRL Cluster 2 vs VPA Cluster 0" = list(ctrl_cluster2, vpa_cluster0),
    "CTRL Cluster 2 vs VPA Cluster 1" = list(ctrl_cluster2, vpa_cluster1),
    "CTRL Cluster 2 vs VPA Cluster 2" = list(ctrl_cluster2, vpa_cluster2)
  )
  
  # --------- Run Tests for Each Comparison ---------
  for (comp_name in names(comparisons)) {
    data1 <- comparisons[[comp_name]][[1]]
    data2 <- comparisons[[comp_name]][[2]]
    
    # Require at least two animals per group to run a valid statistical test
    if (nrow(data1) < 2 | nrow(data2) < 2) {
      cat("Skipping comparison", comp_name, "due to insufficient data\n")
      next
    }
    
    # Run Student's t-test (Welch's by default)
    t_test     <- t.test(data1$mean_value, data2$mean_value)
    # Run Wilcoxon rank sum test (non-parametric)
    wilcox_test <- wilcox.test(data1$mean_value, data2$mean_value)
    
    # Add results to the dataframe
    statistical_results_between <- rbind(
      statistical_results_between,
      data.frame(
        Comparison           = comp_name,
        T_test_statistic     = t_test$statistic,
        T_test_df            = t_test$parameter,
        T_test_p_value       = t_test$p.value,
        T_test_significance  = get_stars(t_test$p.value),
        Wilcox_statistic     = wilcox_test$statistic,
        Wilcox_p_value       = wilcox_test$p.value,
        Wilcox_significance  = get_stars(wilcox_test$p.value)
      )
    )
  }
  
  # --------- Add Feature Column and Bind to Master Results ---------
  statistical_results_between$Feature <- feature
  statistical_results_between <- statistical_results_between %>% 
    select(Feature, everything())
  all_statistical_results <- rbind(all_statistical_results, statistical_results_between)
  
  # --------- Output Results for This Feature ---------
  print(statistical_results_between)
  
  # Save as CSV
  write.csv(
    statistical_results_between, 
    file.path(output_stats_folder, paste0("statistical_results_between_", feature, ".csv")), 
    row.names = FALSE
  )
  
  # Optionally: Save a human-readable text file
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

# --------- Save All Results in a Single File for Further Analysis ---------
write.csv(
  all_statistical_results,
  file.path(output_stats_folder, "all_statistical_results_between.csv"), 
  row.names = FALSE
)

# ------------- End of Script -------------
