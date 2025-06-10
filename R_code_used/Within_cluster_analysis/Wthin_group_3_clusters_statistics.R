# Set working directory
setwd("C:/Users/anton/VPA_vocalisations_project/Results_Analysis/")
rm(list = ls())

# Load required libraries
library(readr)
library(plyr)
library(dplyr)
library(ggplot2)

# Import datasets for control and VPA groups
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv")
ctrl_membership$condition <- "CTRL"

vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_46_3_clusters/hierarchical_clustering_3_distance_membership.csv")
vpa_membership$condition <- "VPA"

# Combine data for both groups
combined_membership <- bind_rows(ctrl_membership, vpa_membership)
t <- combined_membership
t$id <- t$recording

# Function to assign significance stars
get_stars <- function(p_value) {
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("")
}

# Create output folder if it does not exist
output_stats_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_results_within_3_clusters"
if (!dir.exists(output_stats_folder)) {
  dir.create(output_stats_folder, recursive = TRUE)
}

# Check the number of clusters present
num_clusters <- length(unique(vpa_membership$cluster_membership))
print(paste("Number of unique clusters:", num_clusters))

# Initialise results dataframe
statistical_results <- data.frame(
  Feature = character(),
  Condition = character(),
  Cluster1 = numeric(),
  Cluster2 = numeric(),
  T_test_statistic = numeric(),
  T_test_df = numeric(),
  T_test_p_value = numeric(),
  T_test_significance = character(),
  Wilcox_statistic = numeric(),
  Wilcox_p_value = numeric(),
  Wilcox_significance = character(),
  stringsAsFactors = FALSE
)

# List of features to analyse
features <- c(
  "Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis",
  "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean",
  "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio",
  "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
  "Slope", "Attack_magnitude", "Attack_time"
)

# Loop over each feature
for (feature in features) {
  print(paste("Processing:", feature))
  
  # Check if feature exists in the dataset
  if (!(feature %in% colnames(t))) {
    print(paste("Feature", feature, "not found in dataset. Skipping..."))
    next
  }
  
  # Compute summary statistics for each id, within condition and cluster
  summary_data_id <- ddply(
    t, .(condition, cluster_membership, id), summarise,
    average_value = mean(get(feature), na.rm = TRUE),
    sem_value = sd(get(feature), na.rm = TRUE) / sqrt(length(na.omit(get(feature)))),
    N_value = length(na.omit(get(feature)))
  )
  
  # Separate data by condition
  summary_vpa <- subset(summary_data_id, condition == "VPA")
  summary_ctrl <- subset(summary_data_id, condition == "CTRL")
  
  # Split data by cluster (assuming clusters are labelled 0, 1, 2)
  vpa_0 <- subset(summary_vpa, cluster_membership == 0)
  vpa_1 <- subset(summary_vpa, cluster_membership == 1)
  vpa_2 <- subset(summary_vpa, cluster_membership == 2)
  
  ctrl_0 <- subset(summary_ctrl, cluster_membership == 0)
  ctrl_1 <- subset(summary_ctrl, cluster_membership == 1)
  ctrl_2 <- subset(summary_ctrl, cluster_membership == 2)
  
  # Define pairs of clusters to compare within each group
  cluster_pairs <- list(
    list("CTRL", ctrl_0, ctrl_1, 0, 1),
    list("CTRL", ctrl_0, ctrl_2, 0, 2),
    list("CTRL", ctrl_1, ctrl_2, 1, 2),
    list("VPA", vpa_0, vpa_1, 0, 1),
    list("VPA", vpa_0, vpa_2, 0, 2),
    list("VPA", vpa_1, vpa_2, 1, 2)
  )
  
  # Perform tests for each pair of clusters
  for (pair in cluster_pairs) {
    condition <- pair[[1]]
    data1 <- pair[[2]]
    data2 <- pair[[3]]
    cluster1 <- pair[[4]]
    cluster2 <- pair[[5]]
    
    # Only test if both clusters have more than one data point
    if (nrow(data1) > 1 & nrow(data2) > 1) {
      data_subset <- rbind(data1, data2)
      
      # T-test
      t_test_result <- t.test(average_value ~ cluster_membership, data = data_subset)
      
      # Wilcoxon test
      wilcox_test_result <- wilcox.test(average_value ~ cluster_membership, data = data_subset)
      
      # Append results to dataframe
      statistical_results <- rbind(statistical_results, data.frame(
        Feature = feature,
        Condition = condition,
        Cluster1 = cluster1,
        Cluster2 = cluster2,
        T_test_statistic = t_test_result$statistic,
        T_test_df = t_test_result$parameter,
        T_test_p_value = t_test_result$p.value,
        T_test_significance = get_stars(t_test_result$p.value),
        Wilcox_statistic = wilcox_test_result$statistic,
        Wilcox_p_value = wilcox_test_result$p.value,
        Wilcox_significance = get_stars(wilcox_test_result$p.value)
      ))
    } else {
      print(paste("Not enough data to compare cluster", cluster1, "vs", cluster2, "in", condition, "group."))
    }
  }
}

# Save the statistical results to a CSV file
write.csv(statistical_results,
          file.path(output_stats_folder, "statistical_results_46_files_3clusters.csv"),
          row.names = FALSE)

# Optionally, save a more readable text summary
sink(file.path(output_stats_folder, "statistical_results_46_files_3clusters.txt"))
for (feature in unique(statistical_results$Feature)) {
  cat("\nResults for", feature, ":\n")
  for (condition in c("CTRL", "VPA")) {
    results_condition <- statistical_results[statistical_results$Feature == feature & 
                                               statistical_results$Condition == condition, ]
    cat("\n", condition, "group:\n")
    for (i in 1:nrow(results_condition)) {
      cat("Cluster", results_condition$Cluster1[i], "vs", results_condition$Cluster2[i], "\n")
      cat("T-test p-value:", results_condition$T_test_p_value[i], 
          "Significance:", results_condition$T_test_significance[i], "\n")
      cat("Wilcoxon test p-value:", results_condition$Wilcox_p_value[i], 
          "Significance:", results_condition$Wilcox_significance[i], "\n\n")
    }
  }
  cat("\n-----------------------------------\n")
}
sink()
