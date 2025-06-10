# Set working directory and clear workspace
setwd("C:/Users/anton/VPA_vocalisations_project/Results_Analysis/")
rm(list = ls())

# Required libraries
library(readr)
library(plyr)
library(dplyr)
library(ggplot2)
library(ggsignif)
library(stats)

# Import control and VPA group data
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv")
ctrl_membership$condition <- "CTRL"

vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_3/hierarchical_clustering_3_distance_membership.csv")
vpa_membership$condition <- "VPA"

# Merge both groups into one dataframe
combined_membership <- bind_rows(ctrl_membership, vpa_membership)
t <- combined_membership
t$id <- t$recording

# Ensure cluster_membership is treated as a factor for plotting and numeric for analysis
t$cluster_membership <- as.numeric(as.character(t$cluster_membership))

# Create output directories for plots and statistics
output_plot_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Plots_TTest_3clusters"
if (!dir.exists(output_plot_folder)) dir.create(output_plot_folder, recursive = TRUE)
output_stats_folder <- file.path(output_plot_folder, "Statistical_Results")
if (!dir.exists(output_stats_folder)) dir.create(output_stats_folder, recursive = TRUE)

# Function to assign significance stars
get_stars <- function(p_value) {
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("")
}

# Function to perform t-tests between all pairs of clusters
pairwise_t_test <- function(data) {
  results <- data.frame()
  cluster_pairs <- list(
    c(0, 1),
    c(0, 2),
    c(1, 2)
  )
  for (pair in cluster_pairs) {
    cluster1 <- pair[1]
    cluster2 <- pair[2]
    data_c1 <- data[data$cluster_membership == cluster1, ]
    data_c2 <- data[data$cluster_membership == cluster2, ]
    if (nrow(data_c1) > 1 && nrow(data_c2) > 1) { # At least 2 points per cluster
      t_test_result <- t.test(data_c1$average_value, data_c2$average_value)
      results <- rbind(results, data.frame(
        Cluster1 = as.character(cluster1),
        Cluster2 = as.character(cluster2),
        T_statistic = t_test_result$statistic,
        Df = t_test_result$parameter,
        P_value = t_test_result$p.value,
        Significance = get_stars(t_test_result$p.value)
      ))
    } else {
      results <- rbind(results, data.frame(
        Cluster1 = as.character(cluster1),
        Cluster2 = as.character(cluster2),
        T_statistic = NA, Df = NA, P_value = NA, Significance = "NA"
      ))
    }
  }
  return(results)
}

# List of features to analyse
features <- c("Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis", 
              "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean", 
              "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio", 
              "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
              "Slope", "Attack_magnitude", "Attack_time")

# Prepare results dataframe
statistical_results <- data.frame()

# Loop for feature analysis and plotting
for (feature in features) {
  print(paste("Processing:", feature))
  
  # Check if the feature exists in the data
  if (!(feature %in% colnames(t))) {
    print(paste("Feature", feature, "not found in dataset. Skipping..."))
    next
  }
  
  # Calculate average per animal, per group and cluster
  summary_data_id <- ddply(t, .(condition, cluster_membership, id), summarise,
                           average_value = mean(get(feature), na.rm = TRUE),
                           sem_value = sd(get(feature), na.rm = TRUE) / sqrt(length(na.omit(get(feature)))),
                           N_value = length(na.omit(get(feature))))
  
  # Separate by condition
  summary_vpa <- summary_data_id[summary_data_id$condition == "VPA", ]
  summary_ctrl <- summary_data_id[summary_data_id$condition == "CTRL", ]
  
  # Pairwise t-tests for clusters within each group
  vpa_pairwise_results <- pairwise_t_test(summary_vpa)
  ctrl_pairwise_results <- pairwise_t_test(summary_ctrl)
  
  # Collect results
  vpa_results <- cbind(Feature = feature, Condition = "VPA", vpa_pairwise_results)
  ctrl_results <- cbind(Feature = feature, Condition = "CTRL", ctrl_pairwise_results)
  statistical_results <- rbind(statistical_results, vpa_results, ctrl_results)
  
  # Calculate mean and SEM for plotting
  summary_stats <- ddply(summary_data_id, .(condition, cluster_membership), summarise,
                         mean_value = mean(average_value, na.rm = TRUE),
                         sem_value = sd(average_value, na.rm = TRUE) / sqrt(length(na.omit(average_value))),
                         n = length(na.omit(average_value)))
  summary_stats$cluster_membership <- as.factor(summary_stats$cluster_membership)
  
  # Prepare plotting data
  summary_vpa <- summary_stats[summary_stats$condition == "VPA", ]
  summary_ctrl <- summary_stats[summary_stats$condition == "CTRL", ]
  
  # Bar plot for CTRL group
  plot_ctrl <- ggplot(summary_ctrl, aes(x = cluster_membership, y = mean_value, fill = cluster_membership)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value),
                  width = 0.2, position = position_dodge(width = 0.7)) +
    geom_point(data = summary_data_id[summary_data_id$condition == "CTRL", ],
               aes(x = as.factor(cluster_membership), y = average_value, group = id),
               position = position_jitter(width = 0.2), color = "#4D4D4D", size = 1, alpha = 0.5) +
    labs(title = paste(feature, "by Cluster Membership - CTRL"),
         subtitle = paste("n =", paste(summary_ctrl$n, collapse = ", ")),
         x = "Cluster Membership",
         y = paste("Average", feature)) +
    scale_fill_manual(values = c("0" = "#6E6E6E", "1" = "#B0B0B0", "2" = "#D3D3D3")) +
    theme_minimal() +
    theme(legend.position = "none")
  # Add significance annotations for CTRL
  plot_ctrl <- plot_ctrl +
    geom_signif(comparisons = list(c("0", "1"), c("0", "2"), c("1", "2")),
                annotations = ctrl_pairwise_results$Significance,
                tip_length = 0.01, textsize = 5, y_position = c(
                  max(summary_ctrl$mean_value + summary_ctrl$sem_value, na.rm = TRUE) * 1.08,
                  max(summary_ctrl$mean_value + summary_ctrl$sem_value, na.rm = TRUE) * 1.16,
                  max(summary_ctrl$mean_value + summary_ctrl$sem_value, na.rm = TRUE) * 1.24
                ))
  
  # Bar plot for VPA group
  plot_vpa <- ggplot(summary_vpa, aes(x = cluster_membership, y = mean_value, fill = cluster_membership)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value),
                  width = 0.2, position = position_dodge(width = 0.7)) +
    geom_point(data = summary_data_id[summary_data_id$condition == "VPA", ],
               aes(x = as.factor(cluster_membership), y = average_value, group = id),
               position = position_jitter(width = 0.2), color = "#004C99", size = 1, alpha = 0.5) +
    labs(title = paste(feature, "by Cluster Membership - VPA"),
         subtitle = paste("n =", paste(summary_vpa$n, collapse = ", ")),
         x = "Cluster Membership",
         y = paste("Average", feature)) +
    scale_fill_manual(values = c("0" = "#89CFF0", "1" = "#1F77B4", "2" = "#004C99")) +
    theme_minimal() +
    theme(legend.position = "none")
  # Add significance annotations for VPA
  plot_vpa <- plot_vpa +
    geom_signif(comparisons = list(c("0", "1"), c("0", "2"), c("1", "2")),
                annotations = vpa_pairwise_results$Significance,
                tip_length = 0.01, textsize = 5, y_position = c(
                  max(summary_vpa$mean_value + summary_vpa$sem_value, na.rm = TRUE) * 1.08,
                  max(summary_vpa$mean_value + summary_vpa$sem_value, na.rm = TRUE) * 1.16,
                  max(summary_vpa$mean_value + summary_vpa$sem_value, na.rm = TRUE) * 1.24
                ))
  
  # Save plots as PNG
  ggsave(filename = file.path(output_plot_folder, paste0(feature, "_CTRL.png")), 
         plot_ctrl, width = 8, height = 6)
  ggsave(filename = file.path(output_plot_folder, paste0(feature, "_VPA.png")), 
         plot_vpa, width = 8, height = 6)
}

# Save statistical results as CSV
write.csv(statistical_results, 
          file = file.path(output_stats_folder, "pairwise_cluster_statistical_results.csv"), 
          row.names = FALSE)