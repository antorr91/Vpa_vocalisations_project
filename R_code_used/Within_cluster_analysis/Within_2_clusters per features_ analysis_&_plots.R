# Set the working directory
setwd("C:/Users/anton/VPA_vocalisations_project/Results_Analysis/")
rm(list = ls())

# Load required libraries
library(readr)
library(plyr)
library(dplyr)
library(ggplot2)

# Import data for both groups
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_2/hierarchical_clustering_2_distance_membership.csv")
ctrl_membership$condition <- "CTRL"

vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_2/hierarchical_clustering_2_distance_membership.csv")
vpa_membership$condition <- "VPA"

# Optionally invert the cluster labels for the VPA group, to align clusters between groups
vpa_membership <- vpa_membership %>%
  mutate(cluster_membership = ifelse(cluster_membership == 0, 1, 0))

# Combine both groups into a single dataframe
combined_membership <- bind_rows(ctrl_membership, vpa_membership)
t <- combined_membership
t$id <- t$recording

# Function to add significance stars according to p-value thresholds
get_stars <- function(p_value) {
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("")
}

# Output folders for plots and statistical results
output_plot_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_plots_within_condition"
output_stats_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_plots_within_condition"

if (!dir.exists(output_plot_folder)) dir.create(output_plot_folder, recursive = TRUE)
if (!dir.exists(output_stats_folder)) dir.create(output_stats_folder, recursive = TRUE)

# Create a dataframe to collect the results
statistical_results <- data.frame(
  Feature = character(),
  Condition = character(),
  T_test_statistic = numeric(),
  T_test_df = numeric(),
  T_test_p_value = numeric(),
  T_test_significance = character(),
  Wilcox_statistic = numeric(),
  Wilcox_p_value = numeric(),
  Wilcox_significance = character(),
  stringsAsFactors = FALSE
)

# List of features to analyse (column names must match the dataframe)
features <- c(
  "Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis", 
  "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean", 
  "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio", 
  "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
  "Slope", "Attack_magnitude", "Attack_time"
)

for (feature in features) {
  print(paste("Processing:", feature))
  
  # Check if the feature exists in the dataset
  if (!(feature %in% colnames(t))) {
    print(paste("Feature", feature, "not found in dataset. Skipping..."))
    next
  }
  
  # Compute individual means (per chick/subject), grouped by condition and cluster
  summary_data_id <- ddply(
    t, .(condition, cluster_membership, id), summarise,
    average_value = mean(get(feature), na.rm = TRUE),
    sem_value = sd(get(feature), na.rm = TRUE) / sqrt(length(get(feature))),
    N_value = length(get(feature))
  )
  
  # Split by group for separate statistics
  summary_vpa <- summary_data_id[summary_data_id$condition == "VPA", ]
  summary_ctrl <- summary_data_id[summary_data_id$condition == "CTRL", ]
  
  # Run t-test and Wilcoxon test for cluster membership within each group
  t_test_vpa <- t.test(average_value ~ cluster_membership, data = summary_vpa)
  wilcox_test_vpa <- wilcox.test(average_value ~ cluster_membership, data = summary_vpa)
  t_test_ctrl <- t.test(average_value ~ cluster_membership, data = summary_ctrl)
  wilcox_test_ctrl <- wilcox.test(average_value ~ cluster_membership, data = summary_ctrl)
  
  # Append the results for CTRL
  statistical_results <- rbind(statistical_results, data.frame(
    Feature = feature,
    Condition = "CTRL",
    T_test_statistic = as.numeric(t_test_ctrl$statistic),
    T_test_df = as.numeric(t_test_ctrl$parameter),
    T_test_p_value = t_test_ctrl$p.value,
    T_test_significance = get_stars(t_test_ctrl$p.value),
    Wilcox_statistic = as.numeric(wilcox_test_ctrl$statistic),
    Wilcox_p_value = wilcox_test_ctrl$p.value,
    Wilcox_significance = get_stars(wilcox_test_ctrl$p.value)
  ))
  
  # Append the results for VPA
  statistical_results <- rbind(statistical_results, data.frame(
    Feature = feature,
    Condition = "VPA",
    T_test_statistic = as.numeric(t_test_vpa$statistic),
    T_test_df = as.numeric(t_test_vpa$parameter),
    T_test_p_value = t_test_vpa$p.value,
    T_test_significance = get_stars(t_test_vpa$p.value),
    Wilcox_statistic = as.numeric(wilcox_test_vpa$statistic),
    Wilcox_p_value = wilcox_test_vpa$p.value,
    Wilcox_significance = get_stars(wilcox_test_vpa$p.value)
  ))
  
  # Prepare means and SEM for plotting
  summary_stats <- ddply(summary_data_id, .(condition, cluster_membership), summarise,
                         mean_value = mean(average_value, na.rm = TRUE),
                         sem_value = sd(average_value, na.rm = TRUE) / sqrt(length(average_value)),
                         n = length(average_value)
  )
  
  # Separate for plotting
  summary_vpa <- summary_stats[summary_stats$condition == "VPA", ]
  summary_ctrl <- summary_stats[summary_stats$condition == "CTRL", ]
  
  # Plot for CTRL group
  plot_ctrl <- ggplot(summary_ctrl, aes(x = as.factor(cluster_membership), 
                                        y = mean_value, 
                                        fill = as.factor(cluster_membership))) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_errorbar(aes(ymin = mean_value - sem_value, 
                      ymax = mean_value + sem_value),
                  width = 0.2, position = position_dodge(width = 0.7)) +
    geom_point(
      data = summary_data_id[summary_data_id$condition == "CTRL", ],
      aes(y = average_value),
      position = position_jitter(width = 0.2),
      colour = "#4D4D4D", size = 1, alpha = 0.5
    ) +
    labs(
      title = paste(feature, "by Cluster Membership - CTRL"),
      subtitle = paste("n =", paste(summary_ctrl$n, collapse = ", ")),
      x = "Cluster Membership",
      y = paste("Average", feature)
    ) +
    scale_fill_manual(values = c("0" = "#6E6E6E", "1" = "#B0B0B0")) +
    theme_minimal() +
    theme(legend.position = "none") +
    annotate(
      "text", x = 1.5, 
      y = max(summary_ctrl$mean_value + summary_ctrl$sem_value) * 1.1, 
      label = get_stars(t_test_ctrl$p.value), size = 6
    )
  
  # Plot for VPA group
  plot_vpa <- ggplot(summary_vpa, aes(x = as.factor(cluster_membership), 
                                      y = mean_value, 
                                      fill = as.factor(cluster_membership))) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_errorbar(aes(ymin = mean_value - sem_value, 
                      ymax = mean_value + sem_value),
                  width = 0.2, position = position_dodge(width = 0.7)) +
    geom_point(
      data = summary_data_id[summary_data_id$condition == "VPA", ],
      aes(y = average_value),
      position = position_jitter(width = 0.2),
      colour = "#004C99", size = 1, alpha = 0.5
    ) +
    labs(
      title = paste(feature, "by Cluster Membership - VPA"),
      subtitle = paste("n =", paste(summary_vpa$n, collapse = ", ")),
      x = "Cluster Membership",
      y = paste("Average", feature)
    ) +
    scale_fill_manual(values = c("0" = "#89CFF0", "1" = "#1F77B4")) +
    theme_minimal() +
    theme(legend.position = "none") +
    annotate(
      "text", x = 1.5, 
      y = max(summary_vpa$mean_value + summary_vpa$sem_value) * 1.1, 
      label = get_stars(t_test_vpa$p.value), size = 6
    )
  
  # Save the plots
  output_plot_path_ctrl <- file.path(output_plot_folder, 
                                     paste0(gsub(" ", "_", feature), "_CTRL_plot.png"))
  output_plot_path_vpa <- file.path(output_plot_folder, 
                                    paste0(gsub(" ", "_", feature), "_VPA_plot.png"))
  
  ggsave(output_plot_path_ctrl, plot = plot_ctrl, width = 8, height = 6, dpi = 300)
  ggsave(output_plot_path_vpa, plot = plot_vpa, width = 8, height = 6, dpi = 300)
}

# Save the statistical results as a CSV file
write.csv(statistical_results, 
          file.path(output_stats_folder, "statistical_results.csv"), 
          row.names = FALSE)

# Optionally, save a human-readable text summary
sink(file.path(output_stats_folder, "statistical_results.txt"))
for (feature in unique(statistical_results$Feature)) {
  cat("\nResults for", feature, ":\n")
  for (condition in c("CTRL", "VPA")) {
    results <- statistical_results[statistical_results$Feature == feature & 
                                     statistical_results$Condition == condition,]
    cat("\n", condition, "group:\n")
    cat("T-test p-value:", results$T_test_p_value, 
        "Significance:", results$T_test_significance, "\n")
    cat("Wilcoxon test p-value:", results$Wilcox_p_value, 
        "Significance:", results$Wilcox_significance, "\n")
  }
  cat("\n-----------------------------------\n")
}
sink()
