rm(list=ls()) 

# Load required libraries
library(readr)
library(plyr)
library(dplyr)
library(ggplot2)

# ----------- DATA IMPORT -----------
# Import cluster membership for the control and VPA groups
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_2/hierarchical_clustering_2_distance_membership.csv")
ctrl_membership$condition <- "CTRL"

vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_2/hierarchical_clustering_2_distance_membership.csv")
vpa_membership$condition <- "VPA"

# For the VPA group, invert the cluster membership labels (optional: based on your needs)
vpa_membership <- vpa_membership %>%
  mutate(cluster_membership = ifelse(cluster_membership == 0, 1, 0))

# Combine the two groups into a single dataframe
combined_membership <- bind_rows(ctrl_membership, vpa_membership)
t <- combined_membership
t$id <- t$recording  # Set the unique animal/recording ID

# ----------- SIGNIFICANCE STARS FUNCTION -----------
get_stars <- function(p_value) {
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("")
}

# ----------- OUTPUT FOLDERS -----------
output_plot_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_plots_within_condition"
output_stats_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_plots_within_condition"

if (!dir.exists(output_plot_folder)) dir.create(output_plot_folder, recursive = TRUE)
if (!dir.exists(output_stats_folder)) dir.create(output_stats_folder, recursive = TRUE)

# ----------- RESULTS DATAFRAME -----------
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

# ----------- FEATURE LIST -----------
features <- c("Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis", 
              "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean", 
              "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio", 
              "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
              "Slope", "Attack_magnitude", "Attack_time")

# ----------- MAIN ANALYSIS LOOP -----------
for (feature in features) {
  print(paste("Processing:", feature))
  
  # Check the feature exists in the dataset
  if (!(feature %in% colnames(t))) {
    print(paste("Feature", feature, "not found in dataset. Skipping..."))
    next
  }
  
  # Calculate per-animal (ID) average within each group and cluster
  summary_data_id <- ddply(t, .(condition, cluster_membership, id), summarise,
                           average_value = mean(get(feature), na.rm = TRUE),
                           sem_value = sd(get(feature), na.rm = TRUE) / sqrt(length(get(feature))),
                           N_value = length(get(feature)))
  
  # Separate for each condition
  summary_ctrl <- summary_data_id[summary_data_id$condition == "CTRL",]
  summary_vpa  <- summary_data_id[summary_data_id$condition == "VPA",]
  
  # Perform t-test and Wilcoxon test within each group (cluster 0 vs 1)
  t_test_ctrl    <- t.test(average_value ~ cluster_membership, data = summary_ctrl)
  wilcox_test_ctrl <- wilcox.test(average_value ~ cluster_membership, data = summary_ctrl)
  t_test_vpa     <- t.test(average_value ~ cluster_membership, data = summary_vpa)
  wilcox_test_vpa  <- wilcox.test(average_value ~ cluster_membership, data = summary_vpa)
  
  # Store test results
  statistical_results <- rbind(statistical_results, data.frame(
    Feature = feature,
    Condition = "CTRL",
    T_test_statistic = t_test_ctrl$statistic,
    T_test_df = t_test_ctrl$parameter,
    T_test_p_value = t_test_ctrl$p.value,
    T_test_significance = get_stars(t_test_ctrl$p.value),
    Wilcox_statistic = wilcox_test_ctrl$statistic,
    Wilcox_p_value = wilcox_test_ctrl$p.value,
    Wilcox_significance = get_stars(wilcox_test_ctrl$p.value)
  ))
  
  statistical_results <- rbind(statistical_results, data.frame(
    Feature = feature,
    Condition = "VPA",
    T_test_statistic = t_test_vpa$statistic,
    T_test_df = t_test_vpa$parameter,
    T_test_p_value = t_test_vpa$p.value,
    T_test_significance = get_stars(t_test_vpa$p.value),
    Wilcox_statistic = wilcox_test_vpa$statistic,
    Wilcox_p_value = wilcox_test_vpa$p.value,
    Wilcox_significance = get_stars(wilcox_test_vpa$p.value)
  ))
  
  # Calculate group/cluster summary statistics for plotting (mean, SEM, n)
  summary_stats <- ddply(summary_data_id, .(condition, cluster_membership), summarise,
                         mean_value = mean(average_value, na.rm = TRUE),
                         sem_value = sd(average_value, na.rm = TRUE) / sqrt(length(average_value)),
                         n = length(average_value))
  
  # For CTRL plot
  summary_ctrl_stats <- summary_stats[summary_stats$condition == "CTRL",]
  star_ctrl <- get_stars(t_test_ctrl$p.value)
  
  plot_ctrl <- ggplot(summary_ctrl_stats, aes(x = as.factor(cluster_membership), 
                                              y = mean_value, 
                                              fill = as.factor(cluster_membership))) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value),
                  width = 0.2, position = position_dodge(width = 0.7)) +
    geom_point(data = summary_ctrl, aes(y = average_value),
               position = position_jitter(width = 0.2),
               color = "#4D4D4D", size = 1, alpha = 0.5) +
    labs(title = paste(feature, "by Cluster Membership - CTRL"),
         subtitle = paste("n =", paste(summary_ctrl_stats$n, collapse = ", ")),
         x = "Cluster Membership", 
         y = paste("Average", feature)) +
    scale_fill_manual(values = c("0" = "#6E6E6E", "1" = "#B0B0B0")) +
    theme_minimal() +
    theme(legend.position = "none") +
    annotate("text", x = 1.5, 
             y = max(summary_ctrl_stats$mean_value + summary_ctrl_stats$sem_value) * 1.1, 
             label = star_ctrl, size = 6)
  
  # For VPA plot
  summary_vpa_stats <- summary_stats[summary_stats$condition == "VPA",]
  star_vpa <- get_stars(t_test_vpa$p.value)
  
  plot_vpa <- ggplot(summary_vpa_stats, aes(x = as.factor(cluster_membership), 
                                            y = mean_value, 
                                            fill = as.factor(cluster_membership))) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value),
                  width = 0.2, position = position_dodge(width = 0.7)) +
    geom_point(data = summary_vpa, aes(y = average_value),
               position = position_jitter(width = 0.2),
               color = "#004C99", size = 1, alpha = 0.5) +
    labs(title = paste(feature, "by Cluster Membership - VPA"),
         subtitle = paste("n =", paste(summary_vpa_stats$n, collapse = ", ")),
         x = "Cluster Membership", 
         y = paste("Average", feature)) +
    scale_fill_manual(values = c("0" = "#89CFF0", "1" = "#1F77B4")) +
    theme_minimal() +
    theme(legend.position = "none") +
    annotate("text", x = 1.5, 
             y = max(summary_vpa_stats$mean_value + summary_vpa_stats$sem_value) * 1.1, 
             label = star_vpa, size = 6)
  
  # Save the plots for each feature and condition
  output_plot_path_ctrl <- file.path(output_plot_folder, 
                                     paste0(gsub(" ", "_", feature), "_CTRL_plot.png"))
  output_plot_path_vpa <- file.path(output_plot_folder, 
                                    paste0(gsub(" ", "_", feature), "_VPA_plot.png"))
  
  ggsave(output_plot_path_ctrl, plot = plot_ctrl, width = 8, height = 6, dpi = 300)
  ggsave(output_plot_path_vpa, plot = plot_vpa, width = 8, height = 6, dpi = 300)
}

# ----------- SAVE STATISTICAL RESULTS -----------
write.csv(statistical_results, 
          file.path(output_stats_folder, "statistical_results.csv"), 
          row.names = FALSE)

# Optionally, also save a more readable text summary for quick inspection
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
