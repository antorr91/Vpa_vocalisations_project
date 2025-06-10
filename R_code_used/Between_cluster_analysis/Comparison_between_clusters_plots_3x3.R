# ===========================
# Analysis of Vocal Features: 
# Comparing Three-Cluster Memberships Between CTRL and VPA Groups
# ===========================

# --------- Housekeeping: Clear the Workspace and Load Libraries ----------
rm(list = ls())  # Remove all existing objects from the workspace for a clean session

# Import essential libraries for data wrangling, plotting, and statistical testing
library(readr)      # For reading .csv data files
library(plyr)       # For group-based data manipulation
library(dplyr)      # For tidy data pipelines and summary statistics
library(ggplot2)    # For advanced plotting capabilities
library(ggsignif)   # For adding statistical significance bars to ggplot graphs
library(stats)      # For standard statistical tests

# --------- Data Import: Read in Clustering Results for Both Groups ----------
# Load cluster membership assignments for the control group
ctrl_membership <- read_csv(
  "C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv"
)
ctrl_membership$condition <- "CTRL"  # Add a column specifying the experimental group

# Load cluster membership assignments for the VPA group
vpa_membership <- read_csv(
  "C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_3/hierarchical_clustering_3_distance_membership.csv"
)
vpa_membership$condition <- "VPA"    # Add a column specifying the experimental group

# Combine both groups into a single data frame for joint analysis
combined_membership <- bind_rows(ctrl_membership, vpa_membership)
t <- combined_membership
t$id <- t$recording  # Use the recording identifier as 'id' for animal-level aggregation

# --------- Prepare Output Directories for Results and Plots ----------
output_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Plots_between_3_conditions_clusters_comparison"
plot_folder   <- file.path(output_folder, "Plots")
stats_folder  <- file.path(output_folder, "Statistical_Results")

# Create output directories if they do not already exist
if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)
if (!dir.exists(plot_folder))   dir.create(plot_folder)
if (!dir.exists(stats_folder))  dir.create(stats_folder)

# --------- Define a Helper Function for Significance Stars ----------
# Returns asterisks according to British statistical reporting conventions
get_stars <- function(p_value) {
  if (is.na(p_value)) return("NS")  # NS = Not Significant
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("NS")
}

# --------- List of Acoustic Features to Analyse ----------
features <- c(
  "Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis", 
  "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean", 
  "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio", 
  "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
  "Slope", "Attack_magnitude", "Attack_time"
)

# --------- MAIN ANALYSIS LOOP OVER FEATURES ----------
for (feature in features) {
  cat("\nProcessing feature:", feature, "\n")
  
  # Skip to next feature if not present in the dataset
  if (!(feature %in% colnames(t))) {
    cat("Feature", feature, "not found. Skipping...\n")
    next
  }
  
  # ------- Aggregate Per Animal, Condition and Cluster -------
  summary_stats <- tryCatch({
    ddply(t, .(id, condition, cluster_membership), summarise,
          mean_value = mean(get(feature), na.rm = TRUE),
          sem_value  = sd(get(feature), na.rm = TRUE) / sqrt(length(get(feature))),
          n          = length(get(feature))
    )
  }, error = function(e) {
    cat("Error in data aggregation for", feature, ":", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(summary_stats)) next  # If error occurred, skip to next feature
  
  # ------- Prepare All Pairwise Comparisons Between CTRL and VPA Clusters -------
  cluster_comparisons <- data.frame()  # Collects all pairwise stats for this feature
  
  for (ctrl_cluster in 0:2) {
    for (vpa_cluster in 0:2) {
      # Extract per-animal averages for each group/cluster combination
      ctrl_data <- summary_stats %>%
        filter(condition == "CTRL", cluster_membership == ctrl_cluster)
      vpa_data  <- summary_stats %>%
        filter(condition == "VPA", cluster_membership == vpa_cluster)
      
      # Ensure both groups have enough data for meaningful comparison
      if (nrow(ctrl_data) < 3 || nrow(vpa_data) < 3) {
        cat("Insufficient data for comparison: CTRL Cluster", ctrl_cluster, 
            "vs VPA Cluster", vpa_cluster, "\n")
        next
      }
      
      # Conduct parametric and non-parametric statistical tests
      t_test_result <- tryCatch(
        t.test(ctrl_data$mean_value, vpa_data$mean_value),
        error = function(e) list(statistic = NA, parameter = NA, p.value = NA)
      )
      wilcox_result <- tryCatch(
        wilcox.test(ctrl_data$mean_value, vpa_data$mean_value),
        error = function(e) list(statistic = NA, p.value = NA)
      )
      
      # Record the results for this comparison
      comparison_result <- data.frame(
        Feature            = feature,
        CTRL_Cluster       = ctrl_cluster,
        VPA_Cluster        = vpa_cluster,
        T_test_statistic   = t_test_result$statistic,
        T_test_df          = t_test_result$parameter,
        T_test_p_value     = t_test_result$p.value,
        T_test_significance = get_stars(t_test_result$p.value),
        Wilcox_statistic   = wilcox_result$statistic,
        Wilcox_p_value     = wilcox_result$p.value,
        Wilcox_significance = get_stars(wilcox_result$p.value),
        CTRL_n             = nrow(ctrl_data),
        VPA_n              = nrow(vpa_data)
      )
      cluster_comparisons <- rbind(cluster_comparisons, comparison_result)
    }
  }
  
  # If no valid comparisons were found, skip to the next feature
  if (nrow(cluster_comparisons) == 0) {
    cat("No valid comparisons for", feature, "\n")
    next
  }
  
  # ------- Calculate Mean and SEM per Condition/Cluster for Plotting -------
  grand_means <- summary_stats %>%
    group_by(condition, cluster_membership) %>%
    summarise(
      mean = mean(mean_value, na.rm = TRUE),
      sem  = sd(mean_value, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  # ------- Plotting: For Each CTRL Cluster, Show All Three VPA Clusters -------
  for (ctrl_cluster in unique(grand_means$cluster_membership[grand_means$condition == "CTRL"])) {
    # Assemble the data to plot: CTRL cluster plus all VPA clusters
    ctrl_mean <- grand_means %>%
      filter(condition == "CTRL", cluster_membership == ctrl_cluster)
    vpa_means <- grand_means %>%
      filter(condition == "VPA")
    plot_data <- bind_rows(ctrl_mean, vpa_means) %>%
      mutate(
        group = factor(paste(condition, cluster_membership, sep = " - "))
      )
    
    # Prepare the comparison annotations for ggplot2
    comparisons <- cluster_comparisons %>%
      filter(CTRL_Cluster == ctrl_cluster) %>%
      mutate(
        group1 = paste("CTRL -", CTRL_Cluster),
        group2 = paste("VPA -", VPA_Cluster),
        y_pos  = max(plot_data$mean, na.rm = TRUE) + seq(0.1, 0.3, length.out = n())
      )
    
    # Build the bar plot with error bars
    p <- ggplot(plot_data, aes(x = group, y = mean, fill = condition)) +
      geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
      geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem),
                    width = 0.2, position = position_dodge(0.7)) +
      scale_fill_manual(values = c("CTRL" = "grey60", "VPA" = "cornflowerblue")) +
      labs(
        title = paste(feature, "- CTRL Cluster", ctrl_cluster, "vs VPA Clusters"),
        x     = "Cluster Membership",
        y     = "Mean ± SEM",
        fill  = "Condition"
      ) +
      theme_minimal(base_size = 12) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top"
      )
    
    # Add significance bars for each VPA comparison (if available)
    if (nrow(comparisons) > 0) {
      comparisons_list <- lapply(1:nrow(comparisons), function(i) {
        c(comparisons$group1[i], comparisons$group2[i])
      })
      p <- p + geom_signif(
        comparisons = comparisons_list,
        annotations = comparisons$T_test_significance,
        y_position  = comparisons$y_pos,
        tip_length  = 0.01,
        vjust       = -0.2
      )
    }
    
    # Save the plot as a .png file, using feature and cluster in the filename
    ggsave(
      filename = file.path(plot_folder, paste0(feature, "_CTRL", ctrl_cluster, "_vs_VPA.png")),
      plot     = p,
      width    = 8,
      height   = 6
    )
  }
  
  # Save the statistical results for this feature in the stats folder
  write.csv(cluster_comparisons, 
            file = file.path(stats_folder, paste0(feature, "_stats.csv")), 
            row.names = FALSE)
}

# Optionally, after the main loop, aggregate and save all comparisons
# write.csv(all_comparisons, file = file.path(stats_folder, "all_features_clusterwise_stats.csv"), row.names = FALSE)
