# Clear the R environment to avoid any leftover variables from previous analyses
rm(list = ls()) 

# Load all necessary libraries for data handling, stats, and plotting
library(readr)      # For reading CSV files
library(plyr)       # For data wrangling
library(dplyr)      # For tidy data manipulation
library(ggplot2)    # For plotting
library(ggsignif)   # For annotating significance on plots
library(stats)      # For statistical tests (e.g., t-test, Wilcoxon)

# ----------------- DATA IMPORT -------------------
# Load the control group data and assign a condition label
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_2/hierarchical_clustering_2_distance_membership.csv")
ctrl_membership$condition <- "CTRL"

# Load the VPA group data and assign a condition label
vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_2/hierarchical_clustering_2_distance_membership.csv")
vpa_membership$condition <- "VPA"

# Merge the two datasets into one for joint analysis
combined_membership <- bind_rows(ctrl_membership, vpa_membership)
t <- combined_membership
t$id <- t$recording    # Assign recording ID for individual-level aggregation

# -------------- OUTPUT FOLDERS -------------------
# Define output directories for plots and statistical results
output_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Plots_between_2_conditions_clusters_comparison_2cl"
plot_folder   <- file.path(output_folder, "Plots")
stats_folder  <- file.path(output_folder, "Statistical_Results")

# Create output directories if they do not already exist
if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)
if (!dir.exists(plot_folder))   dir.create(plot_folder)
if (!dir.exists(stats_folder))  dir.create(stats_folder)

# -------------- SIGNIFICANCE FUNCTION ------------
# Function to convert p-values to British-style significance asterisks
get_stars <- function(p_value) {
  if (is.na(p_value)) return("NS") # Not significant if p is missing
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("NS") # Not significant if above threshold
}

# ------------- FEATURES TO ANALYSE ---------------
# List of all features to be analysed between groups and clusters
features <- c("Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis", 
              "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean", 
              "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio", 
              "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
              "Slope", "Attack_magnitude", "Attack_time")

# Prepare an empty dataframe to save all statistical results
statistical_results_between <- data.frame()

# ----------- MAIN ANALYSIS LOOP ------------------
for (feature in features) {
  cat("\nProcessing feature:", feature, "\n")
  
  # If the feature is not present, skip to the next
  if (!(feature %in% colnames(t))) {
    cat("Feature", feature, "not found in dataset. Skipping.\n")
    next
  }
  
  # ------- AGGREGATE PER INDIVIDUAL+CLUSTER+GROUP -------
  # For each individual (id), group (condition), and cluster, compute mean, SEM, n
  summary_stats <- tryCatch({
    ddply(t, .(id, condition, cluster_membership), summarise,
          mean_value = mean(get(feature), na.rm = TRUE),
          sem_value  = sd(get(feature), na.rm = TRUE) / sqrt(length(get(feature))),
          n         = length(get(feature)))
  }, error = function(e) {
    cat("Error during aggregation for", feature, ":", e$message, "\n")
    return(NULL)
  })
  
  # If the aggregation fails, skip this feature
  if (is.null(summary_stats)) next
  
  # ---- CLUSTER-BY-CLUSTER COMPARISONS BETWEEN GROUPS ----
  # For each CTRL cluster (0/1), compare to each VPA cluster (0/1)
  cluster_comparisons <- data.frame()
  
  for (ctrl_cluster in 0:1) {
    for (vpa_cluster in 0:1) {
      # Extract data for the current clusters
      ctrl_data <- summary_stats %>%
        filter(condition == "CTRL", cluster_membership == ctrl_cluster)
      vpa_data  <- summary_stats %>%
        filter(condition == "VPA",  cluster_membership == vpa_cluster)
      
      # If either group has less than 3 samples, skip (not enough for stats)
      if (nrow(ctrl_data) < 3 || nrow(vpa_data) < 3) {
        cat("Insufficient data for comparison: CTRL", ctrl_cluster, "vs VPA", vpa_cluster, "\n")
        next
      }
      
      # Conduct t-test and Wilcoxon test between the groups
      t_test_result <- tryCatch(
        t.test(ctrl_data$mean_value, vpa_data$mean_value),
        error = function(e) list(p.value = NA)
      )
      wilcox_result <- tryCatch(
        wilcox.test(ctrl_data$mean_value, vpa_data$mean_value),
        error = function(e) list(p.value = NA)
      )
      
      # Store results in a temporary dataframe
      comparison_result <- data.frame(
        Feature        = feature,
        CTRL_Cluster   = ctrl_cluster,
        VPA_Cluster    = vpa_cluster,
        T_test_p_value = t_test_result$p.value,
        Wilcox_p_value = wilcox_result$p.value,
        CTRL_n         = nrow(ctrl_data),
        VPA_n          = nrow(vpa_data)
      )
      # Append to master results
      cluster_comparisons <- rbind(cluster_comparisons, comparison_result)
    }
  }
  
  # ---- COMPUTE GRAND MEANS (for plotting bars) ----
  # Average mean_value per condition and cluster for plotting
  grand_means <- summary_stats %>%
    group_by(condition, cluster_membership) %>%
    summarise(
      mean = mean(mean_value, na.rm = TRUE),
      sem  = sd(mean_value, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  # ---- PLOTTING FOR EACH CTRL CLUSTER VS VPA CLUSTERS ----
  for (ctrl_cluster in unique(grand_means$cluster_membership[grand_means$condition == "CTRL"])) {
    # Prepare data for plotting: always CTRL_X plus both VPA clusters
    plot_data <- grand_means %>%
      filter((condition == "CTRL" & cluster_membership == ctrl_cluster) | 
               condition == "VPA") %>%
      mutate(group = paste(condition, cluster_membership, sep = " - "))
    
    # Prepare the comparison results for this set
    comparisons <- cluster_comparisons %>%
      filter(Feature == feature, CTRL_Cluster == ctrl_cluster) %>%
      mutate(
        group1 = paste("CTRL -", CTRL_Cluster),
        group2 = paste("VPA -", VPA_Cluster),
        y_pos  = max(plot_data$mean, na.rm = TRUE) + seq(0.1, by = 0.1, length.out = n())
      )
    
    # If there are no valid comparisons, skip the plot
    if (nrow(comparisons) == 0) next
    
    # Create the bar plot with error bars and custom colours
    p <- ggplot(plot_data, aes(x = group, y = mean, fill = condition)) +
      geom_col(position = position_dodge(0.8), width = 0.7) +
      geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem),
                    width = 0.2, position = position_dodge(0.8)) +
      scale_fill_manual(values = c("CTRL" = "grey60", "VPA" = "cornflowerblue")) + 
      labs(
        title = paste(feature, "- Comparison: CTRL Cluster", ctrl_cluster, "vs VPA Clusters"),
        x     = "Group and Cluster",
        y     = "Mean ± SEM"
      ) +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    # Annotate significance with British-style asterisks for each VPA cluster
    if (nrow(comparisons) > 0) {
      comparisons_list <- lapply(1:nrow(comparisons), function(i) {
        c(comparisons$group1[i], comparisons$group2[i])
      })
      
      # Add dynamic annotation for statistical significance
      p <- p + geom_signif(
        comparisons = comparisons_list,
        annotations = sapply(comparisons$T_test_p_value, get_stars),
        y_position  = comparisons$y_pos,
        tip_length  = 0.01,
        vjust       = -0.2
      )
    }
    
    # Save the plot to the designated folder
    ggsave(
      file.path(plot_folder, paste0(feature, "_CTRL", ctrl_cluster, "_vs_VPA.png")),
      plot = p, width = 6, height = 5
    )
  }
  
  # Save the statistical results as CSV for this feature
  write_csv(cluster_comparisons, file.path(stats_folder, paste0(feature, "_stats.csv")))
}


