# Plot acoustic features by condition and cluster (2 and 3 cluster solutions)
# Antonella Torrisi

rm(list = ls())

library(plyr)
library(dplyr)
library(ggplot2)
library(readr)
library(gridExtra)
library(grid)

# load data
setwd("/path/to/project")
#setwd("C:\Users\anton\VPA_vocalisations_project\Results_Clustering")
data_path <- file.path("data", "full_dataset_vpa_ctrl.csv")
t <- read_csv(data_path)

t <- t[complete.cases(t), ]

t <- t %>%
  mutate(
    id                   = as.factor(id),
    condition            = as.factor(condition),
    cluster_membership_2 = as.factor(cluster_membership_2),
    cluster_membership_3 = as.factor(cluster_membership_3),
    bin                  = as.factor(bin)
  )

# features
features <- c(
  "Duration_call", "Attack_time", "Slope",
  "RMS_Mean", "RMS_Std",
  "F0_Mean", "F0_Std", "F0_Skewness", "F0_Kurtosis", "F0_Bandwidth",
  "F0_1st_Order_Diff", "F0_Slope", "F0_Mag_Mean",
  "F1_Mag_Mean", "F2_Mag_Mean", "F1_F0_Ratio", "F2_F0_Ratio",
  "Spectral_Centroid_Mean", "Spectral_Centroid_Std", "Attack_magnitude"
)

feature_labels <- c(
  "Duration_call"         = "Call Duration",
  "Attack_time"           = "Attack Time",
  "Slope"                 = "Amplitude Slope",
  "RMS_Mean"              = "RMS Mean",
  "RMS_Std"               = "RMS Std",
  "F0_Mean"               = "F0 Mean",
  "F0_Std"                = "F0 Std",
  "F0_Skewness"           = "F0 Skewness",
  "F0_Kurtosis"           = "F0 Kurtosis",
  "F0_Bandwidth"          = "F0 Bandwidth",
  "F0_1st_Order_Diff"     = "F0 1st Order Diff",
  "F0_Slope"              = "F0 Slope",
  "F0_Mag_Mean"           = "F0 Magnitude Mean",
  "F1_Mag_Mean"           = "F1 Magnitude Mean",
  "F2_Mag_Mean"           = "F2 Magnitude Mean",
  "F1_F0_Ratio"           = "F1/F0 Ratio",
  "F2_F0_Ratio"           = "F2/F0 Ratio",
  "Spectral_Centroid_Mean" = "Spectral Centroid Mean",
  "Spectral_Centroid_Std" = "Spectral Centroid Std",
  "Attack_magnitude"      = "Attack Magnitude"
)

# plotting function: subject-level means -> group-level means +/- sem
plot_feature_enhanced <- function(data, feature_name, save_plot = TRUE) {
  
  if (!feature_name %in% names(data)) {
    cat("feature not found:", feature_name, "\n")
    return(NULL)
  }
  
  vars_needed  <- c(feature_name, "bin", "condition", "cluster_membership_2", "cluster_membership_3", "id")
  data_complete <- data[complete.cases(data[, vars_needed]), ]
  
  if (nrow(data_complete) == 0) {
    cat("no complete cases:", feature_name, "\n")
    return(NULL)
  }
  
  y_label <- ifelse(feature_name %in% names(feature_labels), feature_labels[feature_name], feature_name)
  
  output_dir_2 <- file.path(getwd(), "stat_plots", "2_Clusters")
  output_dir_3 <- file.path(getwd(), "stat_plots", "3_Clusters")
  if (!dir.exists(output_dir_2)) dir.create(output_dir_2, recursive = TRUE)
  if (!dir.exists(output_dir_3)) dir.create(output_dir_3, recursive = TRUE)
  
  # 2-cluster plot
  subject_summary_2 <- ddply(
    data_complete,
    .(id, bin, condition, cluster_membership_2),
    function(df) {
      vals <- df[[feature_name]]
      n    <- sum(!is.na(vals))
      data.frame(average_value = mean(vals, na.rm = TRUE), n = n,
                 se_value = ifelse(n > 1, sd(vals, na.rm = TRUE) / sqrt(n), NA))
    }
  )
  
  group_summary_2 <- ddply(
    subject_summary_2,
    .(bin, condition, cluster_membership_2),
    function(df) {
      vals       <- df$average_value
      n_subjects <- sum(!is.na(vals))
      data.frame(mean_value = mean(vals, na.rm = TRUE), n_subjects = n_subjects,
                 se_value = ifelse(n_subjects > 1, sd(vals, na.rm = TRUE) / sqrt(n_subjects), NA))
    }
  )
  group_summary_2 <- na.omit(group_summary_2)
  
  if (nrow(group_summary_2) > 0) {
    p2 <- ggplot(group_summary_2, aes(x = bin, y = mean_value, color = condition, group = condition)) +
      geom_line(linewidth = 1.2) +
      geom_point(size = 4) +
      geom_errorbar(aes(ymin = mean_value - se_value, ymax = mean_value + se_value),
                    width = 0.2, linewidth = 0.9, na.rm = TRUE) +
      scale_color_manual(values = c("CTRL" = "#F8766D", "VPA" = "#00BFC4"),
                         labels = c("CTRL" = "CTRL", "VPA" = "VPA")) +
      labs(x = "Time window", y = paste("Mean", y_label), color = "Condition") +
      theme_minimal(base_size = 16, base_family = "sans") +
      theme(
        plot.title       = element_blank(),
        axis.title.x     = element_text(size = 16, face = "italic", margin = margin(t = 15)),
        axis.title.y     = element_text(size = 16, face = "italic", margin = margin(r = 15)),
        axis.text.x      = element_text(size = 14, face = "bold"),
        axis.text.y      = element_text(size = 14, face = "bold"),
        legend.title     = element_text(size = 15, face = "bold"),
        legend.text      = element_text(size = 14),
        legend.position  = "bottom",
        strip.text       = element_text(size = 18, face = "bold"),
        panel.grid.minor = element_blank(),
        panel.spacing    = unit(2, "lines")
      ) +
      facet_wrap(~ cluster_membership_2,
                 labeller = labeller(cluster_membership_2 = c("0" = "Cluster 0", "1" = "Cluster 1")))
    
    if (save_plot) {
      ggsave(file.path(output_dir_2, paste0(feature_name, "_2clusters.png")),
             plot = p2, width = 14, height = 7, dpi = 300, bg = "white")
      ggsave(file.path(output_dir_2, paste0(feature_name, "_2clusters.pdf")),
             plot = p2, width = 14, height = 7, device = "pdf")
    }
  } else {
    p2 <- NULL
  }
  
  # 3-cluster plot
  subject_summary_3 <- ddply(
    data_complete,
    .(id, bin, condition, cluster_membership_3),
    function(df) {
      vals <- df[[feature_name]]
      n    <- sum(!is.na(vals))
      data.frame(average_value = mean(vals, na.rm = TRUE), n = n,
                 se_value = ifelse(n > 1, sd(vals, na.rm = TRUE) / sqrt(n), NA))
    }
  )
  
  group_summary_3 <- ddply(
    subject_summary_3,
    .(bin, condition, cluster_membership_3),
    function(df) {
      vals       <- df$average_value
      n_subjects <- sum(!is.na(vals))
      data.frame(mean_value = mean(vals, na.rm = TRUE), n_subjects = n_subjects,
                 se_value = ifelse(n_subjects > 1, sd(vals, na.rm = TRUE) / sqrt(n_subjects), NA))
    }
  )
  group_summary_3 <- na.omit(group_summary_3)
  
  if (nrow(group_summary_3) > 0) {
    p3 <- ggplot(group_summary_3, aes(x = bin, y = mean_value, color = condition, group = condition)) +
      geom_line(linewidth = 1.2) +
      geom_point(size = 4) +
      geom_errorbar(aes(ymin = mean_value - se_value, ymax = mean_value + se_value),
                    width = 0.2, linewidth = 0.9, na.rm = TRUE) +
      scale_color_manual(values = c("CTRL" = "#F8766D", "VPA" = "#00BFC4"),
                         labels = c("CTRL" = "CTRL", "VPA" = "VPA")) +
      labs(x = "Time window", y = paste("Mean", y_label), color = "Condition") +
      theme_minimal(base_size = 16, base_family = "sans") +
      theme(
        plot.title       = element_blank(),
        axis.title.x     = element_text(size = 16, face = "italic", margin = margin(t = 15)),
        axis.title.y     = element_text(size = 16, face = "italic", margin = margin(r = 15)),
        axis.text.x      = element_text(size = 14, face = "bold"),
        axis.text.y      = element_text(size = 14, face = "bold"),
        legend.title     = element_text(size = 15, face = "bold"),
        legend.text      = element_text(size = 14),
        legend.position  = "bottom",
        strip.text       = element_text(size = 18, face = "bold"),
        panel.grid.minor = element_blank(),
        panel.spacing    = unit(1.5, "lines")
      ) +
      facet_wrap(~ cluster_membership_3, nrow = 1,
                 labeller = labeller(cluster_membership_3 = c("0" = "Cluster 0", "1" = "Cluster 1", "2" = "Cluster 2")))
    
    if (save_plot) {
      ggsave(file.path(output_dir_3, paste0(feature_name, "_3clusters.png")),
             plot = p3, width = 18, height = 7, dpi = 300, bg = "white")
      ggsave(file.path(output_dir_3, paste0(feature_name, "_3clusters.pdf")),
             plot = p3, width = 18, height = 7, device = "pdf")
    }
  } else {
    p3 <- NULL
  }
  
  return(list(plot_2clusters = p2, plot_3clusters = p3,
              data_2clusters = group_summary_2, data_3clusters = group_summary_3))
}

# run for all features
all_results     <- list()
successful_plots <- 0
failed_plots     <- 0

for (i in seq_along(features)) {
  feature <- features[i]
  cat("[", i, "/", length(features), "]", feature, "\n")
  
  result <- tryCatch({
    plot_feature_enhanced(t, feature, save_plot = TRUE)
  }, error = function(e) {
    cat("error:", e$message, "\n")
    NULL
  })
  
  if (!is.null(result)) {
    all_results[[feature]] <- result
    successful_plots <- successful_plots + 1
  } else {
    failed_plots <- failed_plots + 1
  }
}

# combined multi-panel figures for key features
key_features       <- c("Duration_call", "F0_Mean", "RMS_Mean",
                        "Spectral_Centroid_Mean", "Attack_magnitude", "F0_Std")
plot_list_2clusters <- list()
plot_list_3clusters <- list()

for (feature in key_features) {
  if (feature %in% names(all_results) && !is.null(all_results[[feature]]$plot_2clusters)) {
    plot_list_2clusters[[feature]] <- all_results[[feature]]$plot_2clusters +
      ggtitle(feature_labels[feature]) +
      theme(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
            axis.title = element_text(size = 10), axis.text = element_text(size = 9),
            strip.text = element_text(size = 11), legend.position = "none")
    
    if (!is.null(all_results[[feature]]$plot_3clusters)) {
      plot_list_3clusters[[feature]] <- all_results[[feature]]$plot_3clusters +
        ggtitle(feature_labels[feature]) +
        theme(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
              axis.title = element_text(size = 10), axis.text = element_text(size = 9),
              strip.text = element_text(size = 11), legend.position = "none")
    }
  }
}

output_dir_combined <- file.path(getwd(), "stat_plots", "Combined_Panels")
if (!dir.exists(output_dir_combined)) dir.create(output_dir_combined, recursive = TRUE)

if (length(plot_list_2clusters) > 0) {
  combined_2 <- grid.arrange(grobs = plot_list_2clusters, ncol = 2,
                             top = textGrob("Key Acoustic Features - 2 Cluster Solution",
                                            gp = gpar(fontsize = 18, fontface = "bold")))
  ggsave(file.path(output_dir_combined, "Combined_Key_Features_2clusters.png"),
         plot = combined_2, width = 16, height = 12, dpi = 300, bg = "white")
  ggsave(file.path(output_dir_combined, "Combined_Key_Features_2clusters.pdf"),
         plot = combined_2, width = 16, height = 12, device = "pdf")
}

if (length(plot_list_3clusters) > 0) {
  combined_3 <- grid.arrange(grobs = plot_list_3clusters, ncol = 2,
                             top = textGrob("Key Acoustic Features - 3 Cluster Solution",
                                            gp = gpar(fontsize = 18, fontface = "bold")))
  ggsave(file.path(output_dir_combined, "Combined_Key_Features_3clusters.png"),
         plot = combined_3, width = 16, height = 12, dpi = 300, bg = "white")
  ggsave(file.path(output_dir_combined, "Combined_Key_Features_3clusters.pdf"),
         plot = combined_3, width = 16, height = 12, device = "pdf")
}

cat("done:", successful_plots, "plots saved,", failed_plots, "failed\n")

# save workspace
save(all_results, feature_labels,
     file = file.path(getwd(), "stat_plots", "Feature_Plots_Workspace.RData"))