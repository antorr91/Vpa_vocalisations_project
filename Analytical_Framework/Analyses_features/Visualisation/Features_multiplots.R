# Combined multi-panel feature plots with ribbon and shared legend
# Antonella Torrisi

rm(list = ls())

library(plyr)
library(dplyr)
library(ggplot2)
library(readr)
library(gridExtra)
library(grid)
library(gtable)
library(patchwork)

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
t$bin <- relevel(t$bin, ref = "1")

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
  "Duration_call"          = "Call Duration",
  "Attack_time"            = "Attack Time",
  "Slope"                  = "Envelope Slope",
  "RMS_Mean"               = "RMS Mean",
  "RMS_Std"                = "RMS Std Dev",
  "F0_Mean"                = "F0 Mean",
  "F0_Std"                 = "F0 Std Dev",
  "F0_Skewness"            = "F0 Skewness",
  "F0_Kurtosis"            = "F0 Kurtosis",
  "F0_Bandwidth"           = "F0 Bandwidth",
  "F0_1st_Order_Diff"      = "F0 First Order Difference",
  "F0_Slope"               = "F0 Slope",
  "F0_Mag_Mean"            = "F0 Magnitude Mean",
  "F1_Mag_Mean"            = "F1 Magnitude Mean",
  "F2_Mag_Mean"            = "F2 Magnitude Mean",
  "F1_F0_Ratio"            = "F1/F0 Ratio",
  "F2_F0_Ratio"            = "F2/F0 Ratio",
  "Spectral_Centroid_Mean" = "Spectral Centroid Mean",
  "Spectral_Centroid_Std"  = "Spectral Centroid Std Dev",
  "Attack_magnitude"       = "Attack Magnitude"
)

feature_domains <- list(
  Temporal  = c("Duration_call", "Attack_time", "Slope"),
  Energy    = c("RMS_Mean", "RMS_Std"),
  Frequency = c("F0_Mean", "F0_Std", "F0_Skewness", "F0_Kurtosis",
                "F0_Bandwidth", "F0_1st_Order_Diff", "F0_Slope",
                "F0_Mag_Mean", "F1_Mag_Mean", "F2_Mag_Mean",
                "F1_F0_Ratio", "F2_F0_Ratio",
                "Spectral_Centroid_Mean", "Spectral_Centroid_Std", "Attack_magnitude")
)

output_dir <- file.path(getwd(), "Feature_Plots_Ribbons")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# plotting function
plot_feature_with_ribbons <- function(data, feature_name, save_plot = TRUE) {
  
  p2 <- NULL; p3 <- NULL
  group_summary_2 <- NULL; group_summary_3 <- NULL
  
  if (!feature_name %in% names(data)) {
    cat("feature not found:", feature_name, "\n")
    return(NULL)
  }
  
  vars_needed   <- c(feature_name, "bin", "condition", "cluster_membership_2", "cluster_membership_3", "id")
  data_complete <- data[complete.cases(data[, vars_needed]), ]
  
  if (nrow(data_complete) == 0) {
    cat("no complete cases:", feature_name, "\n")
    return(NULL)
  }
  
  y_label <- ifelse(feature_name %in% names(feature_labels), feature_labels[feature_name], feature_name)
  
  # 2-cluster ribbon plot
  subject_summary_2 <- ddply(
    data_complete,
    .(id, bin, condition, cluster_membership_2),
    function(df) {
      x <- df[[feature_name]]; n <- sum(!is.na(x))
      data.frame(average_value = mean(x, na.rm = TRUE), n = n)
    }
  )
  
  group_summary_2 <- ddply(
    subject_summary_2,
    .(bin, condition, cluster_membership_2),
    function(df) {
      n_subjects <- length(df$average_value)
      mean_val   <- mean(df$average_value, na.rm = TRUE)
      se_val     <- ifelse(n_subjects > 1, sd(df$average_value, na.rm = TRUE) / sqrt(n_subjects), 0)
      data.frame(mean_value = mean_val, n_subjects = n_subjects, se_value = se_val,
                 lower = mean_val - se_val, upper = mean_val + se_val)
    }
  )
  group_summary_2 <- na.omit(group_summary_2)
  group_summary_2$bin_numeric <- as.numeric(as.character(group_summary_2$bin))
  
  if (nrow(group_summary_2) > 0) {
    p2 <- ggplot(group_summary_2, aes(x = bin_numeric, y = mean_value,
                                      color = condition, fill = condition,
                                      shape = condition, group = condition)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
      geom_line(linewidth = 0.5) +
      geom_point(size = 4) +
      scale_color_manual(values = c("CTRL" = "#F8766D", "VPA" = "#00BFC4"),
                         labels = c("CTRL" = "CTRL", "VPA" = "VPA")) +
      scale_fill_manual(values  = c("CTRL" = "#F8766D", "VPA" = "#00BFC4"),
                        labels  = c("CTRL" = "CTRL", "VPA" = "VPA")) +
      scale_shape_manual(values = c("CTRL" = 16, "VPA" = 17),
                         labels = c("CTRL" = "CTRL", "VPA" = "VPA")) +
      scale_x_continuous(breaks = unique(group_summary_2$bin_numeric),
                         labels = unique(group_summary_2$bin_numeric)) +
      labs(x = "Time window", y = paste("Mean", y_label),
           color = "Condition", fill = "Condition", shape = "Condition") +
      theme_minimal(base_size = 18, base_family = "sans") +
      theme(
        plot.title        = element_blank(),
        axis.line         = element_line(color = "black", linewidth = 0.8),
        axis.ticks        = element_line(color = "black", linewidth = 0.8),
        axis.ticks.length = unit(0.2, "cm"),
        axis.title.x      = element_text(size = 26, face = "italic", margin = margin(t = 15)),
        axis.title.y      = element_text(size = 26, face = "italic", margin = margin(r = 15)),
        axis.text.x       = element_text(size = 18, color = "black"),
        axis.text.y       = element_text(size = 18, color = "black"),
        legend.title      = element_text(size = 24, face = "bold"),
        legend.text       = element_text(size = 24),
        legend.position   = "bottom",
        strip.text        = element_text(size = 26, face = "bold"),
        panel.grid.major  = element_line(color = "grey90", linewidth = 0.4),
        panel.grid.minor  = element_blank(),
        panel.spacing     = unit(2, "lines"),
        panel.border      = element_blank(),
        panel.background  = element_rect(fill = "white", color = NA),
        plot.background   = element_rect(fill = "white", color = NA)
      ) +
      facet_wrap(~ cluster_membership_2,
                 labeller = labeller(cluster_membership_2 = c("0" = "Cluster 0", "1" = "Cluster 1")))
    
    if (save_plot)
      ggsave(file.path(output_dir, paste0(feature_name, "_2clusters.png")),
             plot = p2, width = 14, height = 7, dpi = 300, bg = "white")
  }
  
  # 3-cluster ribbon plot
  subject_summary_3 <- ddply(
    data_complete,
    .(id, bin, condition, cluster_membership_3),
    function(df) {
      x <- df[[feature_name]]; n <- sum(!is.na(x))
      data.frame(average_value = mean(x, na.rm = TRUE), n = n)
    }
  )
  
  group_summary_3 <- ddply(
    subject_summary_3,
    .(bin, condition, cluster_membership_3),
    function(df) {
      n_subjects <- length(df$average_value)
      mean_val   <- mean(df$average_value, na.rm = TRUE)
      se_val     <- ifelse(n_subjects > 1, sd(df$average_value, na.rm = TRUE) / sqrt(n_subjects), 0)
      data.frame(mean_value = mean_val, n_subjects = n_subjects, se_value = se_val,
                 lower = mean_val - se_val, upper = mean_val + se_val)
    }
  )
  group_summary_3 <- na.omit(group_summary_3)
  group_summary_3$bin_numeric <- as.numeric(as.character(group_summary_3$bin))
  
  if (nrow(group_summary_3) > 0) {
    p3 <- ggplot(group_summary_3, aes(x = bin_numeric, y = mean_value,
                                      color = condition, fill = condition,
                                      shape = condition, group = condition)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.25, color = NA) +
      geom_line(linewidth = 0.75) +
      geom_point(size = 4) +
      scale_color_manual(values = c("CTRL" = "#F8766D", "VPA" = "#00BFC4"),
                         labels = c("CTRL" = "CTRL", "VPA" = "VPA")) +
      scale_fill_manual(values  = c("CTRL" = "#F8766D", "VPA" = "#00BFC4"),
                        labels  = c("CTRL" = "CTRL", "VPA" = "VPA")) +
      scale_shape_manual(values = c("CTRL" = 16, "VPA" = 17),
                         labels = c("CTRL" = "CTRL", "VPA" = "VPA")) +
      scale_x_continuous(breaks = unique(group_summary_3$bin_numeric),
                         labels = unique(group_summary_3$bin_numeric)) +
      labs(x = "Time window", y = paste("Mean", y_label),
           color = "Condition", fill = "Condition", shape = "Condition") +
      theme_minimal(base_size = 18, base_family = "sans") +
      theme(
        plot.title        = element_blank(),
        axis.line         = element_line(color = "black", linewidth = 1),
        axis.ticks        = element_line(color = "black", linewidth = 0.8),
        axis.ticks.length = unit(0.2, "cm"),
        axis.title.x      = element_text(size = 20, face = "italic", margin = margin(t = 15)),
        axis.title.y      = element_text(size = 20, face = "italic", margin = margin(r = 15)),
        axis.text.x       = element_text(size = 18, face = "bold", color = "black"),
        axis.text.y       = element_text(size = 18, face = "bold", color = "black"),
        legend.title      = element_text(size = 18, face = "bold"),
        legend.text       = element_text(size = 16),
        legend.position   = "bottom",
        strip.text        = element_text(size = 20, face = "bold"),
        panel.grid.major  = element_line(color = "grey90", linewidth = 0.3),
        panel.grid.minor  = element_blank(),
        panel.spacing     = unit(1.5, "lines"),
        panel.border      = element_blank(),
        panel.background  = element_rect(fill = "white", color = NA),
        plot.background   = element_rect(fill = "white", color = NA)
      ) +
      facet_wrap(~ cluster_membership_3, nrow = 1,
                 labeller = labeller(cluster_membership_3 = c("0" = "Cluster 0",
                                                              "1" = "Cluster 1",
                                                              "2" = "Cluster 2")))
    
    if (save_plot)
      ggsave(file.path(output_dir, paste0(feature_name, "_3clusters_ribbon.png")),
             plot = p3, width = 18, height = 7, dpi = 300, bg = "white")
  }
  
  return(list(plot_2clusters = p2, plot_3clusters = p3,
              data_2clusters = group_summary_2, data_3clusters = group_summary_3))
}

# run for all features
all_results      <- list()
successful_plots <- 0
failed_plots     <- 0

for (i in seq_along(features)) {
  feature <- features[i]
  cat("[", i, "/", length(features), "]", feature, "\n")
  
  result <- tryCatch({
    plot_feature_with_ribbons(t, feature, save_plot = TRUE)
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

cat("done:", successful_plots, "plots saved,", failed_plots, "failed\n")

# feature groups for combined panels
key_features_temporal  <- c("Attack_time", "Slope")
key_features_energy    <- c("F0_Mean", "F0_Mag_Mean", "F0_Skewness")
key_features_frequency <- c("F1_Mag_Mean", "F1_F0_Ratio", "F2_Mag_Mean", "F2_F0_Ratio")

# extract legend from a ggplot object
get_legend <- function(myggplot) {
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  if (length(leg) > 0) tmp$grobs[[leg]] else NULL
}

# combined panels with shared legend and panel labels (A, B, C...)
create_combined_plot <- function(feature_list, domain_name, n_cols = 2) {
  grob_list   <- list()
  legend_plot <- NULL
  labels      <- LETTERS[seq_along(feature_list)]
  
  for (i in seq_along(feature_list)) {
    feature <- feature_list[i]
    
    if (feature %in% names(all_results) && !is.null(all_results[[feature]]$plot_2clusters)) {
      
      if (i == 1) {
        legend_plot <- all_results[[feature]]$plot_2clusters +
          theme(legend.position  = "bottom",
                legend.title     = element_text(size = 24, face = "bold"),
                legend.text      = element_text(size = 22),
                legend.key.size  = unit(1.8, "cm"),
                legend.key.width = unit(2.5, "cm"),
                legend.spacing.x = unit(1, "cm"),
                legend.box.spacing = unit(0.12, "cm"),
                legend.margin    = margin(t = 5, b = 5))
      }
      
      p <- all_results[[feature]]$plot_2clusters +
        theme(axis.title      = element_text(size = 18, face = "italic"),
              axis.text       = element_text(size = 16),
              strip.text      = element_text(size = 22, face = "bold"),
              legend.position = "none",
              plot.margin     = unit(c(1.2, 0.5, 0.5, 0.5), "cm")) +
        labs(tag = labels[i]) +
        theme(plot.tag          = element_text(size = 24, face = "bold"),
              plot.tag.position = c(0.02, 0.98))
      
      grob_list[[i]] <- ggplotGrob(p)
    }
  }
  
  if (length(grob_list) == 0) return(NULL)
  
  shared_legend <- get_legend(legend_plot)
  n_rows        <- ceiling(length(grob_list) / n_cols)
  plot_height   <- 7 * n_rows + 1.2
  
  combined_dir <- file.path(output_dir, "Combined_Panels")
  if (!dir.exists(combined_dir)) dir.create(combined_dir, recursive = TRUE)
  
  filename_png <- file.path(combined_dir, paste0("Combined_", domain_name, "_Features_Labeled.png"))
  filename_pdf <- file.path(combined_dir, paste0("Combined_", domain_name, "_Features_Labeled.pdf"))
  
  png(filename_png, width = 22, height = plot_height, units = "in", res = 300, bg = "white")
  grid.arrange(
    arrangeGrob(grobs = grob_list, ncol = n_cols,
                top = textGrob(paste(domain_name, "Domain Features"),
                               gp = gpar(fontsize = 13, fontface = "bold"))),
    shared_legend,
    nrow = 2, heights = c(10, 0.65)
  )
  dev.off()
  
  pdf(filename_pdf, width = 22, height = plot_height, bg = "white")
  grid.arrange(
    arrangeGrob(grobs = grob_list, ncol = n_cols,
                top = textGrob(paste(domain_name, "Domain Features"),
                               gp = gpar(fontsize = 13, fontface = "bold"))),
    shared_legend,
    nrow = 2, heights = c(10, 0.65)
  )
  dev.off()
  
  invisible(NULL)
}

create_combined_plot(key_features_temporal,  "Temporal",  n_cols = 2)
create_combined_plot(key_features_energy,    "Energy",    n_cols = 2)
create_combined_plot(key_features_frequency, "Frequency", n_cols = 2)