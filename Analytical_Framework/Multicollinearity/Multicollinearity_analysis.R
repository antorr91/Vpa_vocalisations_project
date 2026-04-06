###Antonella Torrisi - Script Multicollinearity analysis

rm(list=ls())
library(extrafont)
loadfonts(device = "win") 


library(plyr)
library(dplyr)
library(ggplot2) 
library(lme4)
library(lmerTest)
library(emmeans)
library(readr)
library(tidyr)
library(effectsize)
library(corrplot)      # For correlation plots
library(car)           # For VIF calculation
library(psych)         # For correlation analysis
library(RColorBrewer)  # For better colour palettes
library(pheatmap)      # For heatmaps

# Load data
setwd("/path/to/project")
#setwd("C:\Users\anton\VPA_vocalisations_project\Results_Clustering")


data_path <- file.path("data", "full_dataset_vpa_ctrl.csv")
t <- read_csv(data_path)

# check if there are NA data
rm_NA <- function(data) {
  na_counts <- sapply(data, function(x) sum(is.na(x)))
  cat("Number of NA per variable: \n")
  print(na_counts)
  
  na_rows <- which(!complete.cases(data))
  cat("Rows with NA:\n")
  print(head(na_rows, 10))
  
  cleaned <- data[complete.cases(data), ]
  cat("\nRemoved", length(na_rows), "rows with NA.\n")
  
  return(cleaned)
}

t <- rm_NA(t)

# Convert factors
t <- t %>% 
  mutate(
    id = as.factor(id),
    condition = as.factor(condition),
    cluster_membership_2 = as.factor(cluster_membership_2),
    cluster_membership_3 = as.factor(cluster_membership_3)
  )

# Define all acoustic features
features <- c(
  "Duration_call", "F0_Mean", "F0_Std", "F0_Skewness", "F0_Kurtosis", 
  "F0_Bandwidth", "F0_1st_Order_Diff", "F0_Slope", "F0_Mag_Mean", 
  "F1_Mag_Mean", "F2_Mag_Mean", "F1_F0_Ratio", "F2_F0_Ratio", 
  "Spectral_Centroid_Mean", "Spectral_Centroid_Std", "RMS_Mean", 
  "RMS_Std", "Slope", "Attack_magnitude", "Attack_time"
)

# Check which features actually exist in the dataset
available_features <- features[features %in% names(t)]
missing_features <- features[!features %in% names(t)]

cat("Available features:", length(available_features), "\n")
cat("Missing features:", length(missing_features), ":", paste(missing_features, collapse = ", "), "\n")

# Create subset with only complete cases for all available features
feature_data <- t[, c(available_features, "condition", "cluster_membership_2", "id")]
feature_data_complete <- feature_data[complete.cases(feature_data), ]

cat("\nOriginal dataset size:", nrow(t))
cat("\nComplete cases for all features:", nrow(feature_data_complete))
cat("\nData retention:", round(nrow(feature_data_complete)/nrow(t)*100, 2), "%\n")

# Correlation analysis


# Extract only numeric features for correlation
numeric_features <- feature_data_complete[, available_features]

# Calculate correlation matrix
cor_matrix <- cor(numeric_features, use = "complete.obs")
cat("Correlation matrix calculated for", ncol(numeric_features), "features\n")

# Create output directory
output_dir <- file.path(getwd(), "VPA_correlation_results")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save correlation matrix
write.csv(cor_matrix, file.path(output_dir, "correlation_matrix.csv"))

# Visualisation

# 1. Basic correlation plot
pdf(file.path(output_dir, "correlation_plot_basic.pdf"), width = 12, height = 10)
corrplot(cor_matrix, 
         method = "color", 
         type = "upper", 
         order = "hclust",
         tl.cex = 0.8, 
         tl.col = "black",
         title = "Correlation Matrix - Acoustic Features",
         mar = c(0,0,1,0))
dev.off()

# 2. Enhanced heatmap with hierarchical clustering
pdf(file.path(output_dir, "correlation_heatmap_enhanced.pdf"), width = 14, height = 12)
pheatmap(cor_matrix,
         clustering_distance_rows = "euclidean",
         clustering_distance_cols = "euclidean",
         clustering_method = "ward.D2",
         color = colorRampPalette(c("blue", "white", "red"))(100),
         breaks = seq(-1, 1, length.out = 101),
         main = "Hierarchical Clustering of Acoustic Features",
         fontsize = 8,
         cellwidth = 15,
         cellheight = 15)
dev.off()

# 3. Correlation plot with values
pdf(file.path(output_dir, "correlation_plot_with_values.pdf"), width = 14, height = 12)
corrplot(cor_matrix, 
         method = "color", 
         type = "lower",
         order = "hclust",
         tl.cex = 0.7,
         tl.col = "black",
         addCoef.col = "black",
         number.cex = 0.5,
         title = "Correlation Matrix with Values",
         mar = c(0,0,1,0))
dev.off()

# Multicollinearity detection

# Find highly correlated pairs
high_cor_threshold <- 0.8
high_cor_pairs <- which(abs(cor_matrix) > high_cor_threshold & abs(cor_matrix) < 1, arr.ind = TRUE)

if (nrow(high_cor_pairs) > 0) {
  cat("Highly correlated pairs (|r| >", high_cor_threshold, "):\n")
  high_cor_results <- data.frame(
    Feature1 = rownames(cor_matrix)[high_cor_pairs[,1]],
    Feature2 = colnames(cor_matrix)[high_cor_pairs[,2]],
    Correlation = cor_matrix[high_cor_pairs],
    stringsAsFactors = FALSE
  )
  
  # Remove duplicates (since correlation matrix is symmetric)
  high_cor_results <- high_cor_results[high_cor_results$Feature1 != high_cor_results$Feature2, ]
  high_cor_results <- high_cor_results[!duplicated(apply(high_cor_results[,1:2], 1, function(x) paste(sort(x), collapse = "_"))), ]
  
  print(high_cor_results)
  write.csv(high_cor_results, file.path(output_dir, "high_correlations.csv"), row.names = FALSE)
} else {
  cat("No highly correlated pairs found with threshold", high_cor_threshold, "\n")
}

# Feature selection based on correlations



# Function to remove highly correlated features
remove_highly_correlated <- function(cor_matrix, threshold = 0.8) {
  # Find features to remove
  high_cor <- findCorrelation(cor_matrix, cutoff = threshold)
  
  if (length(high_cor) > 0) {
    features_to_remove <- colnames(cor_matrix)[high_cor]
    remaining_features <- colnames(cor_matrix)[-high_cor]
    
    cat("Features to remove due to high correlation (threshold =", threshold, "):\n")
    cat(paste(features_to_remove, collapse = ", "), "\n")
    cat("\nRemaining features:", length(remaining_features), "\n")
    cat(paste(remaining_features, collapse = ", "), "\n")
    
    return(list(
      to_remove = features_to_remove,
      remaining = remaining_features,
      reduced_matrix = cor_matrix[-high_cor, -high_cor]
    ))
  } else {
    cat("No features need to be removed with threshold", threshold, "\n")
    return(list(
      to_remove = character(0),
      remaining = colnames(cor_matrix),
      reduced_matrix = cor_matrix
    ))
  }
}

# Apply feature selection with different thresholds
thresholds <- c(0.9, 0.8, 0.7)
feature_selection_results <- list()

for (thresh in thresholds) {
  cat("\n--- Threshold:", thresh, "---\n")
  result <- remove_highly_correlated(cor_matrix, thresh)
  feature_selection_results[[paste0("threshold_", thresh)]] <- result
  
  # Save reduced correlation matrix
  if (length(result$remaining) > 1) {
    write.csv(result$reduced_matrix, 
              file.path(output_dir, paste0("correlation_matrix_reduced_", thresh, ".csv")))
  }
}

# Variance inflation factor (VIF) analysis

# Prepare data for VIF calculation
# We need to create a simple linear model first
vif_data <- feature_data_complete

# Create a dummy outcome variable for VIF calculation
# We'll use the first feature as outcome and others as predictors
if (length(available_features) > 1) {
  tryCatch({
    # Use a simple linear model for VIF calculation
    vif_formula <- as.formula(paste(available_features[1], "~", paste(available_features[-1], collapse = " + ")))
    vif_model <- lm(vif_formula, data = vif_data)
    
    # Calculate VIF
    vif_values <- vif(vif_model)
    
    # Create VIF results dataframe
    vif_results <- data.frame(
      Feature = names(vif_values),
      VIF = as.numeric(vif_values),
      stringsAsFactors = FALSE
    )
    vif_results <- vif_results[order(vif_results$VIF, decreasing = TRUE), ]
    
    print(vif_results)
    
    # Flag problematic features
    severe_multicollinearity <- vif_results$Feature[vif_results$VIF > 10]
    moderate_multicollinearity <- vif_results$Feature[vif_results$VIF > 5 & vif_results$VIF <= 10]
    
    if (length(severe_multicollinearity) > 0) {
      cat("\nFeatures with severe multicollinearity (VIF > 10):\n")
      cat(paste(severe_multicollinearity, collapse = ", "), "\n")
    }
    
    if (length(moderate_multicollinearity) > 0) {
      cat("\nFeatures with moderate multicollinearity (VIF 5-10):\n")
      cat(paste(moderate_multicollinearity, collapse = ", "), "\n")
    }
    
    write.csv(vif_results, file.path(output_dir, "vif_analysis.csv"), row.names = FALSE)
    
  }, error = function(e) {
    cat("Error in VIF calculation:", e$message, "\n")
    cat("This might be due to perfect multicollinearity or insufficient data.\n")
  })
}

# Recommended feature set



# Based on correlation analysis (threshold 0.8)
recommended_features_08 <- feature_selection_results$threshold_0.8$remaining

# Based on correlation analysis (threshold 0.7) - more conservative
recommended_features_07 <- feature_selection_results$threshold_0.7$remaining

cat("Recommended features (correlation threshold 0.8):\n")
cat("Number of features:", length(recommended_features_08), "\n")
cat("Features:", paste(recommended_features_08, collapse = ", "), "\n\n")

cat("Recommended features (correlation threshold 0.7 - more conservative):\n")
cat("Number of features:", length(recommended_features_07), "\n")
cat("Features:", paste(recommended_features_07, collapse = ", "), "\n\n")

# Save recommended feature sets
recommended_sets <- list(
  all_features = available_features,
  reduced_08 = recommended_features_08,
  reduced_07 = recommended_features_07
)

# Save as JSON-like format for easy reading
sink(file.path(output_dir, "recommended_feature_sets.txt"))
cat("=== RECOMMENDED FEATURE SETS ===\n\n")
cat("ALL FEATURES (", length(recommended_sets$all_features), " features):\n")
cat(paste(recommended_sets$all_features, collapse = ", "), "\n\n")
cat("REDUCED SET - Correlation threshold 0.8 (", length(recommended_sets$reduced_08), " features):\n")
cat(paste(recommended_sets$reduced_08, collapse = ", "), "\n\n")
cat("REDUCED SET - Correlation threshold 0.7 (", length(recommended_sets$reduced_07), " features):\n")
cat(paste(recommended_sets$reduced_07, collapse = ", "), "\n\n")


# Enhanced discriminative power analysis


# Enhanced discriminative power analysis including cluster comparisons
discriminative_power <- data.frame(
  Feature = available_features,
  Cohens_d_Condition = sapply(available_features, function(f) calculate_cohens_d(feature_data_complete, f, "condition")),
  Cohens_d_Cluster = sapply(available_features, function(f) calculate_cohens_d(feature_data_complete, f, "cluster_membership_2")),
  stringsAsFactors = FALSE
)

# Add within condition cluster comparisons
discriminative_power$Cohens_d_Cluster_within_VPA <- sapply(available_features, function(f) {
  vpa_subset <- feature_data_complete[feature_data_complete$condition == "VPA", ]
  if (nrow(vpa_subset) > 10) {
    calculate_cohens_d(vpa_subset, f, "cluster_membership_2")
  } else {
    NA
  }
})

discriminative_power$Cohens_d_Cluster_within_CTRL <- sapply(available_features, function(f) {
  ctrl_subset <- feature_data_complete[feature_data_complete$condition == "CTRL", ]
  if (nrow(ctrl_subset) > 10) {
    calculate_cohens_d(ctrl_subset, f, "cluster_membership_2")
  } else {
    NA
  }
})

# Add within cluster condition comparisons
discriminative_power$Cohens_d_Condition_within_Cluster0 <- sapply(available_features, function(f) {
  cluster0_subset <- feature_data_complete[feature_data_complete$cluster_membership_2 == "0", ]
  if (nrow(cluster0_subset) > 10) {
    calculate_cohens_d(cluster0_subset, f, "condition")
  } else {
    NA
  }
})

discriminative_power$Cohens_d_Condition_within_Cluster1 <- sapply(available_features, function(f) {
  cluster1_subset <- feature_data_complete[feature_data_complete$cluster_membership_2 == "1", ]
  if (nrow(cluster1_subset) > 10) {
    calculate_cohens_d(cluster1_subset, f, "condition")
  } else {
    NA
  }
})

# Add interpretations for all comparisons
discriminative_power$Condition_Effect_Size <- ifelse(
  is.na(discriminative_power$Cohens_d_Condition), "Not calculated",
  ifelse(discriminative_power$Cohens_d_Condition < 0.2, "Negligible (< 0.2)",
         ifelse(discriminative_power$Cohens_d_Condition < 0.5, "Small (0.2-0.5)",
                ifelse(discriminative_power$Cohens_d_Condition < 0.8, "Medium (0.5-0.8)",
                       "Large (≥ 0.8)"))))

discriminative_power$Cluster_Effect_Size <- ifelse(
  is.na(discriminative_power$Cohens_d_Cluster), "Not calculated",
  ifelse(discriminative_power$Cohens_d_Cluster < 0.2, "Negligible (< 0.2)",
         ifelse(discriminative_power$Cohens_d_Cluster < 0.5, "Small (0.2-0.5)",
                ifelse(discriminative_power$Cohens_d_Cluster < 0.8, "Medium (0.5-0.8)",
                       "Large (≥ 0.8)"))))

# Sort by overall discriminative power (average of condition and cluster effects)
discriminative_power$Average_Effect <- rowMeans(discriminative_power[, c("Cohens_d_Condition", "Cohens_d_Cluster")], na.rm = TRUE)
discriminative_power <- discriminative_power[order(discriminative_power$Average_Effect, decreasing = TRUE), ]



print(discriminative_power[, 1:8])  # Show first 8 columns for readability
write.csv(discriminative_power, file.path(output_dir, "enhanced_discriminative_power_analysis.csv"), row.names = FALSE)

# Correlation analysis by condition


# Split data by condition
vpa_data <- feature_data_complete[feature_data_complete$condition == "VPA", available_features]
ctrl_data <- feature_data_complete[feature_data_complete$condition == "CTRL", available_features]

# Calculate correlations for each condition
cor_vpa <- cor(vpa_data, use = "complete.obs")
cor_ctrl <- cor(ctrl_data, use = "complete.obs")

# Calculate difference in correlations
cor_difference_condition <- cor_vpa - cor_ctrl

cat("Correlation matrices calculated separately for VPA and CTRL conditions\n")
cat("VPA data points:", nrow(vpa_data), "\n")
cat("CTRL data points:", nrow(ctrl_data), "\n")

# Save condition-specific correlations
write.csv(cor_vpa, file.path(output_dir, "correlation_matrix_VPA.csv"))
write.csv(cor_ctrl, file.path(output_dir, "correlation_matrix_CTRL.csv"))
write.csv(cor_difference_condition, file.path(output_dir, "correlation_difference_VPA_minus_CTRL.csv"))

# Correlation analysis by cluster



# Split data by cluster
cluster0_data <- feature_data_complete[feature_data_complete$cluster_membership_2 == "0", available_features]
cluster1_data <- feature_data_complete[feature_data_complete$cluster_membership_2 == "1", available_features]

# Calculate correlations for each cluster
cor_cluster0 <- cor(cluster0_data, use = "complete.obs")
cor_cluster1 <- cor(cluster1_data, use = "complete.obs")

# Calculate difference in correlations
cor_difference_cluster <- cor_cluster1 - cor_cluster0

cat("Correlation matrices calculated separately for Cluster 0 and Cluster 1\n")
cat("Cluster 0 data points:", nrow(cluster0_data), "\n")
cat("Cluster 1 data points:", nrow(cluster1_data), "\n")

# Save cluster-specific correlations
write.csv(cor_cluster0, file.path(output_dir, "correlation_matrix_Cluster0.csv"))
write.csv(cor_cluster1, file.path(output_dir, "correlation_matrix_Cluster1.csv"))
write.csv(cor_difference_cluster, file.path(output_dir, "correlation_difference_Cluster1_minus_Cluster0.csv"))

# Correlation analysis by condition x cluster

cat("\n=== CORRELATION ANALYSIS BY CONDITION × CLUSTER ===\n")

# Split data by condition AND cluster (4 groups)
vpa_cluster0_data <- feature_data_complete[feature_data_complete$condition == "VPA" & feature_data_complete$cluster_membership_2 == "0", available_features]
vpa_cluster1_data <- feature_data_complete[feature_data_complete$condition == "VPA" & feature_data_complete$cluster_membership_2 == "1", available_features]
ctrl_cluster0_data <- feature_data_complete[feature_data_complete$condition == "CTRL" & feature_data_complete$cluster_membership_2 == "0", available_features]
ctrl_cluster1_data <- feature_data_complete[feature_data_complete$condition == "CTRL" & feature_data_complete$cluster_membership_2 == "1", available_features]

# Check sample sizes
cat("Sample sizes for condition × cluster combinations:\n")
cat("VPA + Cluster 0:", nrow(vpa_cluster0_data), "\n")
cat("VPA + Cluster 1:", nrow(vpa_cluster1_data), "\n")
cat("CTRL + Cluster 0:", nrow(ctrl_cluster0_data), "\n")
cat("CTRL + Cluster 1:", nrow(ctrl_cluster1_data), "\n")

# Calculate correlations for each combination (only if enough data)
min_sample_size <- 10  # Minimum sample size for reliable correlation

correlations_by_group <- list()

if (nrow(vpa_cluster0_data) >= min_sample_size) {
  correlations_by_group$VPA_Cluster0 <- cor(vpa_cluster0_data, use = "complete.obs")
  write.csv(correlations_by_group$VPA_Cluster0, file.path(output_dir, "correlation_matrix_VPA_Cluster0.csv"))
}

if (nrow(vpa_cluster1_data) >= min_sample_size) {
  correlations_by_group$VPA_Cluster1 <- cor(vpa_cluster1_data, use = "complete.obs")
  write.csv(correlations_by_group$VPA_Cluster1, file.path(output_dir, "correlation_matrix_VPA_Cluster1.csv"))
}

if (nrow(ctrl_cluster0_data) >= min_sample_size) {
  correlations_by_group$CTRL_Cluster0 <- cor(ctrl_cluster0_data, use = "complete.obs")
  write.csv(correlations_by_group$CTRL_Cluster0, file.path(output_dir, "correlation_matrix_CTRL_Cluster0.csv"))
}

if (nrow(ctrl_cluster1_data) >= min_sample_size) {
  correlations_by_group$CTRL_Cluster1 <- cor(ctrl_cluster1_data, use = "complete.obs")
  write.csv(correlations_by_group$CTRL_Cluster1, file.path(output_dir, "correlation_matrix_CTRL_Cluster1.csv"))
}

# Visualisation of all correlations

# 1. Condition-specific correlations
pdf(file.path(output_dir, "correlation_by_condition.pdf"), width = 16, height = 6)
par(mfrow = c(1, 3))

corrplot(cor_vpa, 
         method = "color", 
         type = "upper",
         tl.cex = 0.6,
         title = "VPA Condition",
         mar = c(0,0,1,0))

corrplot(cor_ctrl, 
         method = "color", 
         type = "upper",
         tl.cex = 0.6,
         title = "CTRL Condition", 
         mar = c(0,0,1,0))

corrplot(cor_difference_condition, 
         method = "color", 
         type = "upper",
         tl.cex = 0.6,
         title = "Difference (VPA - CTRL)",
         mar = c(0,0,1,0))

dev.off()

# 2. Cluster-specific correlations
pdf(file.path(output_dir, "correlation_by_cluster.pdf"), width = 16, height = 6)
par(mfrow = c(1, 3))

corrplot(cor_cluster0, 
         method = "color", 
         type = "upper",
         tl.cex = 0.6,
         title = "Cluster 0",
         mar = c(0,0,1,0))

corrplot(cor_cluster1, 
         method = "color", 
         type = "upper",
         tl.cex = 0.6,
         title = "Cluster 1", 
         mar = c(0,0,1,0))

corrplot(cor_difference_cluster, 
         method = "color", 
         type = "upper",
         tl.cex = 0.6,
         title = "Difference (Cluster1 - Cluster0)",
         mar = c(0,0,1,0))

dev.off()

# 3. Combined visualisation: all 4 groups if data is sufficient
if (length(correlations_by_group) >= 2) {
  pdf(file.path(output_dir, "correlation_by_condition_and_cluster.pdf"), width = 20, height = 15)
  
  # Determine layout based on available groups
  n_groups <- length(correlations_by_group)
  if (n_groups == 4) {
    par(mfrow = c(2, 2))
  } else if (n_groups == 3) {
    par(mfrow = c(2, 2))
  } else {
    par(mfrow = c(1, 2))
  }
  
  for (group_name in names(correlations_by_group)) {
    corrplot(correlations_by_group[[group_name]], 
             method = "color", 
             type = "upper",
             tl.cex = 0.6,
             title = gsub("_", " + ", group_name),
             mar = c(0,0,1,0))
  }
  
  dev.off()
}

# Analysis of correlation differences

cat("\n=== ANALYSIS OF CORRELATION DIFFERENCES ===\n")

# Function to find the most different correlations between groups
find_most_different_correlations <- function(cor1, cor2, group1_name, group2_name, top_n = 10) {
  diff_matrix <- cor1 - cor2
  
  # Extract upper triangle (to avoid duplicates)
  diff_matrix[lower.tri(diff_matrix, diag = TRUE)] <- NA
  
  # Find positions of largest differences
  max_diff_idx <- which(abs(diff_matrix) == max(abs(diff_matrix), na.rm = TRUE), arr.ind = TRUE)
  
  # Get all differences and sort them
  differences <- data.frame()
  for (i in 1:(nrow(diff_matrix)-1)) {
    for (j in (i+1):ncol(diff_matrix)) {
      if (!is.na(diff_matrix[i,j])) {
        differences <- rbind(differences, data.frame(
          Feature1 = rownames(diff_matrix)[i],
          Feature2 = colnames(diff_matrix)[j],
          Correlation_Group1 = cor1[i,j],
          Correlation_Group2 = cor2[i,j],
          Difference = diff_matrix[i,j],
          Abs_Difference = abs(diff_matrix[i,j]),
          Group1 = group1_name,
          Group2 = group2_name,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  # Sort by absolute difference and return top N
  differences <- differences[order(differences$Abs_Difference, decreasing = TRUE), ]
  return(differences[1:min(top_n, nrow(differences)), ])
}

# Find biggest correlation differences between conditions
if (exists("cor_vpa") && exists("cor_ctrl")) {
  diff_condition <- find_most_different_correlations(cor_vpa, cor_ctrl, "VPA", "CTRL", 15)
  cat("Top 15 correlation differences between VPA and CTRL:\n")
  print(diff_condition)
  write.csv(diff_condition, file.path(output_dir, "top_correlation_differences_condition.csv"), row.names = FALSE)
}

# Find biggest correlation differences between clusters
if (exists("cor_cluster0") && exists("cor_cluster1")) {
  diff_cluster <- find_most_different_correlations(cor_cluster1, cor_cluster0, "Cluster1", "Cluster0", 15)
  cat("\nTop 15 correlation differences between Cluster 1 and Cluster 0:\n")
  print(diff_cluster)
  write.csv(diff_cluster, file.path(output_dir, "top_correlation_differences_cluster.csv"), row.names = FALSE)
}

# Compare correlations across all 4 groups (if data is sufficient)
if (length(correlations_by_group) >= 2) {
  cat("\n=== CROSS-GROUP CORRELATION COMPARISON ===\n")
  
  group_names <- names(correlations_by_group)
  all_comparisons <- data.frame()
  
  # Compare all pairs of groups
  for (i in 1:(length(group_names)-1)) {
    for (j in (i+1):length(group_names)) {
      group1 <- group_names[i]
      group2 <- group_names[j]
      
      comparison <- find_most_different_correlations(
        correlations_by_group[[group1]], 
        correlations_by_group[[group2]], 
        group1, group2, 5
      )
      
      all_comparisons <- rbind(all_comparisons, comparison)
    }
  }
  
  # Sort by absolute difference
  all_comparisons <- all_comparisons[order(all_comparisons$Abs_Difference, decreasing = TRUE), ]
  
  cat("Top correlation differences across all group combinations:\n")
  print(head(all_comparisons, 20))
  write.csv(all_comparisons, file.path(output_dir, "all_group_correlation_comparisons.csv"), row.names = FALSE)
}

# Combine correlation info with discriminative power
feature_analysis <- merge(discriminative_power, 
                          data.frame(Feature = available_features, stringsAsFactors = FALSE),
                          by = "Feature", all = TRUE)

# Find highly correlated pairs in the full dataset
high_cor_pairs_detailed <- data.frame()
for (i in 1:(nrow(cor_matrix)-1)) {
  for (j in (i+1):ncol(cor_matrix)) {
    if (abs(cor_matrix[i,j]) > 0.7) {  # Lower threshold for detailed analysis
      high_cor_pairs_detailed <- rbind(high_cor_pairs_detailed, data.frame(
        Feature1 = rownames(cor_matrix)[i],
        Feature2 = colnames(cor_matrix)[j],
        Correlation = cor_matrix[i,j],
        Abs_Correlation = abs(cor_matrix[i,j]),
        Feature1_Discriminative = discriminative_power$Cohens_d_Condition[discriminative_power$Feature == rownames(cor_matrix)[i]],
        Feature2_Discriminative = discriminative_power$Cohens_d_Condition[discriminative_power$Feature == colnames(cor_matrix)[j]],
        stringsAsFactors = FALSE
      ))
    }
  }
}

if (nrow(high_cor_pairs_detailed) > 0) {
  # Add recommendations
  high_cor_pairs_detailed$Recommendation <- ifelse(
    high_cor_pairs_detailed$Feature1_Discriminative > high_cor_pairs_detailed$Feature2_Discriminative,
    paste0("Keep ", high_cor_pairs_detailed$Feature1, " (more discriminative)"),
    ifelse(high_cor_pairs_detailed$Feature2_Discriminative > high_cor_pairs_detailed$Feature1_Discriminative,
           paste0("Keep ", high_cor_pairs_detailed$Feature2, " (more discriminative)"),
           "Both have similar discriminative power - choose based on interpretability")
  )
  
  cat("Highly correlated pairs with discriminative power analysis:\n")
  print(high_cor_pairs_detailed[, c("Feature1", "Feature2", "Abs_Correlation", "Feature1_Discriminative", "Feature2_Discriminative", "Recommendation")])
  
  write.csv(high_cor_pairs_detailed, file.path(output_dir, "smart_correlation_analysis.csv"), row.names = FALSE)
} else {
  cat("No highly correlated pairs found.\n")
}

# Smart feature selection

# Strategy 1: Remove less discriminative features from correlated pairs
features_to_remove_smart <- c()
if (nrow(high_cor_pairs_detailed) > 0) {
  for (i in 1:nrow(high_cor_pairs_detailed)) {
    pair <- high_cor_pairs_detailed[i, ]
    if (pair$Feature1_Discriminative < pair$Feature2_Discriminative) {
      features_to_remove_smart <- c(features_to_remove_smart, pair$Feature1)
    } else if (pair$Feature2_Discriminative < pair$Feature1_Discriminative) {
      features_to_remove_smart <- c(features_to_remove_smart, pair$Feature2)
    }
  }
  features_to_remove_smart <- unique(features_to_remove_smart)
}

smart_features <- available_features[!available_features %in% features_to_remove_smart]

cat("Smart feature selection based on discriminative power:\n")
cat("Features to remove (less discriminative in correlated pairs):", paste(features_to_remove_smart, collapse = ", "), "\n")
cat("Remaining features (", length(smart_features), "):", paste(smart_features, collapse = ", "), "\n")

# Summary statistics by condition

# Calculate means and SDs by condition for smart features
descriptive_stats <- feature_data_complete %>%
  group_by(condition) %>%
  summarise_at(vars(all_of(smart_features)), 
               list(mean = ~mean(., na.rm = TRUE), 
                    sd = ~sd(., na.rm = TRUE),
                    n = ~sum(!is.na(.)))) %>%
  ungroup()

write.csv(descriptive_stats, file.path(output_dir, "descriptive_stats_by_condition.csv"), row.names = FALSE)

# Also by cluster
descriptive_stats_cluster <- feature_data_complete %>%
  group_by(cluster_membership_2) %>%
  summarise_at(vars(all_of(smart_features)), 
               list(mean = ~mean(., na.rm = TRUE), 
                    sd = ~sd(., na.rm = TRUE),
                    n = ~sum(!is.na(.)))) %>%
  ungroup()

write.csv(descriptive_stats_cluster, file.path(output_dir, "descriptive_stats_by_cluster.csv"), row.names = FALSE)

# By condition AND cluster
descriptive_stats_both <- feature_data_complete %>%
  group_by(condition, cluster_membership_2) %>%
  summarise_at(vars(all_of(smart_features)), 
               list(mean = ~mean(., na.rm = TRUE), 
                    sd = ~sd(., na.rm = TRUE),
                    n = ~sum(!is.na(.)))) %>%
  ungroup()

write.csv(descriptive_stats_both, file.path(output_dir, "descriptive_stats_by_condition_and_cluster.csv"), row.names = FALSE)


recommended_sets <- list(
  all_features = available_features,
  correlation_reduced_08 = feature_selection_results$threshold_0.8$remaining,
  correlation_reduced_07 = feature_selection_results$threshold_0.7$remaining,
  smart_features = smart_features,
  top_discriminative = discriminative_power$Feature[1:min(10, nrow(discriminative_power))]
)

cat("All features (", length(recommended_sets$all_features), " features):\n")
cat("Correlation-based reduction 0.8 (", length(recommended_sets$correlation_reduced_08), " features):\n")
cat("Low correlation and high discriminative power (", length(recommended_sets$smart_features), " features):\n")
cat("Top discriminative (", length(recommended_sets$top_discriminative), " features):\n")