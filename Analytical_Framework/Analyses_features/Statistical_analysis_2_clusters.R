# Statistical analysis - lmm for all features, 2-cluster solution
# Antonella Torrisi

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

#library(car)
#library(ez)

# load data
setwd("/path/to/project")
#setwd("C:\Users\anton\VPA_vocalisations_project\Results_Clustering")
data_path <- file.path("data", "full_dataset_vpa_ctrl.csv")
t <- read_csv(data_path)
names(t)

# remove rows with missing values
rm_NA <- function(data) {
  na_counts <- sapply(data, function(x) sum(is.na(x)))
  # number of NA per variable
  print(na_counts)
  
  na_rows <- which(!complete.cases(data))
  # rows with NA
  print(head(na_rows, 10))
  
  cleaned <- data[complete.cases(data), ]
  # removed na rows
  
  return(cleaned)
}

t <- rm_NA(t)

vars_needed <- c("onsets_sec", "condition","id","cluster_membership_2", "cluster_membership_3")
t_clean <- t %>% filter(complete.cases(select(., all_of(vars_needed))))

t <- t %>%
  mutate(
    id                   = as.factor(id),
    condition            = as.factor(condition),
    cluster_membership_2 = as.factor(cluster_membership_2),
    cluster_membership_3 = as.factor(cluster_membership_3)
  )

cluster_by_cond_2 <- t %>%
  group_by(condition, cluster_membership_2) %>%
  summarise(n_call = n(), .groups = "drop")
print(cluster_by_cond_2)

cluster_by_cond_3 <- t %>%
  group_by(condition, cluster_membership_3) %>%
  summarise(n_call = n(), .groups = "drop")
print(cluster_by_cond_3)

# duration - preliminary model comparison
t$id                   <- as.factor(t$id)
t$bin                  <- as.factor(t$bin)
t$condition            <- as.factor(t$condition)
t$cluster_membership_3 <- as.factor(t$cluster_membership_3)
t$cluster_membership_2 <- as.factor(t$cluster_membership_2)

model_duration1 <- lmer(Duration_call ~ bin * condition * cluster_membership_2 + (1 | id), data = t)
model_duration2 <- lmer(Duration_call ~ bin * condition + (1 | id), data = t)
anova(model_duration2, model_duration1) # significant, retain model_duration1 (cluster significant)

model_duration3 <- lmer(Duration_call ~ bin * cluster_membership_2 + (1 | id), data = t)
anova(model_duration3, model_duration1) # significant, retain model_duration1 (condition significant)

vars_needed <- c("Duration_call", "bin", "condition", "cluster_membership_2", "id")
t_complete <- t[complete.cases(t[, vars_needed]), ]
model_duration1 <- lmer(Duration_call ~ bin * condition * cluster_membership_2 + (1 | id), data = t_complete)
model_duration4 <- lmer(Duration_call ~ condition * cluster_membership_2 + (1 | id), data = t_complete)
anova(model_duration4, model_duration1) # significant, retain model_duration1 (bin significant)

summary(model_duration1)

# features
features <- c(
  "Duration_call", "F0_Mean", "F0_Std", "F0_Skewness", "F0_Kurtosis",
  "F0_Bandwidth", "F0_1st_Order_Diff", "F0_Slope", "F0_Mag_Mean",
  "F1_Mag_Mean", "F2_Mag_Mean", "F1_F0_Ratio", "F2_F0_Ratio",
  "Spectral_Centroid_Mean", "Spectral_Centroid_Std", "RMS_Mean",
  "RMS_Std", "Slope", "Attack_magnitude", "Attack_time"
)

# approximate partial eta-squared from lrt
calculate_eta_squared <- function(anova_result) {
  if (is.na(anova_result$Chisq[2])) return(NA)
  chi_sq  <- anova_result$Chisq[2]
  df_diff <- anova_result$Df[2]
  chi_sq / (chi_sq + df_diff)
}

# tukey posthoc for condition and cluster effects
perform_tukey_test <- function(model, data, feature_name, comparison_type) {
  tryCatch({
    if (comparison_type == "Condition_Effect") {
      emm          <- emmeans(model, ~ condition)
      tukey_result <- pairs(emm, adjust = "tukey")
      tukey_summary <- summary(tukey_result)
      significant_comparisons <- tukey_summary[tukey_summary$p.value < 0.05, ]
      
      if (nrow(significant_comparisons) > 0) {
        result <- paste(significant_comparisons$contrast,
                        paste0("(p=", round(significant_comparisons$p.value, 4), ")"),
                        collapse = "; ")
      } else {
        result <- "no significant pairwise differences"
      }
      
    } else if (comparison_type == "Cluster_Effect") {
      emm          <- emmeans(model, ~ cluster_membership_2)
      tukey_result <- pairs(emm, adjust = "tukey")
      tukey_summary <- summary(tukey_result)
      significant_comparisons <- tukey_summary[tukey_summary$p.value < 0.05, ]
      
      if (nrow(significant_comparisons) > 0) {
        result <- paste(significant_comparisons$contrast,
                        paste0("(p=", round(significant_comparisons$p.value, 4), ")"),
                        collapse = "; ")
      } else {
        result <- "no significant pairwise differences"
      }
      
    } else {
      result <- "not applicable for this comparison"
    }
    
    return(result)
    
  }, error = function(e) {
    return(paste("error in tukey test:", e$message))
  })
}

# lmm for each feature: cluster, condition, bin effects
analyse_models_for_feature <- function(data, feature_name, alpha = 0.05) {
  
  cat("\n[", feature_name, "]\n")
  
  if (!feature_name %in% names(data)) {
    # feature not found
    return(NULL)
  }
  
  data$id                   <- as.factor(data$id)
  data$bin                  <- as.factor(data$bin)
  data$condition            <- as.factor(data$condition)
  data$cluster_membership_3 <- as.factor(data$cluster_membership_3)
  data$cluster_membership_2 <- as.factor(data$cluster_membership_2)
  data$bin <- relevel(data$bin, ref = "1")
  
  vars_needed   <- c(feature_name, "bin", "condition", "cluster_membership_2", "cluster_membership_3", "id")
  data_complete <- data[complete.cases(data[, vars_needed]), ]
  
  if (nrow(data_complete) == 0) {
    # no complete cases
    return(NULL)
  }
  
  results <- data.frame(
    Feature              = character(),
    Comparison           = character(),
    Full_Model           = character(),
    Reduced_Model        = character(),
    Chi_Square           = numeric(),
    Df                   = numeric(),
    Effect_size_eta_squared = numeric(),
    P_Value              = numeric(),
    P_Value_adjusted     = numeric(),
    Significant          = logical(),
    Significant_Adjusted = logical(),
    Conclusion           = character(),
    Adj_Conclusion       = character(),
    AIC_Full_Model       = numeric(),
    AIC_Reduced_Model    = numeric(),
    Tukey_Test           = character(),
    stringsAsFactors     = FALSE
  )
  
  tryCatch({
    
    # full model: bin x condition x cluster
    formula_full      <- as.formula(paste(feature_name, "~ bin * condition * cluster_membership_2 + (1 | id)"))
    model1 <- lmer(formula_full, data = data_complete)
    
    # cluster effect
    formula_no_cluster <- as.formula(paste(feature_name, "~ bin * condition + (1 | id)"))
    model2 <- lmer(formula_no_cluster, data = data_complete)
    
    anova_result1 <- anova(model2, model1)
    p_value1      <- anova_result1$`Pr(>Chisq)`[2]
    significant1  <- !is.na(p_value1) && p_value1 < alpha
    eta_squared1  <- calculate_eta_squared(anova_result1)
    tukey1 <- if(significant1) perform_tukey_test(model1, data_complete, feature_name, "Cluster_Effect") else "not significant - no post-hoc"
    
    results <- rbind(results, data.frame(
      Feature = feature_name, Comparison = "Cluster_Effect",
      Full_Model = "bin * condition * cluster_membership_2", Reduced_Model = "bin * condition",
      Chi_Square = ifelse(is.na(anova_result1$Chisq[2]), NA, anova_result1$Chisq[2]),
      Df = ifelse(is.na(anova_result1$Df[2]), NA, anova_result1$Df[2]),
      Effect_size_eta_squared = eta_squared1, P_Value = p_value1, P_Value_adjusted = NA,
      Significant = significant1, Significant_Adjusted = NA,
      Conclusion = ifelse(significant1, "cluster significant", "cluster not significant"),
      Adj_Conclusion = NA, AIC_Full_Model = AIC(model1), AIC_Reduced_Model = AIC(model2),
      Tukey_Test = tukey1, stringsAsFactors = FALSE
    ))
    
    # condition effect
    formula_no_condition <- as.formula(paste(feature_name, "~ bin * cluster_membership_2 + (1 | id)"))
    model3 <- lmer(formula_no_condition, data = data_complete)
    
    anova_result2 <- anova(model3, model1)
    p_value2      <- anova_result2$`Pr(>Chisq)`[2]
    significant2  <- !is.na(p_value2) && p_value2 < alpha
    eta_squared2  <- calculate_eta_squared(anova_result2)
    tukey2 <- if(significant2) perform_tukey_test(model1, data_complete, feature_name, "Condition_Effect") else "not significant - no post-hoc"
    
    results <- rbind(results, data.frame(
      Feature = feature_name, Comparison = "Condition_Effect",
      Full_Model = "bin * condition * cluster_membership_2", Reduced_Model = "bin * cluster_membership_2",
      Chi_Square = ifelse(is.na(anova_result2$Chisq[2]), NA, anova_result2$Chisq[2]),
      Df = ifelse(is.na(anova_result2$Df[2]), NA, anova_result2$Df[2]),
      Effect_size_eta_squared = eta_squared2, P_Value = p_value2, P_Value_adjusted = NA,
      Significant = significant2, Significant_Adjusted = NA,
      Conclusion = ifelse(significant2, "condition significant", "condition not significant"),
      Adj_Conclusion = NA, AIC_Full_Model = AIC(model1), AIC_Reduced_Model = AIC(model3),
      Tukey_Test = tukey2, stringsAsFactors = FALSE
    ))
    
    # bin effect
    formula_no_bin <- as.formula(paste(feature_name, "~ condition * cluster_membership_2 + (1 | id)"))
    model4 <- lmer(formula_no_bin, data = data_complete)
    
    anova_result3 <- anova(model4, model1)
    p_value3      <- anova_result3$`Pr(>Chisq)`[2]
    significant3  <- !is.na(p_value3) && p_value3 < alpha
    eta_squared3  <- calculate_eta_squared(anova_result3)
    
    results <- rbind(results, data.frame(
      Feature = feature_name, Comparison = "Bin_Effect",
      Full_Model = "bin * condition * cluster_membership_2", Reduced_Model = "condition * cluster_membership_2",
      Chi_Square = ifelse(is.na(anova_result3$Chisq[2]), NA, anova_result3$Chisq[2]),
      Df = ifelse(is.na(anova_result3$Df[2]), NA, anova_result3$Df[2]),
      Effect_size_eta_squared = eta_squared3, P_Value = p_value3, P_Value_adjusted = NA,
      Significant = significant3, Significant_Adjusted = NA,
      Conclusion = ifelse(significant3, "bin significant", "bin not significant"),
      Adj_Conclusion = NA, AIC_Full_Model = AIC(model1), AIC_Reduced_Model = AIC(model4),
      Tukey_Test = "not applicable for bin effect", stringsAsFactors = FALSE
    ))
    
    significant_effects <- sum(results$Significant, na.rm = TRUE)
    # significant effects
    if (significant_effects > 0)
      print(results[results$Significant == TRUE, c("Comparison", "P_Value", "Effect_size_eta_squared", "Conclusion")])
    
  }, error = function(e) {
    # error fitting models
    return(NULL)
  })
  
  # finished
  return(results)
}

results_stats <- list()

# output directory
output_dir <- file.path(getwd(), "VPA_stats_results")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# initialise results dataframe
all_results <- data.frame(
  Feature = character(), Comparison = character(),
  Full_Model = character(), Reduced_Model = character(),
  Chi_Square = numeric(), Df = numeric(),
  Effect_size_eta_squared = numeric(), P_Value = numeric(),
  P_Value_adjusted = numeric(), Significant = logical(),
  Significant_Adjusted = logical(), Conclusion = character(),
  Adj_Conclusion = character(), AIC_Full_Model = numeric(),
  AIC_Reduced_Model = numeric(), Tukey_Test = character(),
  stringsAsFactors = FALSE
)

for (i in 1:length(features)) {
  feature <- features[i]
  cat("\n[", i, "/", length(features), "]", feature, "\n")
  
  result_stat <- analyse_models_for_feature(t, feature, alpha = 0.05)
  results_stats[[feature]] <- result_stat
  
  if (!is.null(result_stat)) {
    all_results <- rbind(all_results, result_stat)
    feature_filename <- file.path(output_dir, paste0("analysis_", feature, ".csv"))
    write.csv(result_stat, feature_filename, row.names = FALSE)
    # saved
  }
}

# fdr correction
all_results$P_Value_adjusted    <- p.adjust(all_results$P_Value, method = "fdr")
all_results$Significant_Adjusted <- all_results$P_Value_adjusted < 0.05
all_results$Adj_Conclusion <- ifelse(
  all_results$Significant_Adjusted,
  paste(all_results$Conclusion, "(significant after FDR)"),
  paste(all_results$Conclusion, "(not significant after FDR)")
)

original_sig <- sum(all_results$Significant, na.rm = TRUE)
adjusted_sig <- sum(all_results$Significant_Adjusted, na.rm = TRUE)
# originally significant: original_sig
# significant after FDR: adjusted_sig
# lost significance: original_sig - adjusted_sig

csv_filename <- file.path(output_dir, "model_analysis_results_all_enhanced.csv")
write.csv(all_results, csv_filename, row.names = FALSE)
# saved

significant_results <- all_results[all_results$Significant_Adjusted == TRUE, ]
if (nrow(significant_results) > 0) {
  csv_filename_sig <- file.path(output_dir, "model_analysis_results_significant_FDR_corrected.csv")
  write.csv(significant_results, csv_filename_sig, row.names = FALSE)
  # saved
  for (effect in unique(significant_results$Comparison)) {
    effect_results <- significant_results[significant_results$Comparison == effect, ]
    cat(effect, "- significant for", nrow(effect_results), "features:\n")
    print(effect_results[, c("Feature", "P_Value", "P_Value_adjusted", "Effect_size_eta_squared")])
  }
} else {
  # no results significant after FDR correction
}

# effect size categories 
# small: 0.01-0.06
# medium: 0.06-0.14
# large: >= 0.14

all_results$Effect_Size_Category <- ifelse(
  is.na(all_results$Effect_size_eta_squared), "not calculated",
  ifelse(all_results$Effect_size_eta_squared < 0.01, "negligible (< 0.01)",
         ifelse(all_results$Effect_size_eta_squared < 0.06, "small (0.01-0.06)",
                ifelse(all_results$Effect_size_eta_squared < 0.14, "medium (0.06-0.14)",
                       "large (>= 0.14)"))))

csv_filename_final <- file.path(output_dir, "model_analysis_results_FINAL_with_effect_sizes.csv")
write.csv(all_results, csv_filename_final, row.names = FALSE)
# saved