# Lmm feature analysis - model selection and summary tables
# Antonella Torrisi

rm(list = ls())
library(extrafont)
loadfonts(device = "win")

library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(readr)
library(broom.mixed)
library(car)
library(knitr)
library(tibble)

# load data
setwd("/path/to/project")
#setwd("C:\Users\anton\VPA_vocalisations_project\Results_Clustering")
data_path <- file.path("data", "full_dataset_vpa_ctrl.csv")
t <- read_csv(data_path)

t <- t[complete.cases(t), ]

t <- t %>%
  mutate(
    id                   = as.factor(id),
    bin                  = as.factor(bin),
    condition            = as.factor(condition),
    cluster_membership_2 = as.factor(cluster_membership_2),
    cluster_membership_3 = as.factor(cluster_membership_3)
  )

if ("1" %in% levels(t$bin)) t$bin <- relevel(t$bin, ref = "1")

# features
features <- c(
  "Duration_call",
  "Attack_time",
  "Slope",
  "RMS_Mean",
  "RMS_Std",
  "F0_Mean",
  "F0_Std",
  "F0_Skewness",
  "F0_Kurtosis",
  "F0_Bandwidth",
  "F0_1st_Order_Diff",
  "F0_Slope",
  "F0_Mag_Mean",
  "F1_Mag_Mean",
  "F2_Mag_Mean",
  "F1_F0_Ratio",
  "F2_F0_Ratio",
  "Spectral_Centroid_Mean",
  "Spectral_Centroid_Std",
  "Attack_magnitude"
)

# approximate partial eta-squared from anova output
compute_partial_eta_squared <- function(anova_result) {
  chisq <- anova_result$Chisq
  df    <- anova_result$Df
  eta2  <- chisq / (chisq + sum(df, na.rm = TRUE))
  return(eta2)
}

# fit candidate models, select minimum adequate model via lrt, run posthoc
analyze_feature <- function(feature_name, data) {
  
  cat("\n", feature_name, "\n")
  
  vars_needed   <- c(feature_name, "bin", "condition", "cluster_membership_2", "id")
  data_complete <- data[complete.cases(data[, vars_needed]), ]
  
  if (nrow(data_complete) == 0) {
    warning(paste("no complete cases for", feature_name))
    return(NULL)
  }
  
  # candidate models (ml for comparison)
  formula_1 <- paste(feature_name, "~ bin * condition * cluster_membership_2 + (1 | id)")
  formula_2 <- paste(feature_name, "~ bin * condition + bin * cluster_membership_2 + condition * cluster_membership_2 + (1 | id)")
  formula_3 <- paste(feature_name, "~ bin + condition * cluster_membership_2 + (1 | id)")
  formula_4 <- paste(feature_name, "~ bin + condition + cluster_membership_2 + bin:cluster_membership_2 + condition:cluster_membership_2 + (1 | id)")
  formula_5 <- paste(feature_name, "~ bin + condition + cluster_membership_2 + (1 | id)")
  
  model_1 <- lmer(as.formula(formula_1), data = data_complete, REML = FALSE)
  model_2 <- lmer(as.formula(formula_2), data = data_complete, REML = FALSE)
  model_3 <- lmer(as.formula(formula_3), data = data_complete, REML = FALSE)
  model_4 <- lmer(as.formula(formula_4), data = data_complete, REML = FALSE)
  model_5 <- lmer(as.formula(formula_5), data = data_complete, REML = FALSE)
  
  # likelihood ratio tests
  comp_1_2 <- anova(model_1, model_2)
  comp_2_5 <- anova(model_2, model_5)
  comp_3_5 <- anova(model_3, model_5)
  
  cat("3-way interaction:\n");    print(comp_1_2)
  cat("2-way interactions:\n");   print(comp_2_5)
  cat("condition x cluster:\n");  print(comp_3_5)
  
  # model selection
  if (comp_1_2$`Pr(>Chisq)`[2] < 0.05) {
    best_model      <- model_1
    best_model_name <- "Model 1 (Full: Bin × Condition × Cluster)"
    best_formula    <- formula_1
    decision_rationale <- "Three-way interaction significant (p < 0.05)"
  } else if (comp_2_5$`Pr(>Chisq)`[2] < 0.05) {
    best_model      <- model_2
    best_model_name <- "Model 2 (Two-way interactions)"
    best_formula    <- formula_2
    decision_rationale <- "Three-way not significant, but two-way interactions significant"
  } else {
    best_model      <- model_5
    best_model_name <- "Model 5 (Main effects only)"
    best_formula    <- formula_5
    decision_rationale <- "Only main effects significant"
  }
  
  cat("selected:", best_model_name, "\n")
  
  # refit with reml for inference
  best_model_reml <- lmer(as.formula(best_formula), data = data_complete, REML = TRUE)
  
  anova_type3 <- Anova(best_model_reml, type = "III")
  print(anova_type3)
  
  # partial eta-squared
  chisq_vals  <- anova_type3$Chisq
  df_vals     <- anova_type3$Df
  eta2_approx <- chisq_vals / (chisq_vals + sum(df_vals, na.rm = TRUE))
  
  anova_with_eta2 <- anova_type3 %>%
    as.data.frame() %>%
    mutate(partial_eta2 = eta2_approx)
  
  print(anova_with_eta2)
  
  # posthoc if condition x cluster is in the model
  posthoc_results <- NULL
  
  if (grepl("condition.*cluster_membership_2|cluster_membership_2.*condition", best_formula)) {
    tryCatch({
      emm_inter <- emmeans(best_model_reml, ~ condition | cluster_membership_2)
      print(summary(emm_inter))
      posthoc         <- pairs(emm_inter, adjust = "tukey")
      posthoc_results <- summary(posthoc)
      print(posthoc_results)
    }, error = function(e) cat("posthoc failed:", e$message, "\n"))
  }
  
  print(summary(best_model_reml))
  
  list(
    feature            = feature_name,
    n_complete         = nrow(data_complete),
    n_subjects         = length(unique(data_complete$id)),
    model_1 = model_1, model_2 = model_2, model_3 = model_3,
    model_4 = model_4, model_5 = model_5,
    comparisons        = list(comp_1_2 = comp_1_2, comp_2_5 = comp_2_5, comp_3_5 = comp_3_5),
    best_model         = best_model_reml,
    best_model_name    = best_model_name,
    best_formula       = best_formula,
    decision_rationale = decision_rationale,
    anova_type3        = anova_with_eta2,
    posthoc            = posthoc_results,
    AIC                = AIC(best_model_reml),
    BIC                = BIC(best_model_reml)
  )
}

# run for all features
all_results <- list()

for (feature in features) {
  result <- tryCatch({
    analyze_feature(feature, t)
  }, error = function(e) {
    cat("error:", feature, ":", e$message, "\n")
    NULL
  })
  if (!is.null(result)) all_results[[feature]] <- result
}

# table 1: model selection summary
summary_df <- data.frame(
  Feature            = character(),
  N_Complete         = integer(),
  N_Subjects         = integer(),
  Best_Model         = character(),
  Decision_Rationale = character(),
  AIC                = numeric(),
  BIC                = numeric(),
  stringsAsFactors   = FALSE
)

for (feature in names(all_results)) {
  res        <- all_results[[feature]]
  summary_df <- rbind(summary_df, data.frame(
    Feature            = feature,
    N_Complete         = res$n_complete,
    N_Subjects         = res$n_subjects,
    Best_Model         = res$best_model_name,
    Decision_Rationale = res$decision_rationale,
    AIC                = round(res$AIC, 2),
    BIC                = round(res$BIC, 2)
  ))
}

print(summary_df)
write.csv(summary_df, "Table_S1_Model_Selection_Summary.csv", row.names = FALSE)

# table 2: model comparisons for all features
key_features <- c("Duration_call", "RMS_Mean", "Spectral_Centroid_Mean", "Attack_time",
                  "Slope", "RMS_Std", "F0_Mean", "F0_Std", "F0_Skewness", "F0_Kurtosis",
                  "F0_Bandwidth", "F0_1st_Order_Diff", "F0_Slope", "F0_Mag_Mean", "F1_Mag_Mean",
                  "F2_Mag_Mean", "F1_F0_Ratio", "F2_F0_Ratio", "Spectral_Centroid_Std", "Attack_magnitude")

detailed_comp_df <- data.frame(
  Feature    = character(),
  Comparison = character(),
  Chi_square = numeric(),
  df         = integer(),
  p_value    = numeric(),
  stringsAsFactors = FALSE
)

for (feature in key_features) {
  if (feature %in% names(all_results)) {
    res  <- all_results[[feature]]
    
    comp <- res$comparisons$comp_1_2
    detailed_comp_df <- rbind(detailed_comp_df, data.frame(
      Feature    = feature,
      Comparison = "Model 1 vs Model 2 (3-way interaction)",
      Chi_square = round(comp$Chisq[2], 2),
      df         = comp$Df[2],
      p_value    = round(comp$`Pr(>Chisq)`[2], 4)
    ))
    
    comp <- res$comparisons$comp_2_5
    detailed_comp_df <- rbind(detailed_comp_df, data.frame(
      Feature    = feature,
      Comparison = "Model 2 vs Model 5 (2-way interactions)",
      Chi_square = round(comp$Chisq[2], 2),
      df         = comp$Df[2],
      p_value    = round(comp$`Pr(>Chisq)`[2], 4)
    ))
  }
}

print(detailed_comp_df)
write.csv(detailed_comp_df, "Table_S2_Detailed_Model_Comparisons.csv", row.names = FALSE)

# table 3: type iii anova with effect sizes
anova_summary_df <- data.frame(
  Feature      = character(),
  Effect       = character(),
  Chi_square   = numeric(),
  df           = integer(),
  p_value      = numeric(),
  partial_eta2 = numeric(),
  stringsAsFactors = FALSE
)

for (feature in names(all_results)) {
  res      <- all_results[[feature]]
  anova_res <- res$anova_type3
  effects  <- rownames(anova_res)
  
  for (i in seq_along(effects)) {
    anova_summary_df <- rbind(anova_summary_df, data.frame(
      Feature      = feature,
      Effect       = effects[i],
      Chi_square   = round(anova_res$Chisq[i], 2),
      df           = anova_res$Df[i],
      p_value      = round(anova_res$`Pr(>Chisq)`[i], 4),
      partial_eta2 = round(anova_res$partial_eta2[i], 3)
    ))
  }
}

print(head(anova_summary_df, 20))
write.csv(anova_summary_df, "Table_S3_ANOVA_Results_Complete.csv", row.names = FALSE)

# table 4: posthoc comparisons
posthoc_summary_df <- data.frame(
  Feature  = character(),
  Cluster  = character(),
  Contrast = character(),
  Estimate = numeric(),
  SE       = numeric(),
  t_ratio  = numeric(),
  p_value  = numeric(),
  stringsAsFactors = FALSE
)

for (feature in names(all_results)) {
  res <- all_results[[feature]]
  
  if (!is.null(res$posthoc)) {
    ph <- as.data.frame(res$posthoc)
    
    for (i in 1:nrow(ph)) {
      t_value <- if ("t.ratio" %in% names(ph)) {
        as.numeric(ph$t.ratio[i])
      } else if ("t" %in% names(ph)) {
        as.numeric(ph$t[i])
      } else if ("statistic" %in% names(ph)) {
        as.numeric(ph$statistic[i])
      } else {
        NA
      }
      
      posthoc_summary_df <- rbind(posthoc_summary_df, data.frame(
        Feature  = feature,
        Cluster  = as.character(ph$cluster_membership_2[i]),
        Contrast = as.character(ph$contrast[i]),
        Estimate = round(as.numeric(ph$estimate[i]), 4),
        SE       = round(as.numeric(ph$SE[i]), 4),
        t_ratio  = if (!is.na(t_value)) round(t_value, 3) else NA,
        p_value  = round(as.numeric(ph$p.value[i]), 4),
        stringsAsFactors = FALSE
      ))
    }
  }
}

print(head(posthoc_summary_df, 20))
write.csv(posthoc_summary_df, "Table_S4_Posthoc_Comparisons.csv", row.names = FALSE)

# latex table for supplementary (table s1)
latex_lines <- c(
  "% Table S1: Model Selection Summary",
  "\\begin{table}[h]",
  "\\centering",
  "\\scriptsize",
  "\\begin{tabular}{lllc}",
  "\\toprule",
  "\\textbf{Feature} & \\textbf{Final Model} & \\textbf{N} & \\textbf{AIC} \\\\",
  "\\midrule"
)

for (i in 1:nrow(summary_df)) {
  latex_lines <- c(latex_lines,
                   sprintf("%s & %s & %d & %.0f \\\\",
                           gsub("_", "\\\\_", summary_df$Feature[i]),
                           gsub("×", "$\\\\times$", summary_df$Best_Model[i]),
                           summary_df$N_Complete[i],
                           summary_df$AIC[i]))
}

latex_lines <- c(latex_lines,
                 "\\bottomrule",
                 "\\end{tabular}",
                 "\\caption{Model selection summary for all acoustic features.}",
                 "\\label{tab:model_selection_summary}",
                 "\\end{table}")

writeLines(latex_lines, "Table_S1_Model_Selection.tex")

save(all_results, file = "Complete_Feature_Analysis_Results.RData")