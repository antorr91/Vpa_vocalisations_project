# Carica le librerie necessarie
library(dplyr)
library(lme4)       # Per modelli misti
library(lmerTest)   # Per ANOVA con effetti misti
library(xtable)     # Per convertire in LaTeX
library(multcomp)   # Per i test post-hoc di Tukey
library(effectsize) # Per calcolare eta squared
rm(list=ls()) 

# Imposta la directory di lavoro (se necessario)
setwd("C:/Users/anton/VPA_vocalisations_project/Results_Analysis/Statistical_plots_between")

# Importa i dati
ctrl_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_ctrl_3/hierarchical_clustering_3_distance_membership.csv")
ctrl_membership$condition <- "CTRL"

vpa_membership <- read_csv("C:/Users/anton/VPA_vocalisations_project/Results_Clustering/hierarchical_clustering_vpa_3/hierarchical_clustering_3_distance_membership.csv")
vpa_membership$condition <- "VPA"

combined_membership <- bind_rows(ctrl_membership, vpa_membership)

combined_membership$id <- combined_membership$recording

table(combined_membership$cluster_membership)

# Converti 'condition' e 'cluster_membership' in fattori
combined_membership$condition <- as.factor(combined_membership$condition)
combined_membership$cluster_membership <- as.factor(combined_membership$cluster_membership)

names(combined_membership)

# Lista delle feature da analizzare
features <- c("Duration_call", "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis", 
              "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope", "F0 Mag Mean", 
              "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", "F2-F0 Ratio", 
              "Spectral Centroid Mean", "Spectral Centroid Std", "RMS Mean", "RMS Std",
              "Slope", "Attack_magnitude", "Attack_time")

# Crea cartelle per salvare i risultati
output_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_Clustering_analysis/Statistical_plots_bt3_condition_anova/signif"
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# Crea dataframe per salvare i risultati
anova_results_df <- data.frame()
tukey_results_df <- data.frame()
glm_results_df <- data.frame()

# Ciclo FOR per analizzare ogni feature
for (feature in features) {
  if (!(feature %in% colnames(combined_membership))) {
    print(paste("Feature", feature, "non trovata nel dataset. Saltando..."))
    next
  }
  
  print(paste("Analisi in corso per la feature:", feature))
  
  # Modello misto
  formula_mixed <- as.formula(paste("`", feature, "` ~ condition * cluster_membership + (1 | id)", sep = ""))
  mixed_model <- lmer(formula_mixed, data = combined_membership)
  
  # ANOVA sul modello misto
  anova_results <- anova(mixed_model)
  
  # Calcola eta squared
  eta_sq_results <- eta_squared(mixed_model, partial = TRUE)
  
  # Estrarre i risultati e salvarli
  anova_results <- data.frame(
    Feature = feature,
    Term = rownames(anova_results),
    F_value = anova_results$`F value`,
    p_value = anova_results$`Pr(>F)`,
    eta_squared = eta_sq_results$Eta2_partial
  )
  
  anova_results_df <- rbind(anova_results_df, anova_results)
  
  # Test di Tukey (post-hoc)
  tukey_test <- glht(mixed_model, linfct = mcp(cluster_membership = "Tukey"))
  tukey_summary <- summary(tukey_test)
  
  tukey_results <- data.frame(
    Feature = feature,
    Comparison = names(tukey_summary$test$coefficients),
    Estimate = tukey_summary$test$coefficients,
    Std_Error = tukey_summary$test$sigma,
    t_value = tukey_summary$test$tstat,
    p_value = tukey_summary$test$pvalues
  )
  
  tukey_results_df <- rbind(tukey_results_df, tukey_results)
  
  # Modello lineare generalizzato (GLM)
  formula_glm <- as.formula(paste("`", feature, "` ~ condition * cluster_membership", sep = ""))
  glm_model <- glm(formula_glm, data = combined_membership, family = gaussian())
  glm_summary <- summary(glm_model)
  
  glm_results <- data.frame(
    Feature = feature,
    Term = rownames(glm_summary$coefficients),
    Estimate = glm_summary$coefficients[, "Estimate"],
    Std_Error = glm_summary$coefficients[, "Std. Error"],
    t_value = glm_summary$coefficients[, "t value"],
    p_value = glm_summary$coefficients[, "Pr(>|t|)"]
  )
  
  glm_results_df <- rbind(glm_results_df, glm_results)
}

# Salva i risultati in file CSV
write.csv(anova_results_df, file.path(output_folder, "anova_results.csv"), row.names = FALSE)
write.csv(glm_results_df, file.path(output_folder, "glm_results.csv"), row.names = FALSE)
write.csv(tukey_results_df, file.path(output_folder, "tukey_results.csv"), row.names = FALSE)

# Converti i risultati in LaTeX
print(xtable(anova_results_df), include.rownames = FALSE, file = file.path(output_folder, "anova_results.tex"))
print(xtable(glm_results_df), include.rownames = FALSE, file = file.path(output_folder, "glm_results.tex"))
print(xtable(tukey_results_df), include.rownames = FALSE, file = file.path(output_folder, "tukey_results.tex"))

print("Analisi completata e risultati salvati!")

