setwd("C:\\Users\\anton\\VPA_vocalisations_project\\Results_features_wise_analysis_new")

# import 
library(readxl)
library(ggplot2)
library(plyr)
library(lme4)
library(xtable)  # Per esportare i risultati in LaTeX

rm(list=ls()) 

# Caricamento dati
t = read.csv("combined_data.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

t$id= t$recording
t$group= t$Group

# Cartelle per salvare i risultati
output_stats_folder <- "C:/Users/anton/VPA_vocalisations_project/Results_univariate_analysis"
if (!dir.exists(output_stats_folder)) dir.create(output_stats_folder, recursive = TRUE)

# Funzione per ottenere le stelline di significatività
get_stars <- function(p_value) {
  if (p_value < 0.001) return("***")
  else if (p_value < 0.01) return("**")
  else if (p_value < 0.05) return("*")
  else return("")
}

# Lista delle features
features <- c("Duration_call", "F0.Mean", "F0.Std", "F0.Skewness", "F0.Kurtosis", 
              "F0.Bandwidth", "F0.1st.Order.Diff", "F0.Slope", "F0.Mag.Mean", 
              "F1.Mag.Mean", "F2.Mag.Mean", "F1.F0.Ratio", "F2.F0.Ratio", 
              "Spectral.Centroid.Mean", "Spectral.Centroid.Std", "RMS.Mean", "RMS.Std",
              "Slope", "Attack_magnitude", "Attack_time")

# Dataframes separati per i risultati
t_test_results <- data.frame(Feature = character(), Mean_Ctrl = numeric(), SD_Ctrl = numeric(),
                             Mean_VPA = numeric(), SD_VPA = numeric(),
                             t_value = numeric(), df = numeric(), p_value = numeric(),
                             CI_lower = numeric(), CI_upper = numeric(), Significance = character(),
                             stringsAsFactors = FALSE)

wilcoxon_results <- data.frame(Feature = character(), Median_Ctrl = numeric(), Median_VPA = numeric(),
                               W_value = numeric(), p_value = numeric(), Significance = character(),
                               stringsAsFactors = FALSE)

lmm_results <- data.frame(Feature = character(), Formula = character(), Estimate = numeric(), SE = numeric(),
                          t_value = numeric(), p_value = numeric(), AIC = numeric(), Significance = character(),
                          stringsAsFactors = FALSE)

# Loop sulle features
for (feature in features) {
  print(paste("Processing:", feature))
  
  if (!(feature %in% colnames(t))) {
    print(paste("Feature", feature, "not found in dataset. Skipping..."))
    next
  }
  
  # Calcolo statistiche per ogni id e gruppo
  summary_data_reg <- ddply(t, .(id, group), summarize, 
                            mean_feature = mean(get(feature), na.rm = TRUE),
                            sd_feature = sd(get(feature), na.rm = TRUE),
                            median_feature = median(get(feature), na.rm = TRUE),
                            N = length(na.omit(get(feature))))
  
  summary_stats <- ddply(summary_data_reg, .(group), summarize,
                         Mean = mean(mean_feature, na.rm = TRUE),
                         SD = sd(mean_feature, na.rm = TRUE),
                         Median = median(mean_feature, na.rm = TRUE))
  
  # Test t
  t_test_result <- t.test(summary_data_reg$mean_feature ~ summary_data_reg$group)
  t_test_results <- rbind(t_test_results, data.frame(
    Feature = feature,
    Mean_Ctrl = summary_stats$Mean[1], SD_Ctrl = summary_stats$SD[1],
    Mean_VPA = summary_stats$Mean[2], SD_VPA = summary_stats$SD[2],
    t_value = t_test_result$statistic,
    df = t_test_result$parameter,
    p_value = t_test_result$p.value,
    CI_lower = t_test_result$conf.int[1],
    CI_upper = t_test_result$conf.int[2],
    Significance = get_stars(t_test_result$p.value)
  ))
  
  # Test di Wilcoxon
  wilcoxon_result <- wilcox.test(summary_data_reg$mean_feature ~ summary_data_reg$group, exact = FALSE)
  wilcoxon_results <- rbind(wilcoxon_results, data.frame(
    Feature = feature,
    Median_Ctrl = summary_stats$Median[1],
    Median_VPA = summary_stats$Median[2],
    W_value = wilcoxon_result$statistic,
    p_value = wilcoxon_result$p.value,
    Significance = get_stars(wilcoxon_result$p.value)
  ))
  
  # Modello misto
  mod <- lmer(as.formula(paste(feature, "~ group + (1 | id)")), data = t, REML = FALSE)
  mod_null <- lmer(as.formula(paste(feature, "~ (1 | id)")), data = t, REML = FALSE)
  
  model_summary <- summary(mod)
  model_estimate <- model_summary$coefficients[2, 1]
  model_se <- model_summary$coefficients[2, 2]
  model_t <- model_summary$coefficients[2, 3]
  
  anova_result <- anova(mod, mod_null)
  model_p <- anova_result$`Pr(>Chisq)`[2]  
  model_AIC <- AIC(mod)  
  
  lmm_results <- rbind(lmm_results, data.frame(
    Feature = feature,
    Formula = paste(feature, "~ group + (1 | id)"),
    Estimate = model_estimate,
    SE = model_se,
    t_value = model_t,
    p_value = model_p,
    AIC = model_AIC,
    Significance = get_stars(model_p)
  ))
}

# Esporta i risultati in CSV
write.csv(t_test_results, file.path(output_stats_folder, "t_test_results.csv"), row.names = FALSE)
write.csv(wilcoxon_results, file.path(output_stats_folder, "wilcoxon_results.csv"), row.names = FALSE)
write.csv(lmm_results, file.path(output_stats_folder, "lmm_results.csv"), row.names = FALSE)

# Esporta i risultati in LaTeX
print(xtable(t_test_results), type = "latex", file = file.path(output_stats_folder, "t_test_results.tex"))
print(xtable(wilcoxon_results), type = "latex", file = file.path(output_stats_folder, "wilcoxon_results.tex"))
print(xtable(lmm_results), type = "latex", file = file.path(output_stats_folder, "lmm_results.tex"))

print("Analisi completata! Risultati salvati in CSV e LaTeX")
