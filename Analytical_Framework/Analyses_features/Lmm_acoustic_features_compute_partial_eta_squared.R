# Lmm feature analysis - model selection, likelihood ratio tests, partial eta-squared, fdr
# Antonella Torrisi

rm(list = ls())

library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(effectsize)
library(readr)

options(contrasts = c("contr.sum", "contr.poly"))

# load data
setwd("/path/to/project")
data_path <- file.path("data", "full_dataset_vpa_ctrl.csv")
t <- read_csv(data_path)

t <- t[complete.cases(t), ]

t <- t %>%
  mutate(
    id                   = as.factor(id),
    bin                  = as.factor(bin),
    condition            = as.factor(condition),
    cluster_membership_2 = as.factor(cluster_membership_2)
  )

if ("1" %in% levels(t$bin)) t$bin <- relevel(t$bin, ref = "1")

# features
features <- c(
  "Duration_call", "Attack_time", "Slope",
  "RMS_Mean", "RMS_Std", "Attack_magnitude",
  "F0_Mean", "F0_Std", "F0_Skewness", "F0_Kurtosis", "F0_Bandwidth",
  "F0_1st_Order_Diff", "F0_Slope", "F0_Mag_Mean", "F1_Mag_Mean", "F2_Mag_Mean",
  "F1_F0_Ratio", "F2_F0_Ratio", "Spectral_Centroid_Mean", "Spectral_Centroid_Std"
)

alpha <- 0.05

main_effects  <- "bin + condition + cluster_membership_2"
twoway_terms  <- c("bin:condition", "bin:cluster_membership_2", "condition:cluster_membership_2")
threeway_term <- "bin:condition:cluster_membership_2"

build_formula <- function(feature_name, terms) {
  as.formula(paste(feature_name, "~", paste(c(main_effects, terms), collapse = " + "), "+ (1 | id)"))
}

model_label <- function(terms) {
  lab <- c("Bin + Condition + Cluster", terms)
  lab <- gsub("bin", "Bin", lab)
  lab <- gsub("condition", "Condition", lab)
  lab <- gsub("cluster_membership_2", "Cluster", lab)
  lab <- gsub(":", " x ", lab)
  paste(lab, collapse = " + ")
}

# model selection: three-way interaction, then two-way interactions (least significant first)
# chi-square, p and aic of the models with and without each term are saved at every step
step_label <- function(term) {
  gsub(":", " x ", gsub("cluster_membership_2", "Cluster", gsub("condition", "Condition", gsub("bin", "Bin", term))))
}

select_model <- function(feature_name, data) {

  steps <- data.frame()

  # three-way interaction
  m_3way <- lmer(build_formula(feature_name, c(twoway_terms, threeway_term)), data = data, REML = FALSE)
  m_2way <- lmer(build_formula(feature_name, twoway_terms), data = data, REML = FALSE)
  lrt    <- anova(m_2way, m_3way)
  p3     <- lrt$`Pr(>Chisq)`[2]

  steps <- rbind(steps, data.frame(
    Feature = feature_name, Step = 1, Term_tested = step_label(threeway_term),
    Chi_square = lrt$Chisq[2], df = lrt$Df[2], p_value = p3,
    AIC_with_term = AIC(m_3way), AIC_without_term = AIC(m_2way),
    delta_AIC = AIC(m_2way) - AIC(m_3way),
    Decision = ifelse(p3 < alpha, "retained", "dropped")))

  if (p3 < alpha) return(list(terms = c(twoway_terms, threeway_term), steps = steps))

  # two-way interactions
  current <- twoway_terms
  step    <- 1
  repeat {
    if (length(current) == 0) break
    m_cur <- lmer(build_formula(feature_name, current), data = data, REML = FALSE)

    tests <- do.call(rbind, lapply(current, function(tt) {
      m_red <- lmer(build_formula(feature_name, setdiff(current, tt)), data = data, REML = FALSE)
      a     <- anova(m_red, m_cur)
      data.frame(term = tt, chisq = a$Chisq[2], df = a$Df[2], p = a$`Pr(>Chisq)`[2],
                 aic_with = AIC(m_cur), aic_without = AIC(m_red))
    }))

    step  <- step + 1
    worst <- tests[which.max(tests$p), ]

    if (worst$p >= alpha) {
      steps <- rbind(steps, data.frame(
        Feature = feature_name, Step = step, Term_tested = step_label(worst$term),
        Chi_square = worst$chisq, df = worst$df, p_value = worst$p,
        AIC_with_term = worst$aic_with, AIC_without_term = worst$aic_without,
        delta_AIC = worst$aic_without - worst$aic_with, Decision = "dropped"))
      current <- setdiff(current, worst$term)
    } else {
      # all remaining two-way interactions significant
      steps <- rbind(steps, data.frame(
        Feature = feature_name, Step = step, Term_tested = step_label(tests$term),
        Chi_square = tests$chisq, df = tests$df, p_value = tests$p,
        AIC_with_term = tests$aic_with, AIC_without_term = tests$aic_without,
        delta_AIC = tests$aic_without - tests$aic_with, Decision = "retained"))
      break
    }
  }

  list(terms = current, steps = steps)
}

# likelihood ratio tests on the full model: each effect removed with its higher-order interactions
# bin x condition and bin x cluster are not reported in the paper but are included in the fdr correction
full_rhs <- "bin * condition * cluster_membership_2"
effects <- list(
  Three_Way_Interaction           = "bin * condition + bin * cluster_membership_2 + condition * cluster_membership_2",
  Bin_x_Condition_Interaction     = "bin + condition + bin * cluster_membership_2 + condition * cluster_membership_2",
  Bin_x_Cluster_Interaction       = "bin * condition + bin + cluster_membership_2 + condition * cluster_membership_2",
  Condition_x_Cluster_Interaction = "bin * condition + bin * cluster_membership_2 + condition + cluster_membership_2",
  Bin_Effect                      = "condition * cluster_membership_2",
  Condition_Effect                = "bin * cluster_membership_2",
  Cluster_Effect                  = "bin * condition"
)

# type iii term corresponding to each effect
typeIII_term <- c(
  Three_Way_Interaction           = "bin:condition:cluster_membership_2",
  Bin_x_Condition_Interaction     = "bin:condition",
  Bin_x_Cluster_Interaction       = "bin:cluster_membership_2",
  Condition_x_Cluster_Interaction = "condition:cluster_membership_2",
  Bin_Effect                      = "bin",
  Condition_Effect                = "condition",
  Cluster_Effect                  = "cluster_membership_2"
)

test_effects <- function(feature_name, data) {
  m_full <- lmer(as.formula(paste(feature_name, "~", full_rhs, "+ (1 | id)")), data = data, REML = TRUE)

  # partial eta-squared from the type iii anova table
  es <- as.data.frame(eta_squared(anova(m_full, type = 3), partial = TRUE, alternative = "two.sided"))

  do.call(rbind, lapply(names(effects), function(eff) {
    m_red <- lmer(as.formula(paste(feature_name, "~", effects[[eff]], "+ (1 | id)")), data = data, REML = TRUE)
    lrt   <- anova(m_red, m_full)
    data.frame(Feature = feature_name, Comparison = eff,
               Chi_Square = lrt$Chisq[2], Df = lrt$Df[2], P_Value = lrt$`Pr(>Chisq)`[2],
               Partial_eta2 = es$Eta2_partial[es$Parameter == typeIII_term[[eff]]])
  }))
}

# posthoc on the selected model
posthoc_selected <- function(feature_name, data, terms) {
  if (!"condition:cluster_membership_2" %in% terms) return(NULL)
  m_sel <- lmer(build_formula(feature_name, terms), data = data, REML = TRUE)
  ph    <- as.data.frame(summary(pairs(emmeans(m_sel, ~ condition | cluster_membership_2), adjust = "tukey")))
  data.frame(Feature = feature_name, Cluster = as.character(ph$cluster_membership_2),
             Contrast = as.character(ph$contrast), Estimate = ph$estimate, SE = ph$SE,
             z_or_t = if ("z.ratio" %in% names(ph)) ph$z.ratio else ph$t.ratio,
             p_value = ph$p.value)
}

# run for all features
selected_models <- data.frame()
selection_steps <- data.frame()
effect_tests    <- data.frame()
posthoc_all     <- data.frame()

for (feature in features) {
  cat("\n", feature, "\n")
  d <- t[complete.cases(t[, c(feature, "bin", "condition", "cluster_membership_2", "id")]), ]

  sel       <- select_model(feature, d)
  terms_sel <- sel$terms
  selection_steps <- rbind(selection_steps, sel$steps)
  m_sel_ml  <- lmer(build_formula(feature, terms_sel), data = d, REML = FALSE)
  cat("selected:", model_label(terms_sel), "\n")

  selected_models <- rbind(selected_models, data.frame(
    Feature     = feature,
    N_Complete  = nrow(d),
    N_Subjects  = length(unique(d$id)),
    Final_Model = model_label(terms_sel),
    AIC         = round(AIC(m_sel_ml), 2),
    BIC         = round(BIC(m_sel_ml), 2)
  ))

  effect_tests <- rbind(effect_tests, test_effects(feature, d))

  ph <- posthoc_selected(feature, d, terms_sel)
  if (!is.null(ph)) posthoc_all <- rbind(posthoc_all, ph)
}

# fdr correction across all likelihood ratio tests
effect_tests$P_Value_adjusted <- p.adjust(effect_tests$P_Value, method = "fdr")

print(selected_models)

write.csv(selected_models, "Features_selected_models.csv",                  row.names = FALSE)
write.csv(selection_steps, "Features_model_selection_steps.csv",            row.names = FALSE)
write.csv(effect_tests,    "Features_results_LRT_partial_eta2_FDR.csv",     row.names = FALSE)
write.csv(posthoc_all,     "Features_posthoc_condition_within_cluster.csv", row.names = FALSE)

# package versions
writeLines(capture.output(sessionInfo()), "sessionInfo_acoustic_features.txt")
