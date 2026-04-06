# F0 skewness - lmm model selection
# Antonella Torrisi

rm(list = ls())
library(extrafont)
loadfonts(device = "win")

library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(readr)

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

vars_needed <- c("F0_Skewness", "bin", "condition", "cluster_membership_2", "id")
t_complete <- t[complete.cases(t[, vars_needed]), ]
if (nrow(t_complete) == 0) stop("No complete cases for F0_Skewness.")

# model selection
model_1 <- lmer(F0_Skewness ~ bin * condition * cluster_membership_2 + (1 | id),
                data = t_complete)

model_2 <- lmer(F0_Skewness ~ bin * condition + cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_2) # significant, retain model_1

model_3 <- lmer(F0_Skewness ~ bin + condition * cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_3) # significant, retain model_1

model_4 <- lmer(F0_Skewness ~ bin:condition * cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_4) # significant, retain model_1

model_5 <- lmer(F0_Skewness ~ bin:condition:cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_5) # significant, retain model_1

model_full <- model_1

print(summary(model_full))
cat("AIC:", AIC(model_full))

# fixed effects (model_1):
#                                          Estimate Std. Error         df t value Pr(>|t|)
# (Intercept)                            -2.303e-01  5.645e-02  6.415e+01  -4.081 0.000127 ***
# bin2                                    3.702e-02  3.626e-02  2.350e+04   1.021 0.307275
# bin3                                   -2.814e-02  3.574e-02  2.351e+04  -0.788 0.430969
# bin4                                   -4.721e-02  3.464e-02  2.351e+04  -1.363 0.172917
# bin5                                   -8.541e-02  3.437e-02  2.352e+04  -2.485 0.012966 *
# bin6                                   -8.037e-02  3.389e-02  2.352e+04  -2.372 0.017722 *
# conditionVPA                            2.117e-03  7.316e-02  6.235e+01   0.029 0.977007
# cluster_membership_21                  -1.783e-01  8.134e-02  2.352e+04  -2.193 0.028352 *
# bin2:conditionVPA                       8.488e-03  4.501e-02  2.350e+04   0.189 0.850421
# bin3:conditionVPA                       8.432e-03  4.440e-02  2.351e+04   0.190 0.849378
# bin4:conditionVPA                       1.324e-02  4.358e-02  2.351e+04   0.304 0.761293
# bin5:conditionVPA                       1.476e-02  4.313e-02  2.352e+04   0.342 0.732241
# bin6:conditionVPA                      -7.289e-02  4.269e-02  2.352e+04  -1.707 0.087786 .
# bin2:cluster_membership_21              1.531e-02  1.071e-01  2.351e+04   0.143 0.886276
# bin3:cluster_membership_21              1.090e-01  1.040e-01  2.350e+04   1.047 0.294921
# bin4:cluster_membership_21              1.793e-01  1.026e-01  2.351e+04   1.747 0.080623 .
# bin5:cluster_membership_21              8.242e-02  9.684e-02  2.352e+04   0.851 0.394689
# bin6:cluster_membership_21              2.084e-01  9.876e-02  2.352e+04   2.111 0.034823 *
# conditionVPA:cluster_membership_21      1.433e-01  1.714e-01  2.350e+04   0.836 0.403138
# bin2:conditionVPA:cluster_membership_21 -2.622e-01  2.123e-01  2.349e+04  -1.235 0.216821
# bin3:conditionVPA:cluster_membership_21 -1.418e-01  1.985e-01  2.349e+04  -0.714 0.475064
# bin4:conditionVPA:cluster_membership_21 -3.007e-01  1.914e-01  2.350e+04  -1.571 0.116309
# bin5:conditionVPA:cluster_membership_21  4.990e-02  1.876e-01  2.350e+04   0.266 0.790257
# bin6:conditionVPA:cluster_membership_21 -1.648e-01  1.941e-01  2.350e+04  -0.849 0.395692

eta2_from_lrt <- function(lrt) {
  if (is.na(lrt$Chisq[2]) || is.na(lrt$Df[2])) return(NA_real_)
  lrt$Chisq[2] / (lrt$Chisq[2] + lrt$Df[2])
}

# cluster effect
model_no_cluster <- lmer(F0_Skewness ~ bin * condition + (1 | id), data = t_complete)
lrt_cluster <- anova(model_no_cluster, model_full)
print(lrt_cluster)
cat("eta²:", eta2_from_lrt(lrt_cluster))

# condition effect
model_no_condition <- lmer(F0_Skewness ~ bin * cluster_membership_2 + (1 | id), data = t_complete)
lrt_condition <- anova(model_no_condition, model_full)
print(lrt_condition)
cat("eta²:", eta2_from_lrt(lrt_condition))

# bin effect
model_no_bin <- lmer(F0_Skewness ~ condition * cluster_membership_2 + (1 | id), data = t_complete)
lrt_bin <- anova(model_no_bin, model_full)
print(lrt_bin)
cat("eta²:", eta2_from_lrt(lrt_bin))

# condition x cluster interaction
model_no_interaction <- lmer(F0_Skewness ~ bin + condition + cluster_membership_2 +
                               bin:condition + bin:cluster_membership_2 +
                               (1 | id),
                             data = t_complete)
lrt_inter <- anova(model_no_interaction, model_full)
print(lrt_inter)
cat("eta²:", eta2_from_lrt(lrt_inter))

# posthoc
print(pairs(emmeans(model_full, ~ cluster_membership_2), adjust = "tukey"))
print(pairs(emmeans(model_full, ~ condition), adjust = "tukey"))

# post-hoc comparisons not significant within clusters (cluster 0: p = 0.970, cluster 1: p = 0.959)
emm_inter <- emmeans(model_full, ~ condition * cluster_membership_2)
# clusters within condition
print(pairs(emm_inter, by = "condition", adjust = "tukey"))
# conditions within cluster
print(pairs(emm_inter, by = "cluster_membership_2", adjust = "tukey"))