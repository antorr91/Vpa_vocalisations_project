# F2/F0 ratio - lmm model selection
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

vars_needed <- c("F2_F0_Ratio", "bin", "condition", "cluster_membership_2", "id")
t_complete <- t[complete.cases(t[, vars_needed]), ]
if (nrow(t_complete) == 0) stop("No complete cases for F2_F0_Ratio.")

# model selection
model_1 <- lmer(F2_F0_Ratio ~ bin * condition * cluster_membership_2 + (1 | id),
                data = t_complete)

model_2 <- lmer(F2_F0_Ratio ~ bin * condition + cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_2) # non significant (chi2 = 1.21, p = 0.944), drop three-way interaction

model_twoway <- lmer(F2_F0_Ratio ~ bin * condition + bin * cluster_membership_2 +
                       condition * cluster_membership_2 + (1 | id),
                     data = t_complete)
model_main <- lmer(F2_F0_Ratio ~ bin + condition + cluster_membership_2 + (1 | id),
                   data = t_complete)
anova(model_main, model_twoway) # non significant (chi2 = 10.60, p = 0.477), drop two-way interactions
# VPA uniformly affects F2/F0 ratio regardless of cluster or bin
# cluster effect: chi2 = 47.68, p < 0.001; condition effect: chi2 = 5.93, p = 0.015

model_full <- model_main

print(summary(model_full))
cat("AIC:", AIC(model_full))

eta2_from_lrt <- function(lrt) {
  if (is.na(lrt$Chisq[2]) || is.na(lrt$Df[2])) return(NA_real_)
  lrt$Chisq[2] / (lrt$Chisq[2] + lrt$Df[2])
}

# cluster effect
model_no_cluster <- lmer(F2_F0_Ratio ~ bin + condition + (1 | id), data = t_complete)
lrt_cluster <- anova(model_no_cluster, model_full)
print(lrt_cluster)
cat("eta²:", eta2_from_lrt(lrt_cluster))

# condition effect
model_no_condition <- lmer(F2_F0_Ratio ~ bin + cluster_membership_2 + (1 | id), data = t_complete)
lrt_condition <- anova(model_no_condition, model_full)
print(lrt_condition)
cat("eta²:", eta2_from_lrt(lrt_condition))

# bin effect
model_no_bin <- lmer(F2_F0_Ratio ~ condition + cluster_membership_2 + (1 | id), data = t_complete)
lrt_bin <- anova(model_no_bin, model_full)
print(lrt_bin)
cat("eta²:", eta2_from_lrt(lrt_bin))

# posthoc
print(pairs(emmeans(model_full, ~ cluster_membership_2), adjust = "tukey"))
print(pairs(emmeans(model_full, ~ condition), adjust = "tukey"))