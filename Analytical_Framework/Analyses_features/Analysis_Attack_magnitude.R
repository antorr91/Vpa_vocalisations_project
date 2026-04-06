# Attack magnitude - lmm model selection
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

vars_needed <- c("Attack_magnitude", "bin", "condition", "cluster_membership_2", "id")
t_complete  <- t[complete.cases(t[, vars_needed]), ]
if (nrow(t_complete) == 0) stop("No complete cases for Attack_magnitude.")

# model selection
# full model: bin x condition x cluster
model_1 <- lmer(Attack_magnitude ~ bin * condition * cluster_membership_2 + (1 | id),
                data = t_complete)

# drop bin x cluster interaction, keep bin x condition
model_2 <- lmer(Attack_magnitude ~ bin * condition + cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_2) # significant, retain model_1


# drop bin x condition interaction, keep condition x cluster
model_3 <- lmer(Attack_magnitude ~ bin + condition * cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_3) # significant, retain model_1


# drop main effects, keep interactions only
model_4 <- lmer(Attack_magnitude ~ bin:condition * cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_4) # significant, retain model_1

# three-way only, no two-way
model_5 <- lmer(Attack_magnitude ~ bin:condition:cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_5, model_4) # significant, retain model_5
anova(model_1, model_5) # significant, retain model_1

# condition x cluster x bin interaction: chi2 = 556, p < 0.001

# cluster 0..VPA lower attack magnitude; cluster 1..VPA higher attack magnitude

# CTRL progressive increase across time bins; VPA consistently high with reduced temporal variability

# two-way interactions: bin x condition, bin x cluster, condition x cluster
model_twoway <- lmer(Attack_magnitude ~ bin * condition + bin * cluster_membership_2 +
                       condition * cluster_membership_2 + (1 | id),
                     data = t_complete)
# main effects only
model_main <- lmer(Attack_magnitude ~ bin + condition + cluster_membership_2 + (1 | id),
                   data = t_complete)
anova(model_main, model_twoway)
print(summary(model_twoway))
print(summary(model_main))

emmeans(model_1, ~ condition | cluster_membership_2 | bin)

print(summary(model_1))  # if full model retained
print(summary(model_4))  # if model_4 retained
print(summary(model_5))  # if model_5 retained
print(summary(model_3))

model_full <- model_1

print(summary(model_full))
cat("AIC:", AIC(model_full))

eta2_from_lrt <- function(lrt) {
  if (is.na(lrt$Chisq[2]) || is.na(lrt$Df[2])) return(NA_real_)
  lrt$Chisq[2] / (lrt$Chisq[2] + lrt$Df[2])
}

# cluster effect
model_no_cluster <- lmer(Attack_magnitude ~ bin * condition + (1 | id), data = t_complete)
lrt_cluster <- anova(model_no_cluster, model_full)
print(lrt_cluster)
cat("eta²:", eta2_from_lrt(lrt_cluster))

# condition effect
model_no_condition <- lmer(Attack_magnitude ~ bin * cluster_membership_2 + (1 | id), data = t_complete)
lrt_condition <- anova(model_no_condition, model_full)
print(lrt_condition)
cat("eta²:", eta2_from_lrt(lrt_condition))

# bin effect
model_no_bin <- lmer(Attack_magnitude ~ condition * cluster_membership_2 + (1 | id), data = t_complete)
lrt_bin <- anova(model_no_bin, model_full)
print(lrt_bin)
cat("eta²:", eta2_from_lrt(lrt_bin))

# condition x cluster interaction
model_no_interaction <- lmer(Attack_magnitude ~ bin + condition + cluster_membership_2 +
                               bin:condition + bin:cluster_membership_2 +
                               (1 | id),
                             data = t_complete)
lrt_inter <- anova(model_no_interaction, model_full)
print(lrt_inter)
cat("eta²:", eta2_from_lrt(lrt_inter))

# posthoc
print(pairs(emmeans(model_full, ~ cluster_membership_2), adjust = "tukey"))
print(pairs(emmeans(model_full, ~ condition), adjust = "tukey"))

# cluster 0: CTRL > VPA (p = 0.031)
# cluster 1: CTRL < VPA (p < 0.001)

emm_inter <- emmeans(model_full, ~ condition * cluster_membership_2)
# clusters within condition
print(pairs(emm_inter, by = "condition", adjust = "tukey"))
# conditions within cluster
print(pairs(emm_inter, by = "cluster_membership_2", adjust = "tukey"))