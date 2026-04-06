# F0 magnitude mean - lmm model selection
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

vars_needed <- c("F0_Mag_Mean", "bin", "condition", "cluster_membership_2", "id")
t_complete <- t[complete.cases(t[, vars_needed]), ]
if (nrow(t_complete) == 0) stop("No complete cases for F0_Mag_Mean.")

# model selection
model_1 <- lmer(F0_Mag_Mean ~ bin * condition * cluster_membership_2 + (1 | id),
                data = t_complete)

model_2 <- lmer(F0_Mag_Mean ~ bin * condition + cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_2) # significant, retain model_1

model_3 <- lmer(F0_Mag_Mean ~ bin + condition * cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_3) # significant, retain model_1

model_4 <- lmer(F0_Mag_Mean ~ bin:condition * cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_4) # significant, retain model_1

model_5 <- lmer(F0_Mag_Mean ~ bin:condition:cluster_membership_2 + (1 | id),
                data = t_complete)
anova(model_1, model_5) # significant, retain model_1
# cluster 1: VPA initially higher F0 magnitude then progressive decline across session; CTRL opposite trajectory
# cluster 0: no significant condition differences (post-hoc p = 0.908) ... cluster 1: post-hoc p < 0.001

model_full <- model_1

print(summary(model_full))
cat("AIC:", AIC(model_full))

eta2_from_lrt <- function(lrt) {
  if (is.na(lrt$Chisq[2]) || is.na(lrt$Df[2])) return(NA_real_)
  lrt$Chisq[2] / (lrt$Chisq[2] + lrt$Df[2])
}

# cluster effect
model_no_cluster <- lmer(F0_Mag_Mean ~ bin * condition + (1 | id), data = t_complete)
lrt_cluster <- anova(model_no_cluster, model_full)
print(lrt_cluster)
cat("eta²:", eta2_from_lrt(lrt_cluster))

# condition effect
model_no_condition <- lmer(F0_Mag_Mean ~ bin * cluster_membership_2 + (1 | id), data = t_complete)
lrt_condition <- anova(model_no_condition, model_full)
print(lrt_condition)
cat("eta²:", eta2_from_lrt(lrt_condition))

# bin effect
model_no_bin <- lmer(F0_Mag_Mean ~ condition * cluster_membership_2 + (1 | id), data = t_complete)
lrt_bin <- anova(model_no_bin, model_full)
print(lrt_bin)
cat("eta²:", eta2_from_lrt(lrt_bin))

# condition x cluster interaction
model_no_interaction <- lmer(F0_Mag_Mean ~ bin + condition + cluster_membership_2 +
                               bin:condition + bin:cluster_membership_2 +
                               (1 | id),
                             data = t_complete)
lrt_inter <- anova(model_no_interaction, model_full)
print(lrt_inter)
cat("eta²:", eta2_from_lrt(lrt_inter))

# posthoc
print(pairs(emmeans(model_full, ~ cluster_membership_2), adjust = "tukey"))
print(pairs(emmeans(model_full, ~ condition), adjust = "tukey"))

emm_inter <- emmeans(model_full, ~ condition * cluster_membership_2)
# clusters within condition
print(pairs(emm_inter, by = "condition", adjust = "tukey"))
# conditions within cluster
print(pairs(emm_inter, by = "cluster_membership_2", adjust = "tukey"))